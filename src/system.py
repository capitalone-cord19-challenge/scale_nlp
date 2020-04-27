import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import torch

from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel

from pyserini.search import pysearch

from collections import namedtuple

from src.ranking_model import  create_ranking_feature, BertRankingModel
from src.qa_model import BertQAModel, ExampleQA, create_qa_features, find_best_prediction
from src.loaders import  rankingloader
from src.utils import tensor_to_list

QUESTIONS = ["who", "what", "where", "why", "why", "is", "are", "whose", "does", "do", "can", "could", "would", "should",
             "was", "were", "did", "when"]

class ScaleNLP(object):
    def __init__(self, opt):
        self.number_docs = opt.number_docs

        self.max_sequence = opt.max_sequence
        self.max_query = opt.max_query
        self.max_answer = opt.max_answer
        self.stride = opt.stride
        self.ranking_batchsize = opt.ranking_batchsize
        self.qa_batchsize = opt.qa_batchsize
        self.number_paragraphs = opt.number_paragraphs
        self.top = opt.top
        self.index = opt.index

        self.tokenizer = BertTokenizer.from_pretrained(opt.rank_path)


        self.rank_model_config = BertConfig.from_pretrained(opt.rank_path)
        self.rank_model = BertRankingModel.from_pretrained(opt.rank_path, config=self.rank_model_config)
        self.rank_model.to(opt.device)

        self.qa_model_config = BertConfig.from_pretrained(opt.qa_path)
        self.qa_model = BertQAModel.from_pretrained(opt.qa_path, config=self.qa_model_config)
        self.qa_model.to(opt.device)

        self.device = opt.device


    def question_identification(self, query):
        """
        dumb way to check if a query iss a question
        :param query: text
        :return: True or False if first word looks like a question word
        """
        return query.split()[0].lower() in QUESTIONS

    def query_processor(self, query):
        """
        Using aserini for search as a lite weight replacement for elasticsearch
        :param query: query from user
        :return: a function
        """

        #Retrieve set of candidate documents
        searcher = pysearch.SimpleSearcher(self.index)
        results = searcher.search(query, self.number_docs)
        documents = []
        history = set()

        for res in results:
            did = res.docid
            history.add(did)
            title = res.lucene_document.get("title")
            text = res.lucene_document.get("abstract")
            inp_dict = {"id":did, "title":title, "text":text}
            documents.append(inp_dict)
            for i, para in enumerate(res.contents.split("\n")):
                if i == 0 or i == 1:
                    continue
                else:
                    documents.append({"id":did, "title":title, "text":para})


        return self.processor(query, documents)

    def processor(self, query, documents):
        query_tokens = self.tokenizer.tokenize(query)
        if self.question_identification(query):
            search_results =  self.ranking_processor(query_tokens, documents)
            qa_results =self.qa_processor(query_tokens, documents)
            return search_results, qa_results
        return self.ranking_processor(query_tokens, documents)

    def ranking_processor(self, query_tokens, documents):
        """

        :param query: original query from user
        :param documents: set of c andidate documents from annserini
        :return: ranked documents
        """
        ranking_features = []

        #Convert documents and query into ranking feature space
        self.query_idx = 0
        for (doc_idx, doc) in enumerate(documents):
            ranking_features.extend(create_ranking_feature(query_tokens, doc['text'], self.query_idx, doc_idx,
                                                           self.tokenizer, self.max_sequence, self.max_query,
                                                           self.stride))
        #Create Generator of batches
        ranking_data = rankingloader(ranking_features, self.ranking_batchsize)

        rank_dict = namedtuple("rankingresults", ["doc_idx", "title", "text", "score"])

        self.ranking_results = []
        #batch data into model and rerank result based on score
        for g, batch in enumerate(ranking_data):
            self.rank_model.eval()
            self.query_idx, doc_idx = batch[:2]
            batch = tuple(t.to(self.device) for t in batch[2:])
            (dii, dim, dsi) = batch
            with torch.no_grad():
                scores, _ = self.rank_model(dii, dim, dsi)
            doc_scores = tensor_to_list(scores)

            for (did, score) in zip(doc_idx, doc_scores):
                self.ranking_results.append(
                    rank_dict(doc_idx=did, title=documents[did]['title'], text= documents[did]['text'], score=score)
                )

        self.ranking_results = sorted(self.ranking_results, key=lambda x: x.score, reverse=True)

        #Remove duplicate results from search and assign payload to search results
        search_results = []
        unique_titles = set()
        for res in self.ranking_results:
            if res.title not in unique_titles:
                unique_titles.add(res.title)
                payload = {"title":res.title, "text":res.text, "score":res.score}
                search_results.append(payload)

        return search_results

    def qa_processor(self, query, query_tokens, documents, ranking_results):

        paragraphs = []
        index_paragraphs = []

        if self.number_paragraphs > len(self.ranking_results):
            self.number_paragraphs = len(self.ranking_results)

        for idx in range(self.number_paragraphs):
            paragraphs.append(self.ranking_results[idx].text)
            index_paragraphs.append(self.ranking_results[idx].doc_idx)

        examples = ExampleQA(self.query_idx, query, paragraphs)
        features = create_qa_features(examples=[examples], tokenizer=self.tokenizer, max_seq_length=self.max_sequence,
                                      max_query=self.max_query)
        feature = features[0]
        self.qa_model.eval()
        input_ids = torch.tensor([feature.input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([feature.input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            out = self.qa_model(input_ids=input_ids.view(-1, input_ids.shape[2]),
                                attention_mask=input_mask.view(-1, input_mask.shape[2]),
                                token_type_ids=segment_ids.view(-1, segment_ids.shape[2]))
            start, end = out[0], out[1]
            start_logits = start.view(self.qa_batchsize, 3840)
            end_logits = end.view(self.qa_batchsize, 3840)

        starting_list = tensor_to_list(start_logits[0])
        ending_list = tensor_to_list(end_logits[0])

        best_prediction = find_best_prediction(feature, starting_list, ending_list, self.top, self.max_sequence,
                                                self.max_answer)
        for prediction in best_prediction:
            if prediction['doc_id'] != -1:
                doc_idx = index_paragraphs[prediction['doc_id']]
                prediction["title"] = documents[doc_idx]["title"]
            else:
                prediction["title"] = ""
            prediction['doc_id'] = None

            results = best_prediction
            return results















