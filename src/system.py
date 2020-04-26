import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import torch

from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel

from pyserini.search import pysearch

from collections import namedtuple

from src.ranking_model import  create_ranking_feature, BertRankingModel
from src.loaders import  rankingloader
from src.utils import tensor_to_list

QUESTIONS = ["who", "what", "where", "why", "why", "is", "are", "whose", "does", "do", "can", "could", "would", "should",
             "was", "were", "did", "when"]

class ScaleNLP(object):
    def __init__(self, opt):
        self.number_docs = opt.number_docs

        self.max_sequence = opt.max_sequence
        self.max_query = opt.max_query
        self.stride = opt.stride
        self.ranking_batchsize = opt.ranking_batchsize
        self.index = opt.index

        self.tokenizer = BertTokenizer.from_pretrained(opt.qa_path, do_lower_case = opt.lower)


        self.rank_model_config = BertConfig.from_pretrained(opt.rank_path)
        self.rank_model = BertRankingModel.from_pretrained(opt.rank_path, config=self.rank_model_config)
        self.rank_model.to(opt.device)

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
        """

        :param query: original query from user
        :param documents: set of candidate documents from anserini
        :return: ranked documents
        """
        ranking_features = []

        #Convert documents and query into ranking feature space
        query_idx = 0
        query_tokens = self.tokenizer.tokenize(query)
        for (doc_idx, doc) in enumerate(documents):
            ranking_features.extend(create_ranking_feature(query_tokens, doc['text'], query_idx, doc_idx,
                                                           self.tokenizer, self.max_sequence, self.max_query,
                                                           self.stride))
        #Create Generator of batches
        ranking_data = rankingloader(ranking_features, self.ranking_batchsize)

        rank_dict = namedtuple("rankingresults", ["doc_idx", "title", "text", "score"])

        ranking_results = []
        #batch data into model and rerank result based on score
        for g, batch in enumerate(ranking_data):
            self.rank_model.eval()
            query_idx, doc_idx = batch[:2]
            batch = tuple(t.to(self.device) for t in batch[2:])
            (dii, dim, dsi) = batch
            with torch.no_grad():
                scores, _ = self.rank_model(dii, dim, dsi)
            doc_scores = tensor_to_list(scores)

            for (did, score) in zip(doc_idx, doc_scores):
                ranking_results.append(
                    rank_dict(doc_idx=did, title=documents[did]['title'], text= documents[did]['text'], score=score)
                )

        ranking_results = sorted(ranking_results, key=lambda x: x.score, reverse=True)

        #Remove duplicate results from search and assign payload to search results
        search_results = []
        unique_titles = set()
        for res in ranking_results:
            if res.title not in unique_titles:
                unique_titles.add(res.title)
                payload = {"title":res.title, "text":res.text, "score":res.score}
                search_results.append(payload)

        return search_results










