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

QUESTIONS = ["who", "what", "where", "why", "why", "is", "are", "whose", "does", "do", "can", "could", "would", "should",
             "was", "were", "did", "when"]

def to_list(tensor):
    return tensor.detach().cpu().tolist()

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
        return query.split()[0].lower() in QUESTIONS

    def query_processor(self, query):

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
        ranking_features = []

        query_idx = 0
        query_tokens = self.tokenizer.tokenize(query)
        for (doc_idx, doc) in enumerate(documents):
            ranking_features.extend(create_ranking_feature(query_tokens, doc['text'], query_idx, doc_idx,
                                                           self.tokenizer, self.max_sequence, self.max_query,
                                                           self.stride))

        ranking_data = rankingloader(ranking_features, self.ranking_batchsize)

        rank_dict = namedtuple("rankingresults", ["doc_idx", "title", "text", "score"])

        ranking_results = []

        for g, batch in enumerate(ranking_data):
            self.rank_model.eval()
            query_idx, doc_idx = batch[:2]
            batch = tuple(t.to(self.device) for t in batch[2:])
            (dii, dim, dsi) = batch
            with torch.no_grad():
                scores, _ = self.rank_model(dii, dim, dsi)
            doc_scores = to_list(scores)

            for (did, score) in zip(doc_idx, doc_scores):
                ranking_results.append(
                    rank_dict(doc_idx=did, title=documents[did]['title'], text= documents[did]['text'], score=score)
                )

        ranking_results = sorted(ranking_results, key=lambda x: x.score, reverse=True)

        search_results = []
        unique_titles = set()
        for res in ranking_results:
            if res.title not in unique_titles:
                unique_titles.add(res.title)
                payload = {"title":res.title, "text":res.text, "score":res.score}
                search_results.append(payload)

        return search_results










