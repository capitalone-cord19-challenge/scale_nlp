from transformers import BertPreTrainedModel, BertModel
from torch import nn

from collections import namedtuple

class RankingFeature:
    def __init__(self, qid, did, dii, dim, dsi):
        """

        :param qid: query id
        :param did: document id
        :param dii: document input id
        :param dim: document input mask
        :param dsi: document segment ids
        """
        self.qid = qid
        self.did = did
        self.dii = dii
        self.dim = dim
        self.dsi = dsi

def pad_sequence(q_tokens, p_tokens, tokenizer, max_seq_length):
    """

    :param q_tokens: query that has already been tokenized
    :param p_tokens: position that has already been tokenized
    :param tokenizer: what we use to tokenize stuff
    :param max_seq_length: how long a sequence can be
    :return: input text as ints, masking, and segment
    """
    ##Add CLS Tag to tokens
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in q_tokens:
        tokens.append(token)
        segment_ids.append(0)
    #ADDING SEP TAG TO TOKENS
    tokens.append("[SEP]")
    segment_ids.append(0)
    #ADDING POSITION TOKENs
    for token in p_tokens:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    #Convert tokens to index
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Pad until max length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)



    return input_ids, input_mask, segment_ids

def create_ranking_feature(query_tokens, document, qid, did, tokenizer, max_seq, max_query, stride):
    """

    :param query_tokens: already tokenize query
    :param document: document text
    :param qid: query id
    :param did: document id
    :param tokenizer: token function
    :param max_seq: max sequence length
    :param max_query: max query length
    :return: RankingFeature
    """
    query_tokens = query_tokens[:max_query]

    #need to account for CLS and SEPS
    max_document = max_seq - len(query_tokens) - 3
    document_tokens = tokenizer.tokenize(document)

    document_span = namedtuple('span', ["start", "length"])
    spans = []
    offset = 0

    #break document into bite size tasty documents
    while offset < len(document_tokens):
        length = len(document_tokens) - offset
        if length > max_document:
            length = max_document
        spans.append(document_span(start=offset, length=length))

        if offset + length == len(document_tokens):
            break
        offset += min(length, stride)

    features = []
    ##Pad reach bite size documents and transform into a ranking  set
    for (span_index, span) in spans:
        doc_tokens = document_tokens[span.start:span.start + span.length]
        dii, dim, dsi = pad_sequence(query_tokens, doc_tokens, tokenizer, max_seq)

        feature_space = RankingFeature(
            qid = qid,
            did = did,
            dii = dii,
            dim = dim,
            dsi = dsi
        )
        features.append(feature_space)

    return features

class BertRankingModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, inst, token, mask):
        output = self.bert(inst, token_type_ids=token, attention_mask=mask)
        result = self.dense(output[1])[:,1]

        return result, output[1]







