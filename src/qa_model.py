import argparse
import collections
import json
import math
import os
import torch

import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel


class ExampleQA(object):
    """
    Generic Example Holder
    """

    def __init__(self,
                 qid,
                 query,
                 paragraphs):
        self.qid = qid
        self.query = query
        self.paragraphs = paragraphs


class InputFeatures(object):
    """
    Generic Input for single row
    """

    def __init__(self,
                 qid,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.qid = qid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def create_qa_features(examples, tokenizer, max_sequence,
                                 max_query,
                                 num_paragraphs=10,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    """
    
    :param examples: List of QAExamples
    :param tokenizer: tokenizer from your model
    :param max_sequence: maximum length of sequence
    :param max_query: maximum length of query
    :param num_paragraphs: number of paragraphs
    :param cls_token_at_end: cls token at end of sequence
    :param cls_token: special char for cls token
    :param sep_token: special char for spe token
    :param pad_token: special char for pad token
    :param sequence_a_segment_id: id for sequence a usually 0
    :param sequence_b_segment_id: id for squence b usually 1
    :param cls_token_segment_id: token for cls usually 0
    :param pad_token_segment_id: token for pad usually 0
    :param mask_padding_with_zero: padding defaults as 0 but if no padding can be 1
    :param sequence_a_is_doc: is input a long form document?
    :return: 
    """
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.query)

        query_tokens = query_tokens[:max_query]

        max_document = max_sequence - len(query_tokens) - 3

        tokens_list = []
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []

        for idx in range(num_paragraphs):
            paragraph = example.paragraphs[idx] if idx < len(example.paragraphs) else ''

            tokens = []
            segment_ids = []


            pos_mask = []

            #  add cls at beginning 
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                pos_mask.append(0)

            # check if query
            if not sequence_a_is_doc:

                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                pos_mask += [1] * len(query_tokens)

            
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                pos_mask.append(1)

            paragraph_tokens = tokenizer.tokenize(paragraph)[:max_document]

            # Paragraph
            for i in range(len(paragraph_tokens)):
                tokens.append(paragraph_tokens[i])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                pos_mask.append(0)

            if sequence_a_is_doc:

                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                pos_mask.append(1)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                pos_mask += [1] * len(query_tokens)

            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            pos_mask.append(1)

            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                pos_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)


            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


            while len(input_ids) < max_sequence:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                pos_mask.append(1)


            tokens_list.append(tokens)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        features.append(
            InputFeatures(
                qid=example.qid,
                tokens=tokens_list,
                input_ids=input_ids_list,
                input_mask=input_mask_list,
                segment_ids=segment_ids_list))

    return features


class BertQAModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertQAModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.selected_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        selected_logits = self.selected_outputs(pooled_output).squeeze(-1)

        outputs = (start_logits, end_logits, selected_logits,) + outputs[2:]

        return outputs


def best_indexes(logits, num_best):
    """
    
    :param logits: logits from models
    :param num_best: how many do you want
    :return: 
    """
    indexscore = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    indexes = []
    for i in range(len(indexscore)):
        if i >= num_best:
            break
        indexes.append(indexscore[i][0])
    return indexes


def softmax(scores):
    """
    
    :param scores: list of scores
    :return: 
    """
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def find_best_prediction(feature, start_logits, end_logits, num_best, max_sequence, max_answer):
    """

    :param feature: feature class
    :param start_logits: probability where start is
    :param end_logits:  probability where end is
    :param num_best: how many do we want
    :param max_sequence: how long is the squence
    :param max_answer: how long is the answer
    :return:
    """
    initial = collections.namedtuple(
        "initial",
        ["start_index", "end_index", "start_logit", "end_logit"])

    initial_predictions = []
    start_indexes = best_indexes(start_logits, num_best)
    end_indexes = best_indexes(end_logits, num_best)
    ##Error handlers
    for start_index in start_indexes:
        for end_index in end_indexes:

            if start_index >= len(feature.tokens) * max_sequence:
                continue
            if end_index >= len(feature.tokens) * max_sequence:
                continue
            if start_index // max_sequence != end_index // max_sequence:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer:
                continue
            initial_predictions.append(
                initial(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index]))
    prelim_predictions = sorted(
        initial_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    best_prediction = collections.namedtuple(  # pylint: disable=invalid-name
        "bestprediction", ["text", "paragraph_index", "start_logit", "end_logit"])

    history = set()
    top = []
    for pred in prelim_predictions:
        if len(top) >= num_best:
            break
        if pred.start_index > 0:  # this is a non-null prediction
            paragraph_index = pred.start_index // max_sequence
            start_index = pred.start_index % max_sequence
            end_index = pred.end_index % max_sequence
            _tokens = feature.tokens[paragraph_index][start_index:(end_index + 1)]
            _text = " ".join(_tokens).replace(" ##", "").replace("##", "").strip()


            cleantext = " ".join(_text.split()).replace(" [UNK] ", " ")
            if cleantext in history:
                continue

            history.add(cleantext)
        else:
            paragraph_index = -1
            cleantext = ""
            history.add(cleantext)

        top.append(
            best_prediction(
                text=cleantext,
                paragraph_index=paragraph_index,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))


    if not top:
        top.append(
            best_prediction(text="", start_logit=0.0, end_logit=0.0))


    total_scores = []
    for entry in top:
        total_scores.append(entry.start_logit + entry.end_logit)
    probs = softmax(total_scores)

    payloads = []
    for (i, entry) in enumerate(top):
        payload = collections.OrderedDict()
        payload["text"] = entry.text
        payload["doc_id"] = entry.paragraph_index
        payload["probability"] = probs[i]
        payloads.append(payload)

    return payloads
