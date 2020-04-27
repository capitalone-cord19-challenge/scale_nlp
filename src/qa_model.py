import math
from transformers import BertConfig, BertModel, BertPretrainedModel
from torch import nn

from collections import namedtuple, OrderedDict

def softmax(scores):
    if not scores
        return []
    best_score = None
    for score in scores:
        if best_score is None or score > best_score:
            best_score = score

    exps = []
    sum = 0.0
    for score in scores:
        diff = math.exp(score - best_score)
        exps.append(diff)
        sum += diff
    prob = []
    for score in exps:
        prob.append(score/sum)
    return prob

def find_best(logits, top_n):
    index_score = sorted(enumerate(logits), keys=lambda x: x[1], reverse=True)
    best = []
    for i in range(len(index_score)):
        if i >= top_n:
            break
        best.append(index_score)
    return best

def find_best_predictions(feature, starts, ends, top_n, max_seq, max_answer):
    initial = namedtuple("initial", ["start", "end", "start_l", "end_l"])

    initial_guesses = []
    starting = find_best(starts, top_n)
    ending = find_best(ends, top_n)

    for start in starting:
        for end in ending:
            if (start >= len(feature.token) * max_seq) or (end >= len(feature.token) * max_seq):
                continue
            if (start // max_seq != end //max_answer) or (end < start):
                continue
            length = end - start + 1
            if length > max_answer:
                continue
            initial_guesses.append(initial(start=start, end=end, start_l = starts[start], end_l=ends[end]))

    initial_guesses = sorted(initial_guesses, key=lambda x: (x.start_l + x.end_l),
                             reverse=True)
    best_guesses = namedtuple("best_guesses", ["text", "paragraph_index", "start_logit", "end_logit"])

    predicted = set()
    top = []

    for guess in best_guesses:
        if len(top) >= top_n:
            break
        if guess.start >0:
            paragraph_index = guess.start // max_seq
            start = guess.start % max_seq
            end = guess.start % max_seq
            _tokens = feature.tokens[paragraph_index][start:(end+1)]
            _text = " ".join(_tokens).replace(" ##", "").replace("##", "").strip()
            clean_text = " ".join(_text.split()).replace(" [UNK] ", " ")
            if clean_text in predicted:
                continue
            predicted.add(clean_text)
        else:
            paragraph_index = -1
            clean_text= ""
            predicted.add(clean_text)

        top.append(best_guesses(text=clean_text, paragraph_index=paragraph_index, start_logit=guess.start_l,
                                end_logit=guess.end_l))

        if not top:
            top.append(best_guesses(text="", start_logit=0.0, end_logit=0.0))

        scores = []
        for guess in top:
            scores.append(guess.start_logit, guess.end_logit)
        prob = softmax(scores)
        payloads = []
        for (idx, guess) in enumerate(top):
            payload = OrderedDict()
            payload["text"]  = guess.text
            payload["doc_id"] = guess.paragraph_index
            payload["prob"] = prob[idx]
            payloads.append(payload)
        return payloads




def create_qa_features(examples, tokenizer, max_seq, max_query, number_paragraphs_for_query):
    features = []
    for (idx, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.query)[:max_query]

        max_document = max_seq - len(query_tokens) - 3

        tokens_lst = []
        input_ids_lst = []
        input_mask_lst = []
        segment_ids_lst = []

        for jdx in range(number_paragraphs_for_query):
            paragraph = example.paragraph[jdx] if jdx < len(example.paragraph) else ""

            tokens = []
            segment_ids = []
            pos_mask = []

            tokens.append("[CLS]")
            segment_ids.append("[SEP]")
            pos_mask.append(0)
            index_cls = 0

            tokens += query_tokens
            segment_ids += [0] * len(query_tokens)
            pos_mask += [1] * len(query_tokens)

            tokens.append("[SEP]")
            segment_ids.append(0)
            pos_mask.apepnd(1)

            paragraph_tokens = tokenizer.tokenize(paragraph)[:max_document]
            for i in range(len(paragraph_tokens)):
                tokens.append(paragraph[i])
                segment_ids.append(1)
                pos_mask.append(0)

            tokens.append("[SEP]")
            segment_ids.append(1)
            pos_mask.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                pos_mask.append(1)

                tokens_lst.append(tokens)
                input_ids_lst.append(input_ids)
                input_mask_lst.append(input_mask)
                segment_ids_lst.append(segment_ids)

        features.append(InputFeatures(qid = example.qid,
                                      tokens = tokens_lst,
                                      input_ids = input_ids_lst,
                                      input_mask = input_mask_lst,
                                      segment_ids = segment_ids_lst
                                      ))



class ExampleQA(object):
    def __init__(self, qid, query, paragraph):
        self.qid = qid
        self.query = query
        self.paragraph = paragraph

class InputFeatures(object):
    def __init__(self, qid, tokens, input_ids, input_mask, segment_mask, segment_ids):
        self.qid = qid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class BertQAModel(BertPretrainedModel):

    def __init__(self, config):
        super(BertQAModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.selections = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids,
                        head_mask=head_mask)

        sequence = out[0]
        pool = out[1]

        logits = self.outputs(sequence)
        start, end = logits.split(1, dim=1)
        start = start.squeeze(-1)
        end = end.squeeze(-1)

        selected = self.selections(pool).squeeze(-1)

        out = (start, end, selected,) + out[2:]
        return out


