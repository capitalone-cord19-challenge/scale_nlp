
class InputFeatures(object):
    """
    Generic Input Class for Tensorflow
    """

    def __init__(self, query_id, tokens, input_ids, input_mask, segment_ids):
        self.query_id = query_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids




def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length, n_paragraph=10,
                                 cls_token_at_end=False, cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment=0,
                                 pad_token_segment_id=0, mask_padding_with_zero=True, sequence_is_doc=False):
    """Generic way to laod batches"""
    features = []

    for (idx, example) in enumerate(examples):
        q_tokens = tokenizer.tokenize(example.query)
        if len(q_tokens) > max_query_length:
            q_tokens = q_tokens[:max_query_length]

        max_tokens = max_seq_length - len(q_tokens) - 3
        tokens_lst = []
        input_ids_lst = []
        input_mask_lst = []
        segment_ids_lst = []

        max_score = 0.0
        start_pos = None
        end_pos = None
        for n in range(n_paragraph):
            paragraph = example.paragraph[n] if n < len(example.paragraph) else ""

            tokens = []
            segment_ids = []
            pos_mask = []

            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment)
                pos_mask.append(0)
                cls_idx = 0

            if not sequence_is_doc:
                tokens += q_tokens
                segment_ids += [sequence_a_segment_id] * len(q_tokens)
                pos_mask += [1] * len(q_tokens)

                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                pos_mask.append(1)

            paragraph_tokens = tokenizer.tokenize(paragraph)[:max_tokens]

            for idx in range(len(paragraph_tokens)):
                tokens.append(paragraph_tokens[idx])
                if not sequence_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                pos_mask.append(0)

            if sequence_is_doc:
                tokens.append(sep_token)
                segment_ids += [sequence_b_segment_id] * len(q_tokens)
                pos_mask += [1] * len(q_tokens)

            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment)
                pos_mask.append(0)
                cls_index = len(tokens) - 1

            input_ids = tokenizer.convert_to_tokens_to_ids(tokens)

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                pos_mask.append(1)

            features.append(InputFeatures(query_id=example.query_id, tokens=tokens_lst,
                                          input_ids=input_ids_lst, input_mask=input_mask_lst,
                                          segment_ids=segment_ids_lst))
            return features



