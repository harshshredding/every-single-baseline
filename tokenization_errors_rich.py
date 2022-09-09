import sys

from util import *
from train_annos import *
from nn_utils import *
from transformers import AutoTokenizer
import csv

sample_to_annos = get_train_annos_dict()
sample_to_token_data = get_train_data()
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])


def get_tokenization_errors(sample_to_token_data, bert_tokenizer):
    errors = []
    num_gold = 0
    num_mistake = 0
    for sample_id in sample_to_token_data:
        token_data = sample_to_token_data[sample_id]
        tokens = get_token_strings(token_data)
        gold_annos = sample_to_annos.get(sample_id, [])
        offsets_list = get_token_offsets(token_data)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512)
        expanded_labels = extract_expanded_labels(token_data, batch_encoding, gold_annos)
        label_spans_token_index = get_spans_from_seq_labels_3_classes(expanded_labels, batch_encoding)
        label_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1], span[2]) for span in
                                    label_spans_token_index]
        gold_spans_char_offsets = [(anno.begin_offset, anno.end_offset, anno.label_type) for anno in gold_annos]
        label_spans_set = set(label_spans_char_offsets)
        gold_spans_set = set(gold_spans_char_offsets)
        mistake_spans = gold_spans_set.difference(label_spans_set)
        num_gold += len(gold_spans_set)
        num_mistake += len(mistake_spans)
        for mistake_span in mistake_spans:
            errors.append((sample_id, mistake_span))
    print("percentage errors: ", num_mistake / num_gold)
    return errors


errors = get_tokenization_errors(sample_to_token_data, bert_tokenizer)
with open('tokenization_errors_rich.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    header = ['sample_id', 'start', 'end', 'extraction']
    writer.writerow(header)
    for sample_id, (mistake_begin, mistake_end, entity_type) in errors:
        token_data_list = sample_to_token_data[sample_id]
        tokens = get_token_strings(token_data_list)
        offsets = get_token_offsets(token_data_list)
        assert len(tokens) == len(offsets)
        extraction = []
        for i, (start_offset, end_offset) in enumerate(offsets):
            if start_offset >= mistake_begin and end_offset <= mistake_end:
                extraction.append(tokens[i])
        extraction = ' '.join(extraction)
        row = [sample_id, mistake_begin, mistake_end, extraction]
        writer.writerow(row)
