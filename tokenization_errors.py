from read_gate_output import *
from args import args
from util import *
from train_annos import *
from nn_utils import *
from transformers import AutoTokenizer

tweet_to_annos = get_annos_dict(args['annotations_file_path'])
sample_to_token_data = get_train_data(args['training_data_folder_path'])
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])


def get_tokenization_errors(sample_to_token_data, bert_tokenizer):
    errors = []
    for sample_id in sample_to_token_data:
        token_data = sample_to_token_data[sample_id]
        tokens = get_token_strings(token_data)
        labels = get_labels(token_data)
        offsets_list = get_token_offsets(token_data)
        assert len(tokens) == len(labels) == len(offsets_list)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512)
        expanded_labels = expand_labels(batch_encoding, labels)
        expanded_labels = [0 if label == 'o' else 1 for label in expanded_labels]
        label_spans_token_index = get_spans_from_seq_labels(expanded_labels, batch_encoding)
        label_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                                    label_spans_token_index]
        gold_annos = tweet_to_annos.get(sample_id, [])
        gold_spans_char_offsets = [(anno['begin'], anno['end']) for anno in gold_annos]
        label_spans_set = set(label_spans_char_offsets)
        gold_spans_set = set(gold_spans_char_offsets)
        if not label_spans_set == gold_spans_set:
            errors.append((sample_id, label_spans_set, gold_spans_set))
            print(sample_id, label_spans_set, gold_spans_set)
    return errors


errors = get_tokenization_errors(sample_to_token_data, bert_tokenizer)
