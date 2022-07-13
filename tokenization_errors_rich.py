from read_gate_output import *
from args import args
from util import *
from train_annos import *
from nn_utils import *
from transformers import AutoTokenizer
import csv

tweet_to_annos = get_annos_dict(args['gold_file_path'])
sample_to_token_data = get_train_data('/home/harsh/projects/smm4h-social-dis-ner/preprocessing-test/train-dis-gaz'
                                      '-final/train')
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
train_raw_data = get_raw_train_data()


def get_tokenization_errors(sample_to_token_data, bert_tokenizer):
    errors = []
    num_gold = 0
    num_mistake = 0
    for sample_id in sample_to_token_data:
        token_data = sample_to_token_data[sample_id]
        tokens = get_token_strings(token_data)
        gold_annos = tweet_to_annos.get(sample_id, [])
        labels = get_labels_rich(token_data, gold_annos)
        # print(sample_id, labels)
        offsets_list = get_token_offsets(token_data)
        assert len(tokens) == len(labels) == len(offsets_list)
        # print(tokens)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512)
        expanded_labels = expand_labels_rich(batch_encoding, labels)
        label_spans_token_index = get_spans_from_seq_labels_3_classes(expanded_labels, batch_encoding)
        label_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                                    label_spans_token_index]
        gold_spans_char_offsets = [(anno['begin'], anno['end']) for anno in gold_annos]
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
with open('tokenization_errors_latest_rich_labels.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    header = ['sample_id', 'start', 'end', 'extraction', 'context']
    writer.writerow(header)
    for sample_id, (begin, end) in errors:
        sample_data = train_raw_data[sample_id]
        begin_soft = int(begin) - 20
        end_soft = int(end) + 20
        if begin_soft < 0:
            begin_soft = 0
        row = [sample_id, begin, end, sample_data[int(begin):int(end)], sample_data[begin_soft:end_soft]]
        writer.writerow(row)
