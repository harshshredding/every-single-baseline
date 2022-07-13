from transformers import AutoTokenizer

from util import read_disease_gazetteer, get_tweet_data, get_spans_from_seq_labels_3_classes
from read_gate_output import *
from args import args
import numpy as np
from train_annos import get_annos_dict
from nn_utils import *

bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
from evaluation_util import score_task2

# tweet_data = get_tweet_data('/home/claclab/harsh/smm4h/smm4h-2022-social-dis-ner/socialdisner-data/train-valid-txt'
#                             '-files/validation')
# pred_file_path = './submissions/validation_predictions.tsv'
# gold_file_path = './validation_entities.tsv'
# score_task2(pred_file_path, gold_file_path, tweet_data, './results.txt')
tweet_id = '1374005428351295491'
tweet_to_annos = get_annos_dict(args['gold_file_path'])
train_data = sample_to_token_data_train = get_train_data(args['training_data_folder_path'])
sample_data = train_data[tweet_id]
tokens = get_token_strings(sample_data)
labels = get_labels(sample_data)
offsets_list = get_token_offsets(sample_data)
annos = tweet_to_annos[tweet_id]
new_labels = get_labels_rich(sample_data, annos)
print(new_labels)
batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                add_special_tokens=False, truncation=True, max_length=512)
new_labels_expanded = expand_labels_rich(batch_encoding, new_labels)
print(new_labels_expanded)
print(annos)
label_spans_token_index = get_spans_from_seq_labels_3_classes(new_labels_expanded, batch_encoding)
label_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                            label_spans_token_index]
print(label_spans_char_offsets)
