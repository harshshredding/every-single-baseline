import torch
from transformers import AutoTokenizer, AutoModel

from util import read_disease_gazetteer, get_tweet_data, get_spans_from_seq_labels_3_classes
from read_gate_output import *
from args import args
import numpy as np
from train_annos import get_annos_dict
from nn_utils import *
from torch.nn import TransformerEncoderLayer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
model = AutoModel.from_pretrained('xlm-roberta-large')
tweet_id = '1374005428351295491'
tweet_to_annos = get_annos_dict(args['gold_file_path'])
train_data = sample_to_token_data_train = get_train_data(args['training_data_folder_path'])
sample_data = train_data[tweet_id]
tokens = get_token_strings(sample_data)
labels = get_labels(sample_data)
offsets_list = get_token_offsets(sample_data)
annos = tweet_to_annos[tweet_id]
new_labels = get_labels_rich(sample_data, annos)
silver_labels_one_hot = get_silver_dis_one_hot(sample_data)
silver_labels = get_silver_dis_labels(sample_data)
batch_encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                            add_special_tokens=False, truncation=True, max_length=512)
extract_expanded_labels(sample_data, batch_encoding, )