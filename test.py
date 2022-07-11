from util import read_disease_gazetteer, get_tweet_data
from read_gate_output import *
from args import args
import numpy as np
from nn_utils import *
from evaluation import score_task2

# tweet_data = get_tweet_data('/home/claclab/harsh/smm4h/smm4h-2022-social-dis-ner/socialdisner-data/train-valid-txt'
#                             '-files/validation')
# pred_file_path = './submissions/validation_predictions.tsv'
# gold_file_path = './validation_entities.tsv'
# score_task2(pred_file_path, gold_file_path, tweet_data, './results.txt')

train_data = sample_to_token_data_train = get_train_data(args['training_data_folder_path'])
sample_data = train_data['1092477766434471936']
tokens = get_token_strings(sample_data)
labels = get_labels(sample_data)
offsets = get_token_offsets(sample_data)
print(list(zip(tokens, labels, offsets)))