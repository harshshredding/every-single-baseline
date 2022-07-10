from util import read_disease_gazetteer
from read_gate_output import *
from args import args
import numpy as np
from nn_utils import *

sample_to_token_data_train = get_train_data_small(args['training_data_folder_path'])
pos_dict = read_pos_embeddings_file()
pos_dict = {k: np.array(v) for k, v in pos_dict.items()}
pos_to_index = get_key_to_index(pos_dict)
pos_tag_set = set()
for sample_id in sample_to_token_data_train:
    sample_data = sample_to_token_data_train[sample_id]
    diz_gaz = get_dis_gaz_labels(sample_data)
    one_hot = get_dis_gaz_one_hot(sample_data)
    tokens = get_token_strings(sample_data)
    labels = get_labels(sample_data)
    umls_labels = get_umls_data(sample_data)
    print(umls_labels)
    print(list(zip(diz_gaz, tokens, labels)))
    print(list(zip(diz_gaz, one_hot)))
    break
