import torch.nn as nn
import torch
from read_gate_output import *
from args import args
from args import device
import pandas as pd


class Embedding(nn.Module):
    def __init__(self, emb_dim, vocab_size, initialize_emb, word_to_ix):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim).requires_grad_(False)
        if initialize_emb:
            inv_dic = {v: k for k, v in word_to_ix.items()}
            for key in initialize_emb.keys():
                if key in word_to_ix:
                    ind = word_to_ix[key]
                    self.embedding.weight.data[ind].copy_(torch.from_numpy(initialize_emb[key]))

    def forward(self, input):
        return self.embedding(input)


def expand_labels(batch_encoding, labels):
    """
    return a list of labels with each label in the list
    corresponding to each token in batch_encoding
    """
    new_labels = []
    for token_idx in range(len(batch_encoding.tokens())):
        word_idx = batch_encoding.token_to_word(token_idx)
        new_labels.append(labels[word_idx])
    return new_labels


def read_umls_file(umls_file_path):
    umls_embedding_dict = {}
    with open(umls_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line_split = line.split(',')
            assert len(line_split) == 51
            umls_id = line_split[0]
            embedding_vector = [float(val) for val in line_split[1:len(line_split)]]
            umls_embedding_dict[umls_id] = embedding_vector
    return umls_embedding_dict


def get_key_to_index(some_dict):
    key_to_index = {}
    for index, key in enumerate(some_dict.keys()):
        key_to_index[key] = index
    return key_to_index


def read_umls_file_small(umls_file_path):
    umls_embedding_dict = {}
    with open(umls_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line_split = line.split(',')
            assert len(line_split) == 51
            umls_id = line_split[0]
            embedding_vector = [float(val) for val in line_split[1:len(line_split)]]
            umls_embedding_dict[umls_id] = embedding_vector
            break
    return umls_embedding_dict


def extract_labels(sample_data, batch_encoding):
    labels = get_labels(sample_data)
    expanded_labels = expand_labels(batch_encoding, labels)
    expanded_labels = [0 if label == 'o' else 1 for label in expanded_labels]
    return expanded_labels


def read_pos_embeddings_file():
    return pd.read_pickle(args['pos_embeddings_path'])
