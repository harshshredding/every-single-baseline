# import torch.nn as nn
# import torch
# from read_gate_output import *
# from args import args
# import pandas as pd
# from torch import Tensor
# import math
# from typing import Dict, List
# from structs import *
# from util import get_labels_bio, get_label_strings

# class Embedding(nn.Module):
#     def __init__(self, emb_dim, vocab_size, initialize_emb, word_to_ix):
#         super(Embedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_dim).requires_grad_(False)
#         if initialize_emb:
#             inv_dic = {v: k for k, v in word_to_ix.items()}
#             for key in initialize_emb.keys():
#                 if key in word_to_ix:
#                     ind = word_to_ix[key]
#                     self.embedding.weight.data[ind].copy_(torch.from_numpy(initialize_emb[key]))

#     def forward(self, input):
#         return self.embedding(input)


# ######################################################################
# # ``PositionalEncoding`` module injects some information about the
# # relative or absolute position of the tokens in the sequence. The
# # positional encodings have the same dimension as the embeddings so that
# # the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# # different frequencies.
# #
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.unsqueeze(x, dim=1)
#         x = x + self.pe[:x.size(0)]
#         x = self.dropout(x)
#         x = torch.squeeze(x, dim=1)
#         return x


# def expand_labels(batch_encoding, labels):
#     """
#     return a list of labels with each label in the list
#     corresponding to each token in batch_encoding
#     """
#     new_labels = []
#     for token_idx in range(len(batch_encoding.tokens())):
#         word_idx = batch_encoding.token_to_word(token_idx)
#         new_labels.append(labels[word_idx])
#     return new_labels


# def expand_labels_rich(batch_encoding, labels: List[Label]) -> List[Label]:
#     """
#     return a list of labels with each label in the list
#     corresponding to each token in batch_encoding
#     """
#     new_labels = []
#     prev_word_idx = None
#     prev_label = None
#     for token_idx in range(len(batch_encoding.tokens())):
#         word_idx = batch_encoding.token_to_word(token_idx)
#         label = labels[word_idx]
#         if (label.bio_tag == BioTag.begin) and (prev_word_idx == word_idx):
#             new_labels.append(Label(label_type=prev_label.label_type, bio_tag=BioTag.inside))
#         else:
#             new_labels.append(labels[word_idx])
#         prev_word_idx = word_idx
#         prev_label = label
#     return new_labels


# def read_umls_file(umls_file_path):
#     umls_embedding_dict = {}
#     with open(umls_file_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip()
#             line_split = line.split(',')
#             assert len(line_split) == 51
#             umls_id = line_split[0]
#             embedding_vector = [float(val) for val in line_split[1:len(line_split)]]
#             umls_embedding_dict[umls_id] = embedding_vector
#     return umls_embedding_dict


# def get_key_to_index(some_dict):
#     key_to_index = {}
#     for index, key in enumerate(some_dict.keys()):
#         key_to_index[key] = index
#     return key_to_index


# def read_umls_file_small(umls_file_path):
#     umls_embedding_dict = {}
#     with open(umls_file_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip()
#             line_split = line.split(',')
#             assert len(line_split) == 51
#             umls_id = line_split[0]
#             embedding_vector = [float(val) for val in line_split[1:len(line_split)]]
#             umls_embedding_dict[umls_id] = embedding_vector
#             break
#     return umls_embedding_dict


# def extract_expanded_labels(sample_data, batch_encoding, annos, labels_dict) -> List[Label]:
#     if '3Classes' in args['model_name']:
#         labels = get_labels_bio(sample_data, annos, labels_dict)
#         expanded_labels = expand_labels_rich(batch_encoding, labels)
#         return expanded_labels
#     elif '2Classes' in args['model_name']:
#         labels = get_label_strings(sample_data, labels_dict)
#         expanded_labels = expand_labels(batch_encoding, labels)
#         return expanded_labels
#     raise Exception('Have to specify num of classes in model name ' + args['model_name'])


# def read_pos_embeddings_file():
#     return pd.read_pickle(args['pos_embeddings_path'])
