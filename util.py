import torch
from args import args, device
import os
import pandas as pd
from nn_utils import *
import numpy as np
from models import *

if args['model_name'] != 'base':
    if args['testing_mode']:
        umls_embedding_dict = read_umls_file_small(args['umls_embeddings_path'])
        umls_embedding_dict[default_key] = [0 for _ in range(50)]
        umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
        umls_key_to_index = get_key_to_index(umls_embedding_dict)
    else:
        umls_embedding_dict = read_umls_file(args['umls_embeddings_path'])
        umls_embedding_dict[default_key] = [0 for _ in range(50)]
        umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
        umls_key_to_index = get_key_to_index(umls_embedding_dict)
    pos_dict = read_pos_embeddings_file()
    pos_dict[default_key] = [0 for _ in range(20)]
    pos_dict = {k: np.array(v) for k, v in pos_dict.items()}
    pos_to_index = get_key_to_index(pos_dict)


def get_spans_from_seq_labels_2_classes(predictions_sub, batch_encoding):
    span_list = []
    start = None
    for i, label in enumerate(predictions_sub):
        if label == 0:
            if start is not None:
                span_list.append((start, i - 1))
                start = None
        else:
            assert label == 1
            if start is None:
                start = i
    if start is not None:
        span_list.append((start, len(predictions_sub) - 1))
    span_list_word = [(batch_encoding.token_to_word(span[0]), batch_encoding.token_to_word(span[1])) for span in
                      span_list]
    return span_list_word


def get_spans_from_seq_labels(predictions_sub, batch_encoding):
    if '3Classes' in args['model_name']:
        return get_spans_from_seq_labels_3_classes(predictions_sub, batch_encoding)
    elif '2Classes' in args['model_name']:
        return get_spans_from_seq_labels_2_classes(predictions_sub, batch_encoding)
    else:
        raise Exception('Have to specify num of classes in model name ' + args['model_name'])


def get_spans_from_seq_labels_3_classes(predictions_sub, batch_encoding):
    span_list = []
    start = None
    for i, label in enumerate(predictions_sub):
        if label == 0:
            if start is not None:
                span_list.append((start, i - 1))
                start = None
        elif label == 1:
            if start is not None:
                span_list.append((start, i - 1))
            start = i
        elif label == 2:
            if start is None:
                start = i
        else:
            raise Exception(f'Illegal label {label}')
    if start is not None:
        span_list.append((start, len(predictions_sub) - 1))
    span_list_word_idx = [(batch_encoding.token_to_word(span[0]), batch_encoding.token_to_word(span[1])) for span in
                          span_list]
    return span_list_word_idx


def f1(TP, FP, FN):
    if (TP + FP) == 0:
        precision = None
    else:
        precision = TP / (TP + FP)
    if (FN + TP) == 0:
        recall = None
    else:
        recall = TP / (FN + TP)
    if (precision is None) or (recall is None) or ((precision + recall) == 0):
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)


def get_raw_validation_data():
    output_dict = {}
    input_folder_path = args['raw_validation_files_path']
    data_files_list = os.listdir(input_folder_path)
    for filename in data_files_list:
        data_file_path = os.path.join(input_folder_path, filename)
        with open(data_file_path, 'r') as f:
            data = f.read()
        twitter_id = filename[:-4]
        output_dict[twitter_id] = data
    return output_dict


def get_raw_test_data():
    output_dict = {}
    input_folder_path = args['raw_test_files_path']
    data_files_list = os.listdir(input_folder_path)
    for filename in data_files_list:
        data_file_path = os.path.join(input_folder_path, filename)
        with open(data_file_path, 'r') as f:
            data = f.read()
        twitter_id = filename[:-4]
        output_dict[twitter_id] = data
    return output_dict


def get_validation_ids():
    output = []
    input_folder_path = args['raw_validation_files_path']
    data_files_list = os.listdir(input_folder_path)
    for filename in data_files_list:
        twitter_id = filename[:-4]
        output.append(twitter_id)
    return output


def get_raw_train_data():
    output_dict = {}
    input_folder_path = args['raw_train_files_path']
    data_files_list = os.listdir(input_folder_path)
    for filename in data_files_list:
        data_file_path = os.path.join(input_folder_path, filename)
        with open(data_file_path, 'r') as f:
            data = f.read()
        twitter_id = filename[:-4]
        output_dict[twitter_id] = data
    return output_dict


def read_disease_gazetteer():
    disease_list = []
    df = pd.read_csv(args['disease_gazetteer_path'], sep='\t')
    for _, row in df.iterrows():
        disease_term = row['term']
        disease_list.append(disease_term)
    return disease_list


def prepare_model_input(batch_encoding, sample_data):
    umls_indices = torch.tensor(expand_labels(batch_encoding, get_umls_indices(sample_data, umls_key_to_index)),
                                device=device)
    pos_indices = torch.tensor(expand_labels(batch_encoding, get_pos_indices(sample_data, pos_to_index)),
                               device=device)
    if args['model_name'] == 'SeqLabelerAllResourcesSmallerTopK':
        model_input = (batch_encoding, umls_indices, pos_indices)
    elif args['model_name'] == 'SeqLabelerDisGaz':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings)
    elif args['model_name'] == 'SeqLabelerUMLSDisGaz':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings)
    elif args['model_name'] == 'SeqLabelerUMLSDisGaz3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings)
    elif args['model_name'] == 'Silver3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings,
                       silver_dis_embeddings)
    elif args['model_name'] == 'LightWeightRIM3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'OneEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'TransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'PositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'SmallPositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'ComprehensivePositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings,
                       silver_dis_embeddings)
    elif args['model_name'] == 'PosEncod3ClassesNoSilverNewGaz':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'PosEncod3ClassesNoSilverBig':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    else:
        raise Exception('Not implemented!')
    return model_input


def prepare_model():
    if args['model_name'] == 'SeqLabelerAllResourcesSmallerTopK':
        return SeqLabelerAllResourcesSmallerTopK(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
                                                 pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    if args['model_name'] == 'SeqLabelerDisGaz':
        return SeqLabelerDisGaz(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
                                pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    if args['model_name'] == 'SeqLabelerUMLSDisGaz':
        return SeqLabelerUMLSDisGaz(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
                                    pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    if args['model_name'] == 'SeqLabelerUMLSDisGaz3Classes':
        return SeqLabelerUMLSDisGaz3Classes(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
                                            pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    if args['model_name'] == 'Silver3Classes':
        return Silver3Classes(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
                              pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    if args['model_name'] == 'LightWeightRIM3Classes':
        return LightWeightRIM3Classes().to(device)
    if args['model_name'] == 'OneEncoder3Classes':
        return OneEncoder3Classes().to(device)
    if args['model_name'] == 'TransformerEncoder3Classes':
        return TransformerEncoder3Classes().to(device)
    if args['model_name'] == 'PositionalTransformerEncoder3Classes':
        return PositionalTransformerEncoder3Classes().to(device)
    if args['model_name'] == 'SmallPositionalTransformerEncoder3Classes':
        return SmallPositionalTransformerEncoder3Classes().to(device)
    if args['model_name'] == 'ComprehensivePositionalTransformerEncoder3Classes':
        return ComprehensivePositionalTransformerEncoder3Classes(umls_pretrained=umls_embedding_dict,
                                                                 umls_to_idx=umls_key_to_index,
                                                                 pos_pretrained=pos_dict, pos_to_idx=pos_to_index) \
            .to(device)
    if args['model_name'] == 'PosEncod3ClassesNoSilverNewGaz':
        return PosEncod3ClassesNoSilverNewGaz().to(device)
    if args['model_name'] == 'PosEncod3ClassesNoSilverBig':
        return PosEncod3ClassesNoSilverBig().to(device)
    raise Exception(f"no code to prepare model {args['model_name']}")


def get_tweet_data(folder_path):
    id_to_data = {}
    data_files_list = os.listdir(folder_path)
    for filename in data_files_list:
        data_file_path = os.path.join(folder_path, filename)
        with open(data_file_path, 'r') as f:
            data = f.read()
        twitter_id = filename[:-4]
        id_to_data[twitter_id] = data
    return id_to_data
