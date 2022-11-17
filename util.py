import csv
from pathlib import Path
from models import *
from gatenlp import Document
from args import *
from structs import *
from typing import Dict, List
import json
import os
import pandas as pd
import torch
from read_gate_output import *
import pandas as pd

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


def extract_expanded_labels(sample_data, batch_encoding, annos) -> List[Label]:
    if '3Classes' in args['model_name']:
        labels = get_labels_bio(sample_data, annos)
        expanded_labels = expand_labels_rich(batch_encoding, labels)
        return expanded_labels
    elif '2Classes' in args['model_name']:
        labels = get_label_strings(sample_data, annos)
        expanded_labels = expand_labels(batch_encoding, labels)
        return expanded_labels
    raise Exception('Have to specify num of classes in model name ' + args['model_name'])


def read_pos_embeddings_file():
    return pd.read_pickle(args['pos_embeddings_path'])

# if args['model_name'] != 'base':
#     if TESTING_MODE:
#         umls_embedding_dict = read_umls_file_small(args['umls_embeddings_path'])
#         umls_embedding_dict[default_key] = [0 for _ in range(50)]
#         umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
#         umls_key_to_index = get_key_to_index(umls_embedding_dict)
#     else:
#         umls_embedding_dict = read_umls_file(args['umls_embeddings_path'])
#         umls_embedding_dict[default_key] = [0 for _ in range(50)]
#         umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
#         umls_key_to_index = get_key_to_index(umls_embedding_dict)
#     pos_dict = read_pos_embeddings_file()
#     pos_dict[default_key] = [0 for _ in range(20)]
#     pos_dict = {k: np.array(v) for k, v in pos_dict.items()}
#     pos_to_index = get_key_to_index(pos_dict)


def print_args():
    print("EXPERIMENT:", EXPERIMENT)
    print("TESTING_MODE", TESTING_MODE)
    print(json.dumps(args,indent=4))


def get_extraction(tokens, offsets, start, end):
    extraction = []
    for i, (start_offset, end_offset) in enumerate(offsets):
        if start_offset >= start and end_offset <= end:
            extraction.append(tokens[i])
    return ' '.join(extraction)


def get_label_idx_dicts() -> tuple[Dict[Label, int], Dict[int, Label]]:
    label_to_idx_dict = {}
    with open(args['types_file_path'], 'r') as types_file:
        for line in types_file.readlines():
            type_string = line.strip()
            if len(type_string):
                label_to_idx_dict[Label(type_string, BioTag.begin)] = len(label_to_idx_dict)
                label_to_idx_dict[Label(type_string, BioTag.inside)] = len(label_to_idx_dict)
    label_to_idx_dict[Label.get_outside_label()] = len(label_to_idx_dict)
    idx_to_label_dict = {}
    for label in label_to_idx_dict:
        idx_to_label_dict[label_to_idx_dict[label]] = label
    assert len(label_to_idx_dict) == len(idx_to_label_dict)
    return label_to_idx_dict, idx_to_label_dict


def get_optimizer(model):
    if args['optimizer'] == 'Ranger':
        #return torch_optimizer.Ranger(model.parameters(), args['learning_rate'])
        raise Exception("no ranger optimizer")
    elif args['optimizer'] == 'Adam':
        return torch.optim.Adam(model.parameters(), args['learning_rate'])
    elif args['optimizer'] == 'AdamW':
        return torch.optim.AdamW(model.parameters(), args['learning_rate'])
    else:
        raise Exception(f"optimizer not found: {args['optimizer']}")


def print_list(some_list):
    print('\n'.join([str(el) for el in some_list]))


def create_gate_input_file(output_file_path, sample_to_token_data: Dict[str, List[TokenData]],
                           annos_dict: Dict[str, List[Anno]], num_samples=None):
    with open(output_file_path, 'w') as output_file:
        writer = csv.writer(output_file)
        header = ['sample_id', 'text', 'spans']
        writer.writerow(header)
        sample_list = list(sample_to_token_data.keys())
        if num_samples is not None:
            sample_list = sample_list[:num_samples]
        for sample_id in sample_list:
            gold_annos = annos_dict.get(sample_id, [])
            sample_data = sample_to_token_data[sample_id]
            sample_text = ''.join(get_token_strings(sample_data))
            spans = "@".join([f"{anno.begin_offset}:{anno.end_offset}" for anno in gold_annos])
            row_to_write = [sample_id, sample_text, spans]
            writer.writerow(row_to_write)

def create_gate_file(output_file_path: str, sample_to_token_data: Dict[str, List[TokenData]],
                     annos_dict: Dict[str, List[Anno]], num_samples=None) -> None:
    assert output_file_path[-7:] == '.bdocjs'
    sample_list = list(sample_to_token_data.keys())
    if num_samples is not None:
        sample_list = sample_list[:num_samples]
    curr_sample_offset = 0
    document_text = ''
    all_gate_annos = []
    for sample_id in sample_list:
        sample_start_offset = curr_sample_offset
        gold_annos = annos_dict.get(sample_id, [])
        sample_data = sample_to_token_data[sample_id]
        sample_text = ' '.join(get_token_strings(sample_data)) + '\n'
        all_gate_annos.extend([(curr_sample_offset + anno.begin_offset, curr_sample_offset + anno.end_offset,
                           anno.label_type, anno.features) for anno in gold_annos])
        all_gate_annos.extend([(curr_sample_offset + anno.begin_offset, curr_sample_offset + anno.end_offset,'Span', anno.features) for anno in gold_annos])
        document_text += sample_text
        curr_sample_offset += len(sample_text)
        sample_end_offset = curr_sample_offset
        all_gate_annos.append((sample_start_offset, sample_end_offset, 'Sample', {'sample_id': sample_id}))
    gate_document = Document(document_text)
    default_ann_set = gate_document.annset()
    for gate_anno in all_gate_annos:
        default_ann_set.add(int(gate_anno[0]), int(gate_anno[1]), gate_anno[2], gate_anno[3])
    gate_document.save(output_file_path)

def p_string(obj) -> str:
    return json.dumps(obj=obj, indent=4)

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
        return get_spans_from_bio_labels(predictions_sub, batch_encoding)
    elif '2Classes' in args['model_name']:
        return get_spans_from_seq_labels_2_classes(predictions_sub, batch_encoding)
    else:
        raise Exception('Have to specify num of classes in model name ' + args['model_name'])


def get_spans_from_bio_labels(predictions_sub: List[Label], batch_encoding):
    span_list = []
    start = None
    start_label = None
    for i, label in enumerate(predictions_sub):
        if label.bio_tag == BioTag.out:
            if start is not None:
                span_list.append((start, i - 1, start_label))
                start = None
                start_label = None
        elif label.bio_tag == BioTag.begin:
            if start is not None:
                span_list.append((start, i - 1, start_label))
            start = i
            start_label = label.label_type
        elif label.bio_tag == BioTag.inside:
            if (start is not None) and (start_label != label.label_type):
                span_list.append((start, i - 1, start_label))
                start = None
                start_label = None
        else:
            raise Exception(f'Illegal label {label}')
    if start is not None:
        span_list.append((start, len(predictions_sub) - 1, start_label))
    span_list_word_idx = [(batch_encoding.token_to_word(span[0]), batch_encoding.token_to_word(span[1]), span[2])
                          for span in span_list]
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
        return 0, 0, 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall


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


def prepare_model_input(batch_encoding, sample_data: List[TokenData]):
    # umls_indices = torch.tensor(expand_labels(batch_encoding, get_umls_indices(sample_data, umls_key_to_index)),
    #                             device=device)
    # pos_indices = torch.tensor(expand_labels(batch_encoding, get_pos_indices(sample_data, pos_to_index)),
    #                            device=device)
    umls_indices = None
    pos_indices = None
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
    elif args['model_name'] == 'PosEncod3ClassesNoSilverSpanish':
        dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(expand_labels(batch_encoding, get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(expand_labels(batch_encoding, get_silver_dis_one_hot(sample_data)),
                                             device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'PosEncod3ClassesOnlyRoberta':
        model_input = [batch_encoding]
    elif args['model_name'] == 'OnlyRoberta3Classes':
        model_input = [batch_encoding]
    elif args['model_name'] == 'JustBert3Classes':
        model_input = [batch_encoding]
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
    if args['model_name'] == 'PosEncod3ClassesNoSilverSpanish':
        return PosEncod3ClassesNoSilverSpanish().to(device)
    if args['model_name'] == 'PosEncod3ClassesOnlyRoberta':
        return PosEncod3ClassesOnlyRoberta().to(device)
    if args['model_name'] == 'OnlyRoberta3Classes':
        return OnlyRoberta3Classes().to(device)
    if args['model_name'] == 'JustBert3Classes':
        return JustBert3Classes().to(device)
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


def get_mistakes_annos(mistakes_file_path) -> SampleAnnotations:
    """
    Get the annotations that correspond to mistakes for each sample using
    the given mistakes file. 

    Args:
        mistakes_file_path: the file-path representing the file(a .tsv file)
        that contains the mistakes made by a model.
    """
    df = pd.read_csv(mistakes_file_path, sep='\t')
    sample_to_annos = {}
    for _, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        print(row['begin'], row['sample_id'])
        annos_list.append(Anno(int(row['begin']), int(row['end']), row['mistake_type'], row['extraction'], {"type":row['type']}))
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos


def create_gate_input_mistakes(mistakes_file_path: str, gate_visualization_file_path: str) -> None:
    """
    Create a gate-visualization-file(.bdocjs format) that contains the mistakes
    made by a trained model.

    Args:
        - mistakes_file_path: the file path containing the mistakes of the model
        - gate_visualization_file_path: the gate visualization file path to create 
    """
    sample_to_token_data = get_valid_data()
    gold_annos_dict = get_valid_annos_dict()
    mistake_annos_dict = get_mistakes_annos(mistakes_file_path)
    combined_annos_dict = {}
    for sample_id in gold_annos_dict:
        gold_annos_list = gold_annos_dict[sample_id]
        mistake_annos_list = mistake_annos_dict.get(sample_id, [])
        combined_list = gold_annos_list + mistake_annos_list
        for anno in combined_list:
            anno.begin_offset = int(anno.begin_offset)
            anno.end_offset = int(anno.end_offset)
        combined_annos_dict[sample_id] = combined_list 
    create_gate_file(gate_visualization_file_path, sample_to_token_data, combined_annos_dict)


def get_train_annos_dict() -> Dict[str, List[Anno]]:
    if curr_dataset == Dataset.social_dis_ner:
        df = pd.read_csv(args['train_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['tweets_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], 'Disease', row['extraction']))
            sample_to_annos[str(row['tweets_id'])] = annos_list
        return sample_to_annos
    elif curr_dataset == Dataset.few_nerd:
        df = pd.read_csv(args['train_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['sample_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
            sample_to_annos[str(row['sample_id'])] = annos_list
        return sample_to_annos
    elif curr_dataset == Dataset.genia:
        df = pd.read_csv(args['train_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['sample_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
            sample_to_annos[str(row['sample_id'])] = annos_list
        return sample_to_annos
    elif curr_dataset == Dataset.multiconer:
        df = pd.read_csv(args['train_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['sample_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
            sample_to_annos[str(row['sample_id'])] = annos_list
        return sample_to_annos
    elif curr_dataset == Dataset.legaleval:
        df = pd.read_csv(args['train_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['sample_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
            sample_to_annos[str(row['sample_id'])] = annos_list
        return sample_to_annos
    else:
        raise Exception(f"{args['dataset_name']} is not supported")


def open_make_dirs(file_path, mode):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return open(file_path, mode)

def create_directory_structure(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def get_valid_annos_dict() -> Dict[str, List[Anno]]:
    if curr_dataset == Dataset.social_dis_ner:
        df = pd.read_csv(args['valid_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['tweets_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], 'Disease', row['extraction']))
            sample_to_annos[str(row['tweets_id'])] = annos_list
        return sample_to_annos
    else:
        df = pd.read_csv(args['valid_annos_file_path'], sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['sample_id']), [])
            annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
            sample_to_annos[str(row['sample_id'])] = annos_list
        return sample_to_annos




def parse_token_data(token_data_raw) -> TokenData:
    """
    in: a json object representing a token
    out: a TokenData object parsed from the input json object
    """
    if curr_dataset == Dataset.few_nerd:
        return TokenData(
            str(token_data_raw['Sample'][0]['id']),
            token_data_raw['Token'][0]['string'],
            token_data_raw['Token'][0]['length'],
            token_data_raw['Token'][0]['startOffset'],
            token_data_raw['Token'][0]['endOffset']
        )
    elif curr_dataset == Dataset.social_dis_ner:
        return TokenData(
            sample_id=str(token_data_raw['tweet_text'][0]['twitter_id']),
            token_string=token_data_raw['Token'][0]['string'],
            token_len=token_data_raw['Token'][0]['length'],
            token_start_offset=token_data_raw['Token'][0]['startOffset'],
            token_end_offset=token_data_raw['Token'][0]['endOffset'],
        )
    elif curr_dataset == Dataset.genia:
        return TokenData(
            str(token_data_raw['Sample'][0]['id']),
            token_data_raw['Token'][0]['string'],
            token_data_raw['Token'][0]['length'],
            token_data_raw['Token'][0]['startOffset'],
            token_data_raw['Token'][0]['endOffset'],
        )
    elif curr_dataset == Dataset.multiconer:
        return TokenData(
            str(token_data_raw['Sample'][0]['id']),
            token_data_raw['Token'][0]['string'],
            token_data_raw['Token'][0]['length'],
            token_data_raw['Token'][0]['startOffset'],
            token_data_raw['Token'][0]['endOffset'],
        )
    elif curr_dataset == Dataset.legaleval:
        return TokenData(
            str(token_data_raw['Sample'][0]['id']),
            token_data_raw['Token'][0]['string'],
            token_data_raw['Token'][0]['length'],
            token_data_raw['Token'][0]['startOffset'],
            token_data_raw['Token'][0]['endOffset'],
        )  
    else:
        raise NotImplementedError(f"implement token data parsing for dataset {args['dataset_name']}")


def read_data_from_folder(data_folder) -> Dict[str, List[TokenData]]:
    sample_to_tokens = {}
    data_files_list = os.listdir(data_folder)
    for filename in data_files_list:
        data_file_path = os.path.join(data_folder, filename)
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        for token_data in data:
            parsed_token_data = parse_token_data(token_data)
            sample_id = str(parsed_token_data.sample_id)
            sample_tokens = sample_to_tokens.get(sample_id, [])
            sample_tokens.append(parsed_token_data)
            sample_to_tokens[sample_id] = sample_tokens
    return sample_to_tokens


def get_train_data() -> Dict[str, List[TokenData]]:
    return read_data_from_folder(args['training_data_folder_path'])


def get_valid_data() -> Dict[str, List[TokenData]]:
    return read_data_from_folder(args['validation_data_folder_path'])


def get_test_data() -> Dict[str, List[TokenData]]:
    return read_data_from_folder(args['test_data_folder_path'])


def get_token_strings(sample_data: List[TokenData]):
    only_token_strings = []
    for token_data in sample_data:
        only_token_strings.append(token_data.token_string)
    return only_token_strings

def get_annos_surrounding_token(annos: List[Anno], token: TokenData) -> List[Anno]:
    return [anno for anno in annos \
        if (anno.begin_offset <= token.token_start_offset) and (token.token_end_offset <= anno.end_offset)]

def get_label_strings(sample_data: List[TokenData], annos: List[Anno]):
    ret_labels = []
    for token_data in sample_data:
        surrounding_annos = get_annos_surrounding_token(annos, token_data)
        if not len(surrounding_annos):
            ret_labels.append(OUTSIDE_LABEL_STRING)
        else:
            ret_labels.append(surrounding_annos[0].label_type)
    return ret_labels


# def get_labels_bio(sample_data: List[TokenData], annos: List[Anno], types_dict) -> List[Label]:
#     labels = get_label_strings(sample_data, types_dict)
#     offsets = get_token_offsets(sample_data)
#     new_labels = []
#     for (label_string, curr_offset) in zip(labels, offsets):
#         if label_string != OUTSIDE_LABEL_STRING:
#             anno_same_start = [anno for anno in annos if anno.begin_offset == curr_offset[0]]
#             in_anno = [anno for anno in annos if
#                        (curr_offset[0] >= anno.begin_offset) and (curr_offset[1] <= anno.end_offset)]
#             if len(anno_same_start) > 0:
#                 new_labels.append(Label(label_string, BioTag.begin))
#             else:
#                 # avoid DiseaseMid without a DiseaseStart
#                 if (len(new_labels) > 0) and (new_labels[-1].bio_tag != BioTag.out) \
#                         and (label_string == new_labels[-1].label_type) and len(in_anno):
#                     new_labels.append(Label(label_string, BioTag.inside))
#                 else:
#                     new_labels.append(Label.get_outside_label())
#         else:
#             new_labels.append(Label.get_outside_label())
#     return new_labels

def get_labels_bio(sample_token_data: List[TokenData], annos: List[Anno]) -> List[Label]:
    """
    Takes all tokens and gold annotations for a sample
    and outputs a labels(one for each token) representing 
    whether a token is at the beginning(B), inside(I), or outside(O) of an entity.
    """
    new_labels = []
    for token in sample_token_data:
        annos_that_surround = get_annos_surrounding_token(annos, token)
        if not len(annos_that_surround):
            new_labels.append(Label.get_outside_label())
        else:
            assert len(annos_that_surround) == 1
            annos_with_same_start = [anno for anno in annos_that_surround if anno.begin_offset == token.token_start_offset]
            if len(annos_with_same_start):
                new_labels.append(Label(annos_with_same_start[0].label_type, BioTag.begin))
            else:
                new_labels.append(Label(annos_that_surround[0].label_type, BioTag.inside))
    return new_labels

def get_token_offsets(sample_data: List[TokenData]) -> List[tuple]:
    offsets_list = []
    for token_data in sample_data:
        offsets_list.append((token_data.token_start_offset,
                             token_data.token_end_offset))
    return offsets_list

def get_umls_data(sample_data):
    raise NotImplementedError()
    # umls_tags = []
    # for token_data in sample_data:
    #     if 'UMLS' in token_data:
    #         umls_tags.append(token_data['UMLS'])
    #     else:
    #         umls_tags.append('o')
    # return umls_tags


def get_dis_gaz_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'DisGaz' in token_data:
    #         output.append('DisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_dis_gaz_one_hot(sample_data):
    # dis_labels = get_dis_gaz_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_umls_diz_gaz_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'UMLS_Disease' in token_data:
    #         output.append('UmlsDisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_umls_dis_gaz_one_hot(sample_data):
    # dis_labels = get_umls_diz_gaz_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_silver_dis_one_hot(sample_data):
    # dis_labels = get_silver_dis_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_silver_dis_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'SilverDisGaz' in token_data:
    #         output.append('SilverDisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_pos_data(sample_data):
    # pos_tags = []
    # for token_data in sample_data:
    #     pos_tags.append(token_data['Token'][0]['category'])
    # return pos_tags
    raise NotImplementedError()


def get_umls_indices(sample_data, umls_key_to_index):
    # umls_data = get_umls_data(sample_data)
    # umls_keys = [default_key if umls == 'o' else umls[0]['CUI'] for umls in umls_data]
    # default_index = umls_key_to_index[default_key]
    # umls_indices = [umls_key_to_index.get(key, default_index) for key in umls_keys]
    # return umls_indices
    raise NotImplementedError()


def get_pos_indices(sample_data, pos_key_to_index):
    # pos_tags = get_pos_data(sample_data)
    # default_index = pos_key_to_index[default_key]
    # pos_indices = [pos_key_to_index.get(tag, default_index) for tag in pos_tags]
    # return pos_indices
    raise NotImplementedError()


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


def expand_labels_rich(batch_encoding, labels: List[Label]) -> List[Label]:
    """
    return a list of labels with each label in the list
    corresponding to each token in batch_encoding
    """
    new_labels = []
    prev_word_idx = None
    prev_label = None
    for token_idx in range(len(batch_encoding.tokens())):
        word_idx = batch_encoding.token_to_word(token_idx)
        label = labels[word_idx]
        if (label.bio_tag == BioTag.begin) and (prev_word_idx == word_idx):
            assert prev_label is not None
            new_labels.append(Label(label_type=prev_label.label_type, bio_tag=BioTag.inside))
        else:
            new_labels.append(labels[word_idx])
        prev_word_idx = word_idx
        prev_label = label
    return new_labels