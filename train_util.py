from structs import *
from typing import Dict, List
import json
import pandas as pd
from args import *
import util
from models import *

def print_args() -> None:
    """Print the configurations of the current run"""
    print("EXPERIMENT:", EXPERIMENT)
    print("TESTING_MODE", TESTING_MODE)
    print(json.dumps(args, indent=4, sort_keys=True))

def get_train_annos_dict() -> Dict[str, List[Anno]]:
    return util.get_annos_dict(args['train_annos_file_path'])

def get_valid_annos_dict() -> Dict[str, List[Anno]]:
    return util.get_annos_dict(args['valid_annos_file_path'])

def create_mistakes_visualization(mistakes_file_path: str, mistakes_visualization_file_path: str) -> None:
    """
    Create a gate-visualization-file(.bdocjs format) that contains the mistakes
    made by a trained model.

    Args:
        - mistakes_file_path: the file path containing the mistakes of the model
        - gate_visualization_file_path: the gate visualization file path to create 
    """
    gold_annos_dict = get_valid_annos_dict()
    mistake_annos_dict = util.get_mistakes_annos(mistakes_file_path)
    sample_to_text_valid = get_valid_texts()
    combined_annos_dict = {}
    for sample_id in gold_annos_dict:
        gold_annos_list = gold_annos_dict[sample_id]
        mistake_annos_list = mistake_annos_dict.get(sample_id, [])
        combined_list = gold_annos_list + mistake_annos_list
        for anno in combined_list:
            anno.begin_offset = int(anno.begin_offset)
            anno.end_offset = int(anno.end_offset)
        combined_annos_dict[sample_id] = combined_list 
    util.create_visualization_file(
        mistakes_visualization_file_path,
        combined_annos_dict,
        sample_to_text_valid
        )


def extract_expanded_labels(sample_data, batch_encoding, annos) -> List[Label]:
    if '3Classes' in args['model_name']:
        labels = util.get_labels_bio(sample_data, annos)
        expanded_labels = util.expand_labels_rich(batch_encoding, labels)
        return expanded_labels
    elif '2Classes' in args['model_name']:
        labels = util.get_label_strings(sample_data, annos)
        expanded_labels = util.expand_labels(batch_encoding, labels)
        return expanded_labels
    raise Exception('Have to specify num of classes in model name ' + args['model_name'])


def read_pos_embeddings_file():
    return pd.read_pickle(args['pos_embeddings_path'])

def get_label_idx_dicts() -> tuple[Dict[Label, int], Dict[int, Label]]:
    return util.get_label_idx_dicts(args['types_file_path'])

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


def get_spans_from_seq_labels(predictions_sub, batch_encoding):
    if '3Classes' in args['model_name']:
        return util.get_spans_from_bio_labels(predictions_sub, batch_encoding)
    elif '2Classes' in args['model_name']:
        return util.get_spans_from_seq_labels_2_classes(predictions_sub, batch_encoding)
    else:
        raise Exception('Have to specify num of classes in model name ' + args['model_name'])

def read_disease_gazetteer():
    disease_list = []
    df = pd.read_csv(args['disease_gazetteer_path'], sep='\t')
    for _, row in df.iterrows():
        disease_term = row['term']
        disease_list.append(disease_term)
    return disease_list

def prepare_model_input(batch_encoding, sample_data: List[TokenData]):
    # umls_indices = torch.tensor(util.expand_labels(batch_encoding, get_umls_indices(sample_data, umls_key_to_index)),
    #                             device=device)
    # pos_indices = torch.tensor(util.expand_labels(batch_encoding, get_pos_indices(sample_data, pos_to_index)),
    #                            device=device)
    umls_indices = None
    pos_indices = None
    if args['model_name'] == 'SeqLabelerAllResourcesSmallerTopK':
        model_input = (batch_encoding, umls_indices, pos_indices)
    elif args['model_name'] == 'SeqLabelerDisGaz':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings)
    elif args['model_name'] == 'SeqLabelerUMLSDisGaz':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings)
    elif args['model_name'] == 'SeqLabelerUMLSDisGaz3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings)
    elif args['model_name'] == 'Silver3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings,
                       silver_dis_embeddings)
    elif args['model_name'] == 'LightWeightRIM3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'OneEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'TransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'PositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'SmallPositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'ComprehensivePositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings,
                       silver_dis_embeddings)
    elif args['model_name'] == 'PosEncod3ClassesNoSilverNewGaz':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'PosEncod3ClassesNoSilverBig':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
                                             device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif args['model_name'] == 'PosEncod3ClassesNoSilverSpanish':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding,util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
                                               device=device)
        silver_dis_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
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


# TODO: move to different module
def get_train_data() -> Dict[str, List[TokenData]]:
    return util.read_data_from_folder(args['training_data_folder_path'])


# TODO: move to different module
def get_valid_data() -> Dict[str, List[TokenData]]:
    return util.read_data_from_folder(args['validation_data_folder_path'])


# TODO: move to different module
def get_test_data() -> Dict[str, List[TokenData]]:
    return util.read_data_from_folder(args['test_data_folder_path'])


# TODO: move to different module
def get_train_texts() -> Dict[SampleId, str]:
    return util.get_texts(args['train_sample_text_data_file_path'])


# TODO: move to different module
def get_valid_texts() -> Dict[SampleId, str]:
    return util.get_texts(args['valid_sample_text_data_file_path'])