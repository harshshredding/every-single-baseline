from structs import *
import pandas as pd
from args import EXPERIMENT_NAME, TESTING_MODE
from models import *
import csv
import util
from typing import Dict
from colorama import Fore, Style
from utils.config import ModelConfig, DatasetConfig


def print_args(dataset_config: Dict) -> None:
    """Print the configurations of the current run"""
    print(Fore.GREEN)
    print("\n\n------ DATASET CONFIG --------")
    print("EXPERIMENT_NAME:", EXPERIMENT_NAME)
    print("TESTING_MODE:", TESTING_MODE)
    for k, v in dataset_config.items():
        print(k, v)
    print("-----------CONFIG----------\n\n")
    print(Style.RESET_ALL)


def get_loss_function():
    return nn.CrossEntropyLoss()


def get_bert_tokenizer(model_config: ModelConfig):
    """
    Get the bert tokenizer
    """
    return AutoTokenizer.from_pretrained(model_config.bert_model_name)


def get_train_annos_dict(dataset_config: DatasetConfig) -> Dict[str, List[Anno]]:
    return util.get_annos_dict(dataset_config.train_annos_file_path)


def get_valid_annos_dict(dataset_config: DatasetConfig) -> Dict[str, List[Anno]]:
    return util.get_annos_dict(dataset_config.valid_annos_file_path)


def extract_expanded_labels(sample_token_data: List[TokenData],
                            batch_encoding,
                            annos: List[Anno],
                            model_config: ModelConfig) -> List[Label]:
    if '3Classes' in model_config.model_name:
        labels = util.get_labels_bio(sample_token_data, annos)
        expanded_labels = util.expand_labels_rich(batch_encoding, labels)
        return expanded_labels
    elif '2Classes' in model_config.model_name:
        labels = util.get_label_strings(sample_token_data, annos)
        expanded_labels = util.expand_labels(batch_encoding, labels)
        return expanded_labels
    raise Exception('Have to specify num of classes in model name ' + model_config.model_name)

def read_pos_embeddings_file(dataset_config: DatasetConfig):
    return pd.read_pickle(dataset_config['pos_embeddings_path'])


def get_optimizer(model, model_config: ModelConfig):
    if model_config.optimizer == 'Ranger':
        # return torch_optimizer.Ranger(model.parameters(), dataset_config['learning_rate'])
        raise Exception("no ranger optimizer")
    elif model_config.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), model_config.learning_rate)
    elif model_config.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), model_config.learning_rate)
    else:
        raise Exception(f"optimizer not found: {model_config.optimizer}")


def store_performance_result(performance_file_path, f1_score, epoch: int, experiment_name: str, dataset: Dataset):
    with open(performance_file_path, 'a') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow([experiment_name, dataset.name, str(epoch), str(f1_score)])


def create_performance_file_header(performance_file_path):
    with open(performance_file_path, 'w') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow(['experiment_name', 'dataset_name', 'epoch', 'f1_score'])


# if dataset_config['model_name'] != 'base':
#     if TESTING_MODE:
#         umls_embedding_dict = read_umls_file_small(dataset_config['umls_embeddings_path'])
#         umls_embedding_dict[default_key] = [0 for _ in range(50)]
#         umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
#         umls_key_to_index = get_key_to_index(umls_embedding_dict)
#     else:
#         umls_embedding_dict = read_umls_file(dataset_config['umls_embeddings_path'])
#         umls_embedding_dict[default_key] = [0 for _ in range(50)]
#         umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
#         umls_key_to_index = get_key_to_index(umls_embedding_dict)
#     pos_dict = read_pos_embeddings_file()
#     pos_dict[default_key] = [0 for _ in range(20)]
#     pos_dict = {k: np.array(v) for k, v in pos_dict.items()}
#     pos_to_index = get_key_to_index(pos_dict)


def get_spans_from_seq_labels(predictions_sub, batch_encoding, model_config: ModelConfig):
    if '3Classes' in model_config.model_name:
        return util.get_spans_from_bio_labels(predictions_sub, batch_encoding)
    elif '2Classes' in model_config.model_name:
        return util.get_spans_from_seq_labels_2_classes(predictions_sub, batch_encoding)
    else:
        raise Exception(f"Have to specify num of classes in model name {model_config.model_name}")


def read_disease_gazetteer(dataset_config: DatasetConfig):
    disease_list = []
    df = pd.read_csv(dataset_config['disease_gazetteer_path'], sep='\t')
    for _, row in df.iterrows():
        disease_term = row['term']
        disease_list.append(disease_term)
    return disease_list


def prepare_model_input(batch_encoding, sample_data: List[TokenData], model_config: ModelConfig):
    # umls_indices = torch.tensor(util.expand_labels(batch_encoding, get_umls_indices(sample_data, umls_key_to_index)),
    #                             device=device)
    # pos_indices = torch.tensor(util.expand_labels(batch_encoding, get_pos_indices(sample_data, pos_to_index)),
    #                            device=device)
    umls_indices = None
    pos_indices = None
    if model_config.model_name == 'SeqLabelerAllResourcesSmallerTopK':
        model_input = (batch_encoding, umls_indices, pos_indices)
    elif model_config.model_name == 'SeqLabelerDisGaz':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings)
    elif model_config.model_name == 'SeqLabelerUMLSDisGaz':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings)
    elif model_config.model_name == 'SeqLabelerUMLSDisGaz3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings)
    elif model_config.model_name == 'Silver3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings,
                       silver_dis_embeddings)
    elif model_config.model_name == 'LightWeightRIM3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'OneEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'TransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'PositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'SmallPositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'ComprehensivePositionalTransformerEncoder3Classes':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        model_input = (batch_encoding, umls_indices, pos_indices, dis_gaz_embeddings, umls_dis_gaz_embeddings,
                       silver_dis_embeddings)
    elif model_config.model_name == 'PosEncod3ClassesNoSilverNewGaz':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'PosEncod3ClassesNoSilverBig':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'PosEncod3ClassesNoSilverSpanish':
        dis_gaz_embeddings = torch.tensor(util.expand_labels(batch_encoding, util.get_dis_gaz_one_hot(sample_data)),
                                          device=device)
        umls_dis_gaz_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_umls_dis_gaz_one_hot(sample_data)),
            device=device)
        silver_dis_embeddings = torch.tensor(
            util.expand_labels(batch_encoding, util.get_silver_dis_one_hot(sample_data)),
            device=device)
        # silver embeddings are going to be ignored during training
        model_input = (batch_encoding, dis_gaz_embeddings, umls_dis_gaz_embeddings, silver_dis_embeddings)
    elif model_config.model_name == 'PosEncod3ClassesOnlyRoberta':
        model_input = [batch_encoding]
    elif model_config.model_name == 'OnlyRoberta3Classes':
        model_input = [batch_encoding]
    elif model_config.model_name == 'JustBert3Classes':
        model_input = [batch_encoding]
    else:
        raise Exception('Not implemented!')
    return model_input


def prepare_model(model_config: ModelConfig, dataset_config: DatasetConfig):
    # if dataset_config['model_name'] == 'SeqLabelerAllResourcesSmallerTopK':
    #     return SeqLabelerAllResourcesSmallerTopK(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
    #                                              pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    # if dataset_config['model_name'] == 'SeqLabelerDisGaz':
    #     return SeqLabelerDisGaz(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
    #                             pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    # if dataset_config['model_name'] == 'SeqLabelerUMLSDisGaz':
    #     return SeqLabelerUMLSDisGaz(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
    #                                 pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    # if dataset_config['model_name'] == 'SeqLabelerUMLSDisGaz3Classes':
    #     return SeqLabelerUMLSDisGaz3Classes(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
    #                                         pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    # if dataset_config['model_name'] == 'Silver3Classes':
    #     return Silver3Classes(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
    #                           pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
    # if dataset_config['model_name'] == 'LightWeightRIM3Classes':
    #     return LightWeightRIM3Classes().to(device)
    # if dataset_config['model_name'] == 'OneEncoder3Classes':
    #     return OneEncoder3Classes().to(device)
    # if dataset_config['model_name'] == 'TransformerEncoder3Classes':
    #     return TransformerEncoder3Classes().to(device)
    # if dataset_config['model_name'] == 'PositionalTransformerEncoder3Classes':
    #     return PositionalTransformerEncoder3Classes().to(device)
    # if dataset_config['model_name'] == 'SmallPositionalTransformerEncoder3Classes':
    #     return SmallPositionalTransformerEncoder3Classes().to(device)
    # if dataset_config['model_name'] == 'ComprehensivePositionalTransformerEncoder3Classes':
    #     return ComprehensivePositionalTransformerEncoder3Classes(umls_pretrained=umls_embedding_dict,
    #                                                              umls_to_idx=umls_key_to_index,
    #                                                              pos_pretrained=pos_dict, pos_to_idx=pos_to_index) \
    #         .to(device)
    # if dataset_config['model_name'] == 'PosEncod3ClassesNoSilverNewGaz':
    #     return PosEncod3ClassesNoSilverNewGaz().to(device)
    # if dataset_config['model_name'] == 'PosEncod3ClassesNoSilverBig':
    #     return PosEncod3ClassesNoSilverBig().to(device)
    # if dataset_config['model_name'] == 'PosEncod3ClassesNoSilverSpanish':
    #     return PosEncod3ClassesNoSilverSpanish().to(device)
    # if dataset_config['model_name'] == 'PosEncod3ClassesOnlyRoberta':
    #     return PosEncod3ClassesOnlyRoberta().to(device)
    # if dataset_config['model_name'] == 'OnlyRoberta3Classes':
    #     return OnlyRoberta3Classes().to(device)
    if model_config.model_name == 'JustBert3Classes':
        all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
        return JustBert3Classes(all_types, model_config, dataset_config).to(device)
    if model_config.model_name == 'SpanBert':
        all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
        return SpanBert(all_types, model_config).to(device)
    raise Exception(f"no code to prepare model {model_config.model_name}")


# TODO: move to different module
def get_train_token_data_dict(dataset_config: DatasetConfig) -> Dict[SampleId, List[TokenData]]:
    return util.get_tokens_from_file(dataset_config.train_tokens_file_path)


# TODO: move to different module
def get_valid_token_data_dict(dataset_config: DatasetConfig) -> Dict[SampleId, List[TokenData]]:
    return util.get_tokens_from_file(dataset_config.valid_tokens_file_path)


# TODO: move to different module
def get_test_data(dataset_config: DatasetConfig) -> Dict[str, List[TokenData]]:
    return util.read_data_from_folder(dataset_config.test_data_folder_path)


# TODO: move to different module
def get_train_texts(dataset_config: DatasetConfig) -> Dict[SampleId, str]:
    return util.get_texts(dataset_config.train_sample_text_data_file_path)


# TODO: move to different module
def get_valid_texts(dataset_config: DatasetConfig) -> Dict[SampleId, str]:
    return util.get_texts(dataset_config.valid_sample_text_data_file_path)


def prepare_file_headers(mistakes_file_writer, predictions_file_writer):
    predictions_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction']
    predictions_file_writer.writerow(predictions_file_header)

    mistakes_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction', 'mistake_type']
    mistakes_file_writer.writerow(mistakes_file_header)


def store_predictions(
        sample_id: str,
        token_data_valid: List[TokenData],
        predicted_annos_valid: List[Anno],
        predictions_file_writer
):
    # write predictions
    for anno in predicted_annos_valid:
        extraction = util.get_extraction(token_data_valid, anno.begin_offset, anno.end_offset)
        predictions_file_writer.writerow(
            [sample_id, str(anno.begin_offset), str(anno.end_offset), anno.label_type, extraction]
        )


def store_mistakes(
        sample_id: str,
        false_positives,
        false_negatives,
        mistakes_file_writer,
        token_data_valid: List[TokenData],
):
    # write false positive errors
    for span in false_positives:
        start_offset = span[0]
        end_offset = span[1]
        extraction = util.get_extraction(token_data_valid, start_offset, end_offset)
        mistakes_file_writer.writerow(
            [sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FP']
        )
    # write false negative errors
    for span in false_negatives:
        start_offset = span[0]
        end_offset = span[1]
        extraction = util.get_extraction(token_data_valid, start_offset, end_offset)
        mistakes_file_writer.writerow(
            [sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FN']
        )
