from re import L
import time

from structs import *
import pandas as pd
import csv
import util
from utils.config import ExperimentConfig
from structs import EvaluationType
from typing import Dict
from utils.config import ModelConfig, DatasetConfig
import utils.dropbox as dropbox_util
import argparse
import glob
from pathlib import Path
from dataclasses import dataclass
from pydoc import locate
from preamble import *
from transformers import AutoTokenizer
import transformers
import torch.nn as nn
from pyfzf.pyfzf import FzfPrompt
from utils.evaluation_general import f1


def has_external_features(samples: list[Sample]) -> bool:
    for sample in samples:
        if len(sample.annos.external):
            return True
    return False

def check_external_features(samples: list[Sample], external_feature_type: str):
    for sample in samples:
        for anno in sample.annos.external:
            if anno.label_type == external_feature_type:
                return True
    raise RuntimeError(f"External Feature {external_feature_type} not found in data")


def get_all_model_modules():
    all_module_paths = [file_path for file_path in glob.glob('./models/*.py') 
                        if '__init__.py' not in file_path]
    return [Path(module_path).stem for module_path in all_module_paths]


def get_experiment_name_from_user() -> str:
    all_experiment_file_paths = glob.glob('./experiments/*.py')
    all_experiment_names = [Path(file_path).stem for file_path in all_experiment_file_paths]
    # ignore the init file
    all_experiment_names.remove('__init__')
    assert all(experiment_name.startswith('experiment') for experiment_name in all_experiment_names)

    # use fzf to select an experiment
    fzf = FzfPrompt()
    chosen_experiment = fzf.prompt(all_experiment_names)[0]
    return chosen_experiment


@dataclass
class TrainingArgs:
    device: torch.device
    is_dry_run_mode: bool
    experiment_name: str
    is_testing: bool


def parse_training_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description='Train models and store their output for inspection.')
    parser.add_argument('--production', action='store_true',
                        help='start training on ALL data (10 samples only by default)')
    parser.add_argument('--test', action='store_true', help="Evaluate on the test dataset.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    experiment_name = get_experiment_name_from_user()

    return TrainingArgs(device, not args.production, experiment_name, args.test)


def print_experiment_info(
        experiment_config: ExperimentConfig,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        experiment_name: str,
        is_dry_run: bool,
        is_testing: bool,
        test_evaluation_frequency: int,
) -> None:
    """Print the configurations of the current run"""
    print("\n\n------ EXPERIMENT OVERVIEW --------")
    print(blue("Experiment:"), green(experiment_name))
    print(blue("DRY_RUN_MODE:"), green(is_dry_run))
    print(blue("Is Testing:"), green(is_testing))
    print(blue("Dataset Config Name:"), green(dataset_config.dataset_config_name))
    print(blue("Dataset:"), green(dataset_config.dataset_name))
    print(blue("Model Config Name:"), green(model_config.model_config_name))
    print(blue("Model Name:"), green(model_config.model_name))
    print(blue("Batch_size"), green(model_config.batch_size))
    print(blue("Testing Frequency"), green(test_evaluation_frequency))
    print("----------------------------\n\n")
    print("Experiment Config")
    print(experiment_config)
    print()
    print("Model Config:")
    print(model_config)
    print()
    print("Dataset Config:")
    print(dataset_config)
    print()


def check_label_types(train_samples: List[Sample], valid_samples: List[Sample], all_types: List[str]):
    # verify that all label types in annotations are valid types
    for sample in train_samples:
        for anno in sample.annos.gold:
            assert anno.label_type in all_types, f"anno label type {anno.label_type} not expected"
    for sample in valid_samples:
        for anno in sample.annos.gold:
            assert anno.label_type in all_types, f"anno label type {anno.label_type} not expected"


def get_batches(samples: List[Sample], batch_size: int) -> list[list[Sample]]:
    return [
        samples[batch_start_idx: batch_start_idx + batch_size]
        for batch_start_idx in range(0, len(samples), batch_size)
    ]


def get_loss_function():
    return nn.CrossEntropyLoss()


def get_bert_tokenizer(model_config: ModelConfig):
    """
    Get the bert tokenizer
    """
    return AutoTokenizer.from_pretrained(model_config.pretrained_model_name)


def get_train_annos_dict(dataset_config: DatasetConfig) -> Dict[str, List[Annotation]]:
    return util.get_annos_dict(dataset_config.train_annos_file_path)


def get_valid_annos_dict(dataset_config: DatasetConfig) -> Dict[str, List[Annotation]]:
    return util.get_annos_dict(dataset_config.valid_annos_file_path)


def get_bio_labels_from_annos(token_annos: List[Option[Annotation]],
                              batch_encoding,
                              gold_annos: List[Annotation]) -> List[Label]:
    labels = util.get_labels_bio_old(token_annos, gold_annos)
    expanded_labels = util.expand_labels_rich_batch(batch_encoding, labels, batch_idx=0)
    return expanded_labels


def get_bio_labels_from_annos_batch(
        token_annos_batch: List[List[Option[Annotation]]],
        batch_encoding,
        gold_annos_batch: List[List[Annotation]]
) -> List[List[Label]]:
    labels_batch = [util.get_labels_bio_old(token_annos, gold_annos)
                    for token_annos, gold_annos in zip(token_annos_batch, gold_annos_batch)]
    expanded_labels_batch = [util.expand_labels_rich_batch(batch_encoding=batch_encoding, labels=labels, batch_idx=i)
                             for i, labels in enumerate(labels_batch)
                             ]
    return expanded_labels_batch


def get_bio_labels_for_bert_tokens_batch(
        token_annos_batch: List[List[Option[Annotation]]],
        gold_annos_batch: List[List[Annotation]]
):
    labels_batch = [util.get_labels_bio_new(token_annos, gold_annos)
                    for token_annos, gold_annos in zip(token_annos_batch, gold_annos_batch)]
    return labels_batch


# TODO: remove following method
def read_pos_embeddings_file(dataset_config: DatasetConfig):
    return pd.read_pickle(dataset_config['pos_embeddings_path'])


def get_optimizer(model, experiment_config: ExperimentConfig):
    if experiment_config.optimizer == 'Ranger':
        # return torch_optimizer.Ranger(model.parameters(), dataset_config['learning_rate'])
        raise Exception("no ranger optimizer")
    elif experiment_config.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), experiment_config.model_config.learning_rate)
    elif experiment_config.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), experiment_config.model_config.learning_rate)
    elif experiment_config.optimizer == 'Adafactor':
        return transformers.Adafactor(
            model.parameters(),
            lr=experiment_config.model_config.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    else:
        raise Exception(f"optimizer not found: {experiment_config.optimizer}")


def store_performance_result(
    performance_file_path,
    f1_score,
    precision_score,
    recall_score,
    epoch: int,
    experiment_name: str,
    dataset_config_name: str,
    model_config_name: str,
    dataset_split: DatasetSplit
):
    with open(performance_file_path, 'a') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow([experiment_name, dataset_config_name, dataset_split.name,
                                       model_config_name, str(epoch),
                                       str((f1_score, precision_score, recall_score))])


def store_performance_result_accuracy(
    performance_file_path,
    accuracy,
    epoch: int,
    experiment_name: str,
    dataset_config_name: str,
    model_config_name: str,
    dataset_split: DatasetSplit
):
    with open(performance_file_path, 'a') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow([experiment_name, dataset_config_name, dataset_split.name,
                                       model_config_name, str(epoch),
                                       str(accuracy)])


def create_performance_file_header(performance_file_path):
    with open(performance_file_path, 'w') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow(['experiment_name', 'dataset_config_name', 'dataset_split', 'model_name', 'epoch', 'f1_score'])


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


def get_spans_from_bio_seq_labels(bio_labels, batch_encoding, batch_idx: int):
    return util.get_spans_from_bio_labels(bio_labels, batch_encoding, batch_idx=batch_idx)




# TODO: remove following method
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

def get_model_class(model_config: ModelConfig) -> type:
    model_class = None
    for model_module_name in get_all_model_modules():
        if model_class is None:
            model_class = locate(f"models.{model_module_name}.{model_config.model_name}")
        else:
            if locate(f"models.{model_module_name}.{model_config.model_name}") is not None:
                print(red(f"WARN: Found duplicate model class: {model_config.model_name}"))
    assert model_class is not None, f"model class name {model_config.model_name} could not be found"
    return model_class


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
    all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
    model_class = get_model_class(model_config=model_config)
    return model_class(all_types, model_config=model_config, dataset_config=dataset_config).to(device)


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


def get_train_samples(dataset_config: DatasetConfig) -> List[Sample]:
    return util.read_samples(dataset_config.train_samples_file_path)


def get_valid_samples(dataset_config: DatasetConfig) -> List[Sample]:
    return util.read_samples(dataset_config.valid_samples_file_path)


def get_test_samples(dataset_config: DatasetConfig) -> List[Sample]:
    return util.read_samples(dataset_config.test_samples_file_path)


def prepare_file_headers_accuracy(mistakes_file_writer, predictions_file_writer):
    predictions_file_header = ['sample_id', 'label']
    predictions_file_writer.writerow(predictions_file_header)

    mistakes_file_header = ['sample_id', 'label']
    mistakes_file_writer.writerow(mistakes_file_header)

def prepare_file_headers(mistakes_file_writer, predictions_file_writer):
    prepare_predictions_file_header(predictions_file_writer)

    mistakes_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction', 'mistake_type']
    mistakes_file_writer.writerow(mistakes_file_header)


def prepare_predictions_file_header(predictions_file_writer):
    predictions_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction']
    predictions_file_writer.writerow(predictions_file_header)


def store_predictions(
        sample: Sample,
        predicted_annos_valid: List[Annotation],
        predictions_file_writer
):
    # write predictions
    for anno in predicted_annos_valid:
        predictions_file_writer.writerow(
            [sample.id, str(anno.begin_offset), str(anno.end_offset), anno.label_type, anno.extraction]
        )


def store_prediction_accuracy(
        sample: Sample,
        predicted_anno: str,
        predictions_file_writer
):
    # write predictions
    predictions_file_writer.writerow(
        [sample.id, predicted_anno]
    )


def store_mistakes(
        sample: Sample,
        false_positives,
        false_negatives,
        mistakes_file_writer
):
    # write false positive errors
    for span in false_positives:
        start_offset = span[0]
        end_offset = span[1]
        extraction = sample.text[start_offset: end_offset]
        mistakes_file_writer.writerow(
            [sample.id, str(start_offset), str(end_offset), span[2], extraction, 'FP']
        )
    # write false negative errors
    for span in false_negatives:
        start_offset = span[0]
        end_offset = span[1]
        extraction = sample.text[start_offset: end_offset]
        mistakes_file_writer.writerow(
            [sample.id, str(start_offset), str(end_offset), span[2], extraction, 'FN']
        )


def evaluate_validation_split(
        logger,
        model: torch.nn.Module,
        validation_samples: List[Sample],
        mistakes_folder_path: str,
        predictions_folder_path: str,
        error_visualization_folder_path: str,
        validation_performance_file_path: str,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        epoch: int,
        experiment_idx: int,
        evaluation_type: EvaluationType
):
    evaluate_dataset_split(
        logger=logger,
        model=model,
        samples=validation_samples,
        mistakes_folder_path=mistakes_folder_path,
        predictions_folder_path=predictions_folder_path,
        error_visualization_folder_path=error_visualization_folder_path,
        performance_file_path=validation_performance_file_path,
        experiment_name=experiment_name,
        dataset_config_name=dataset_config_name,
        model_config_name=model_config_name,
        epoch=epoch,
        dataset_split=DatasetSplit.valid,
        experiment_idx=experiment_idx,
        evaluation_type=evaluation_type
    )


def evaluate_test_split(
        logger,
        model: torch.nn.Module,
        test_samples: List[Sample],
        mistakes_folder_path: str,
        predictions_folder_path: str,
        error_visualization_folder_path: str,
        test_performance_file_path: str,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        epoch: int,
        experiment_idx: int,
        evaluation_type: EvaluationType
):
    evaluate_dataset_split(
        logger=logger,
        model=model,
        samples=test_samples,
        mistakes_folder_path=mistakes_folder_path,
        predictions_folder_path=predictions_folder_path,
        error_visualization_folder_path=error_visualization_folder_path,
        performance_file_path=test_performance_file_path,
        experiment_name=experiment_name,
        dataset_config_name=dataset_config_name,
        model_config_name=model_config_name,
        epoch=epoch,
        dataset_split=DatasetSplit.test,
        experiment_idx=experiment_idx,
        evaluation_type=evaluation_type
    )


def evaluate_dataset_split(
        logger,
        model: torch.nn.Module,
        samples: List[Sample],
        mistakes_folder_path: str,
        predictions_folder_path: str,
        error_visualization_folder_path: str,
        performance_file_path: str,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        epoch: int,
        dataset_split: DatasetSplit,
        experiment_idx: int,
        evaluation_type: EvaluationType
):
    logger.info(f"\n\nEvaluating {dataset_split.name} data")
    model.eval()
    output_file_prefix = f"{experiment_name}_{experiment_idx}_{dataset_config_name}_{model_config_name}_{dataset_split.name}" \
                         f"_epoch_{epoch}"
    mistakes_file_path = f"{mistakes_folder_path}/{output_file_prefix}_mistakes.tsv"
    predictions_file_path = f"{predictions_folder_path}/{output_file_prefix}_predictions.tsv"

    if evaluation_type == EvaluationType.f1:
        evaluate_with_f1(
                predictions_file_path=predictions_file_path,
                mistakes_file_path=mistakes_file_path,
                samples=samples,
                model=model,
                logger=logger,
                performance_file_path=performance_file_path,
                error_visualization_folder_path=error_visualization_folder_path,
                output_file_prefix=output_file_prefix,
                epoch=epoch,
                experiment_name=experiment_name,
                dataset_config_name=dataset_config_name,
                model_config_name=model_config_name,
                dataset_split=dataset_split
                )
    elif evaluation_type == EvaluationType.accuracy:
        evaluate_with_accuracy(
                predictions_file_path=predictions_file_path,
                mistakes_file_path=mistakes_file_path,
                samples=samples,
                model=model,
                logger=logger,
                performance_file_path=performance_file_path,
                error_visualization_folder_path=error_visualization_folder_path,
                output_file_prefix=output_file_prefix,
                epoch=epoch,
                experiment_name=experiment_name,
                dataset_config_name=dataset_config_name,
                model_config_name=model_config_name,
                dataset_split=dataset_split
                )
        



def evaluate_with_accuracy(
        predictions_file_path: str,
        mistakes_file_path: str,
        samples: list[Sample],
        model,
        logger,
        performance_file_path: str,
        error_visualization_folder_path: str,
        output_file_prefix: str,
        epoch: int,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        dataset_split: DatasetSplit):

    evaluation_start_time = time.time()

    total_num_samples = len(samples)
    num_correct_labels = 0

    with open(predictions_file_path, 'w') as predictions_file, \
            open(mistakes_file_path, 'w') as mistakes_file:
        #  --- GET FILES READY FOR WRITING ---
        predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
        mistakes_file_writer = csv.writer(mistakes_file, delimiter='\t')
        prepare_file_headers_accuracy(mistakes_file_writer, predictions_file_writer)
        with torch.no_grad():
            # Eval Loop
            for sample in samples:
                loss, [predicted_anno] = model([sample])
                assert len(sample.annos.gold) == 1
                gold_anno = sample.annos.gold[0].label_type
                assert gold_anno in ['correct', 'incorrect']
                assert predicted_anno in ['correct', 'incorrect']
                if gold_anno == predicted_anno:
                    num_correct_labels += 1
                # write sample predictions
                store_prediction_accuracy(sample, predicted_anno, predictions_file_writer)
    accuracy = num_correct_labels/total_num_samples
    logger.info(blue(f"Accuracy: {accuracy}"))
    visualize_errors_file_path = f"{error_visualization_folder_path}/{output_file_prefix}_visualize_errors.bdocjs"
    store_performance_result_accuracy(
            performance_file_path=performance_file_path,
            epoch=epoch,
            experiment_name=experiment_name,
            dataset_config_name=dataset_config_name,
            model_config_name=model_config_name,
            dataset_split=dataset_split,
            accuracy=accuracy
    )
    # upload files to dropbox
    dropbox_util.upload_file(predictions_file_path)
    # dropbox_util.upload_file(mistakes_file_path)
    dropbox_util.upload_file(performance_file_path)

    logger.info(green(f"Done evaluating {dataset_split.name} data.\n"
                      f"Took {str(time.time() - evaluation_start_time)} secs."
                      f"\n\n"))


def evaluate_with_f1(
        predictions_file_path: str,
        mistakes_file_path: str,
        samples: list[Sample],
        model,
        logger,
        performance_file_path: str,
        error_visualization_folder_path: str,
        output_file_prefix: str,
        epoch: int,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        dataset_split: DatasetSplit):

    evaluation_start_time = time.time()

    with open(predictions_file_path, 'w') as predictions_file, \
            open(mistakes_file_path, 'w') as mistakes_file:
        #  --- GET FILES READY FOR WRITING ---
        predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
        mistakes_file_writer = csv.writer(mistakes_file, delimiter='\t')
        prepare_file_headers(mistakes_file_writer, predictions_file_writer)
        with torch.no_grad():
            num_TP_total = 0
            num_FP_total = 0
            num_FN_total = 0
            # Eval Loop
            for sample in samples:
                loss, [predicted_annos] = model([sample])
                gold_annos_set = set(
                    [
                        (gold_anno.begin_offset, gold_anno.end_offset, gold_anno.label_type)
                        for gold_anno in sample.annos.gold
                    ]
                )
                predicted_annos_set = set(
                    [
                        (predicted_anno.begin_offset, predicted_anno.end_offset, predicted_anno.label_type)
                        for predicted_anno in predicted_annos
                    ]
                )

                # calculate true positives, false positives, and false negatives
                true_positives_sample = gold_annos_set.intersection(predicted_annos_set)
                false_positives_sample = predicted_annos_set.difference(gold_annos_set)
                false_negatives_sample = gold_annos_set.difference(predicted_annos_set)
                num_TP = len(true_positives_sample)
                num_TP_total += num_TP
                num_FP = len(false_positives_sample)
                num_FP_total += num_FP
                num_FN = len(false_negatives_sample)
                num_FN_total += num_FN

                # write sample predictions
                store_predictions(sample, predicted_annos, predictions_file_writer)
                # write sample mistakes
                store_mistakes(sample, false_positives_sample, false_negatives_sample,
                               mistakes_file_writer)
    micro_f1, micro_precision, micro_recall = f1(num_TP_total, num_FP_total, num_FN_total)
    logger.info(blue(f"Micro f1 {micro_f1}, prec {micro_precision}, recall {micro_recall}"))
    visualize_errors_file_path = f"{error_visualization_folder_path}/{output_file_prefix}_visualize_errors.bdocjs"
    util.create_mistakes_visualization(mistakes_file_path, visualize_errors_file_path, samples)
    store_performance_result(
            performance_file_path=performance_file_path,
            f1_score=micro_f1,
            precision_score=micro_precision,
            recall_score=micro_recall,
            epoch=epoch,
            experiment_name=experiment_name,
            dataset_config_name=dataset_config_name,
            model_config_name=model_config_name,
            dataset_split=dataset_split
    )

    # upload files to dropbox
    dropbox_util.upload_file(visualize_errors_file_path)
    dropbox_util.upload_file(predictions_file_path)
    # dropbox_util.upload_file(mistakes_file_path)
    dropbox_util.upload_file(performance_file_path)

    logger.info(green(f"Done evaluating {dataset_split.name} data.\n"
                      f"Took {str(time.time() - evaluation_start_time)} secs."
                      f"\n\n"))



def test(
        logger,
        model: torch.nn.Module,
        test_samples: List[Sample],
        test_predictions_folder_path: str,
        experiment_name: str,
        dataset_name: str,
        model_config_name: str,
        epoch: int,
):
    logger.info("Starting Testing")
    model.eval()
    predictions_file_path = f"{test_predictions_folder_path}/{experiment_name}_{dataset_name}_{model_config_name}_" \
                            f"epoch_{epoch}" \
                            f"_test_predictions.tsv"
    with open(predictions_file_path, 'w') as predictions_file:
        #  --- GET FILES READY FOR WRITING ---
        predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
        prepare_predictions_file_header(predictions_file_writer)

        with torch.no_grad():
            # Test Loop
            for test_sample in test_samples:
                loss, predicted_annos_valid = model(test_sample)
                # write sample predictions
                store_predictions(test_sample, predicted_annos_valid, predictions_file_writer)
    # Upload predictions to dropbox
    dropbox_util.upload_file(predictions_file_path)
    logger.info(f"Done validating!\n\n\n")


def save_model(model, models_folder_path, epoch, EXPERIMENT):
    torch.save(model.state_dict(), f"{models_folder_path}/Epoch_{epoch}_{EXPERIMENT}")
