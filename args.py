import logging
import torch
from structs import *
from typing import List


def get_user_input(input_message: str, possible_values: List[str]):
    user_input = input(f"{input_message}\n choose from {possible_values}")
    if len(possible_values):
        while user_input not in possible_values:
            user_input = input(f"incorrect input '{user_input}', please choose from {possible_values}")
    return user_input


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
TESTING_MODE = True


if not TESTING_MODE:
    EXPERIMENT = get_user_input('specify experiment name:', [])
else:
    EXPERIMENT = 'test'


curr_dataset = Dataset.genia


if curr_dataset == Dataset.few_nerd:
    args = {
        "train_annos_file_path": "./datasets/few-nerd-dataset/gold-annos/few_nerd_train_annos.tsv",
        "valid_annos_file_path": "./datasets/few-nerd-dataset/gold-annos/few_nerd_dev_annos.tsv",
        "training_data_folder_path": f"./datasets/few-nerd-dataset/input_files_train{'_small' if TESTING_MODE else ''}",
        "validation_data_folder_path": f"./datasets/few-nerd-dataset/input_files_dev{'_small' if TESTING_MODE else ''}",
        "test_data_folder_path": f"./datasets/few-nerd-dataset/input_files_test{'_small' if TESTING_MODE else ''}",
        "types_file_path": "./datasets/few-nerd-dataset/types.txt",
        "num_types": 66,
        # "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_name": "xlm-roberta-large",
        "bert_model_output_dim": 1024,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "raw_validation_files_path": "./socialdisner-data/train-valid-txt-files/validation",
        "raw_train_files_path": "./socialdisner-data/train-valid-txt-files/training",
        "raw_test_files_path": "./test_data/test-data/test-data-txt-files",
        "umls_embeddings_path": "./embeddings.csv",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "OnlyRoberta3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": "few_nerd"
    }
elif curr_dataset == Dataset.genia:
    args = {
        "train_annos_file_path": "./preprocessed_data/genia_train_annos.tsv",
        "valid_annos_file_path": "./preprocessed_data/genia_valid_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/genia_train_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/genia_valid_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/genia_train_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/genia_valid_text.json",
        "types_file_path": "./preprocessed_data/genia_train_types.txt",
        "num_types": 5,
        # "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_name": "bert-base-cased",
        "bert_model_output_dim": 768,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "errors_dir": './errors',
        "model_name": "SpanBert",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": "GENIA"
    }
elif curr_dataset == Dataset.social_dis_ner:
    args = {
        "train_annos_file_path": f"./preprocessed_data/social_dis_ner_train_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/social_dis_ner_valid_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/social_dis_ner_train_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/social_dis_ner_valid_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/social_dis_ner_train_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/social_dis_ner_valid_sample_text.json",
        "types_file_path": "./preprocessed_data/social_dis_ner_train_types.txt",
        "num_types": 1,
        # MODEL DETAILS 
        "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_output_dim": 768,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "raw_validation_files_path": "./socialdisner-data/train-valid-txt-files/validation",
        "raw_train_files_path": "./socialdisner-data/train-valid-txt-files/training",
        "raw_test_files_path": "./test_data/test-data/test-data-txt-files",
        "umls_embeddings_path": "./embeddings.csv",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "SpanBert",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": "social-dis-ner"
    }
elif curr_dataset == Dataset.multiconer:
    granularity = get_user_input('specify multiconer label granularity', ['coarse', 'fine'])
    args = {
        "train_annos_file_path": f"./preprocessed_data/multiconer_train_coarse_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/multiconer_valid_coarse_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/multiconer_train_coarse_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/multiconer_valid_coarse_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/multiconer_train_coarse_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/multiconer_valid_coarse_sample_text.json",
        "types_file_path": "./preprocessed_data/multiconer_train_coarse_types.txt",
        "num_types": 6,
        # "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_name": "bert-base-uncased",
        "granularity": granularity,
        "bert_model_output_dim": 768,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "umls_embeddings_path": "./embeddings.csv",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": "multiconer",
        "coarse_to_fine_dict": {
            'Coarse_Location': ['Facility', 'OtherLOC', 'HumanSettlement', 'Station'],
            'Coarse_Creative_Work': ['VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software', 'OtherCW'],
            'Coarse_Group': ['MusicalGRP', 'PublicCorp', 'PrivateCorp', 'OtherCorp', 'AerospaceManufacturer',
                             'SportsGRP', 'CarManufacturer', 'TechCorp', 'ORG'],
            'Coarse_Person': ['Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric', 'SportsManager', 'OtherPER'],
            'Coarse_Product': ['Clothing', 'Vehicle', 'Food', 'Drink', 'OtherPROD'],
            'Coarse_Medical': ['Medication/Vaccine', 'MedicalProcedure', 'AnatomicalStructure', 'Symptom', 'Disease'],
            'O': ['O']
        }
    }
elif curr_dataset == Dataset.legaleval:
    section = get_user_input('specify section', ['JUDGEMENT', 'PREAMBLE'])
    args = {
        "train_annos_file_path": f"./preprocessed_data/legaleval_train_judgement_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/legaleval_valid_judgement_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/legaleval_train_judgement_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/legaleval_valid_judgement_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/legaleval_train_judgement_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/legaleval_valid_judgement_sample_text.json",
        "types_file_path": "./preprocessed_data/legaleval_train_judgement_types.txt",
        "num_types": 14,
        # MODEL DETAILS
        "bert_model_name": "bert-base-uncased",
        "bert_model_output_dim": 768,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "umls_embeddings_path": "./embeddings.csv",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": curr_dataset.name,
    }
elif curr_dataset == Dataset.living_ner:
    args = {
        "train_annos_file_path": f"./preprocessed_data/living_ner_train_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/living_ner_valid_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/living_ner_train_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/living_ner_valid_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/living_ner_train_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/living_ner_valid_sample_text.json",
        "types_file_path": "./preprocessed_data/living_ner_train_types.txt",
        "num_types": 2,
        # MODEL DETAILS
        "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_output_dim": 768,
        "num_epochs": 20,
        "save_models_dir": "./models",
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": curr_dataset.name,
    }
else:
    raise Exception(f'Dataset {curr_dataset} is not supported')
default_key = "DEFAULT"
