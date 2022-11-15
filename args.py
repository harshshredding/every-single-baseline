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
curr_dataset = Dataset.legaleval
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
        "train_annos_file_path": "./datasets/genia-dataset/gold-annos/annos.tsv",
        "valid_annos_file_path": "./datasets/genia-dataset/gold-annos/annos.tsv",
        "training_data_folder_path": "./datasets/genia-dataset/input-files",
        "gate_input_folder_path": "./datasets/genia-dataset/gate-input",
        "types_file_path": "./datasets/genia-dataset/types.txt",
        "num_types": 5,
        # "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_name": "xlm-roberta-large",
        "bert_model_output_dim": 1024,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "umls_embeddings_path": "./embeddings.csv",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "OnlyRoberta3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": "GENIA"
    }
elif curr_dataset == Dataset.social_dis_ner:
    args = {
        "train_annos_file_path": "./datasets/social-dis-ner-dataset/mentions.tsv",
        "valid_annos_file_path": "./datasets/social-dis-ner-dataset/mentions.tsv",
        "training_data_folder_path": f"./datasets/social-dis-ner-dataset/gate-output-no-custom-tokenization/train",
        "validation_data_folder_path": f"./datasets/social-dis-ner-dataset/gate-output-no-custom-tokenization/valid",
        "types_file_path": "./datasets/social-dis-ner-dataset/types.txt",
        "num_types": 1,
        # "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_name": "xlm-roberta-large",
        "bert_model_output_dim": 1023,
        "num_epochs": 14,
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
        "dataset_name": "social-dis-ner"
    }
elif curr_dataset == Dataset.multiconer:
    granularity = get_user_input('specify multiconer label granularity', ['coarse', 'fine'])
    args = {
        "train_annos_file_path": f"./datasets/multiconer/gold-annos/train/{granularity}/annos-train.tsv",
        "valid_annos_file_path": f"./datasets/multiconer/gold-annos/valid/{granularity}/annos-valid.tsv",
        "training_data_folder_path": f"./datasets/multiconer/input-files/train/{granularity}",
        "validation_data_folder_path": f"./datasets/multiconer/input-files/valid/{granularity}",
        "gate_input_folder_path": "./datasets/multiconer/gate-input",
        "types_file_path": f"./datasets/multiconer/{granularity}/types.txt",
        "num_types": 33 if granularity == 'fine' else 6,
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
            'Coarse_Location':['Facility', 'OtherLOC', 'HumanSettlement', 'Station'],
            'Coarse_Creative_Work': ['VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software', 'OtherCW'],
            'Coarse_Group': ['MusicalGRP', 'PublicCorp', 'PrivateCorp', 'OtherCorp', 'AerospaceManufacturer', 'SportsGRP', 'CarManufacturer', 'TechCorp', 'ORG'],
            'Coarse_Person': ['Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric', 'SportsManager', 'OtherPER'],
            'Coarse_Product': ['Clothing', 'Vehicle', 'Food', 'Drink', 'OtherPROD'],
            'Coarse_Medical': ['Medication/Vaccine', 'MedicalProcedure', 'AnatomicalStructure', 'Symptom', 'Disease'],
            'O': ['O']
        }
    }
elif curr_dataset == Dataset.legaleval:
    section = get_user_input('specify section', ['JUDGEMENT', 'PREAMBLE'])
    args = {
        "train_annos_file_path": f"./datasets/legaleval/gold-annos/train/{section}/annos-{section}-train.tsv",
        "valid_annos_file_path": f"./datasets/legaleval/gold-annos/valid/{section}/annos-{section}-valid.tsv",
        "gate_input_folder_path": "./datasets/legaleval/gate-input",
        "training_data_folder_path": f"./datasets/legaleval/input-files/train/{section}",
        "validation_data_folder_path": f"./datasets/legaleval/input-files/valid/{section}",
        "types_file_path": "./datasets/legaleval/types.txt",
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
else:
    raise Exception(f'Dataset {curr_dataset} is not supported')
default_key = "DEFAULT"
