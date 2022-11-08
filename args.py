import torch
from structs import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
TESTING_MODE = True
EXPERIMENT = 'social-dis-ner-test'
curr_dataset = Dataset.multiconer
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
    args = {
        "train_annos_file_path": "./datasets/multiconer/gold-annos/annos.tsv",
        "valid_annos_file_path": "./datasets/multiconer/gold-annos/annos.tsv",
        "training_data_folder_path": "./datasets/multiconer/input-files",
        "gate_input_folder_path": "./datasets/multiconer/gate-input",
        "types_file_path": "./datasets/multiconer/types.txt",
        "num_types": 33,
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
        "dataset_name": "multiconer"
    }
else:
    raise Exception(f'Dataset {curr_dataset} is not supported')
default_key = "DEFAULT"
