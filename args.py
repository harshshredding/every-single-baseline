import torch
from structs import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
testing_mode = True
curr_dataset = Dataset.few_nerd
if curr_dataset == Dataset.few_nerd:
    args = {
        "train_annos_file_path": "./few-nerd-dataset/gold-annos/few_nerd_train_annos.tsv",
        "valid_annos_file_path": "./few-nerd-dataset/gold-annos/few_nerd_dev_annos.tsv",
        "training_data_folder_path": f"./few-nerd-dataset/input_files_train{'_small' if testing_mode else ''}",
        "validation_data_folder_path": f"./few-nerd-dataset/input_files_dev{'_small' if testing_mode else ''}",
        "test_data_folder_path": f"./few-nerd-dataset/input_files_test{'_small' if testing_mode else ''}",
        "types_file_path": "./few-nerd-dataset/types.txt",
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
        "experiment_name": "few_nerd_baseline",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "OnlyRoberta3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset_name": "few_nerd"
    }
elif curr_dataset == Dataset.social_dis_ner:
    args = {
        "gold_file_path": "./mentions.tsv",
        "silver_file_path": "./silver_disease_mentions.tsv",
        "training_data_folder_path": "./gate-output-no-custom-tokenization/train",
        "validation_data_folder_path": "./gate-output-no-custom-tokenization/valid",
        "test_data_folder_path": "./gate-output-test",
        # "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "bert_model_name": "xlm-roberta-large",
        "bert_model_output_dim": 1023,
        "num_epochs": 14,
        "save_models_dir": "./models",
        "raw_validation_files_path": "./socialdisner-data/train-valid-txt-files/validation",
        "raw_train_files_path": "./socialdisner-data/train-valid-txt-files/training",
        "raw_test_files_path": "./test_data/test-data/test-data-txt-files",
        "umls_embeddings_path": "./embeddings.csv",
        "experiment_name": "only_roberta_without_custom_tokenization",
        "pos_embeddings_path": './spanish_pos_emb.p',
        "disease_gazetteer_path": './dictionary_distemist.tsv',
        "errors_dir": './errors',
        "model_name": "OnlyRoberta2Classes",
        "optimizer": "Adam",
        "learning_rate": 0e-5,
        "dataset_name": "few_nerd"
    }
else:
    raise Exception(f'Dataset {curr_dataset} is not supported')
default_key = "DEFAULT"
