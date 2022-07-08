import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
args = {
    "annotations_file_path": "./mentions.tsv",
    "training_data_folder_path": "./gate-output/train",
    "validation_data_folder_path": "./gate-output/valid",
    "bert_model_name": "dccuchile/bert-base-spanish-wwm-cased",
    "num_epochs": 10,
    "save_models_dir": "./models",
    "raw_validation_files_path": "./socialdisner-data/train-valid-txt-files/validation",
    "raw_train_files_path": "./socialdisner-data/train-valid-txt-files/training",
    "umls_embeddings_path": "/home/claclab/embeddings.csv",
    "resources": True,
    "testing_mode": True,
    "experiment_name": "include_pos",
    "pos_embeddings_path": './spanish_pos_emb.p',
    "disease_gazetteer_path": './dictionary_distemist.tsv',
}
default_key = "DEFAULT"
