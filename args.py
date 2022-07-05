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
    "include_umls": True,
    "testing_mode": True
}
