from structs import Dataset

dataset_config_list = [
    # LEGAL EVAL JUDGEMENT
    {
        "train_annos_file_path": f"./preprocessed_data/legaleval_train_judgement_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/legaleval_valid_judgement_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/legaleval_train_judgement_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/legaleval_valid_judgement_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/legaleval_train_judgement_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/legaleval_valid_judgement_sample_text.json",
        "types_file_path": "./preprocessed_data/legaleval_train_judgement_types.txt",
        "num_types": 14,
        # MODEL DETAILS
        "bert_model_name": "bert-base-cased",
        "bert_model_output_dim": 768,
        "num_epochs": 10,
        "save_models_dir": "./models",
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset": Dataset.legaleval_judgement
    },
    # LEGAL EVAL PREAMBLE
    {
        "train_annos_file_path": f"./preprocessed_data/legaleval_train_preamble_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/legaleval_valid_preamble_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/legaleval_train_preamble_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/legaleval_valid_preamble_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/legaleval_train_preamble_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/legaleval_valid_preamble_sample_text.json",
        "types_file_path": "./preprocessed_data/legaleval_train_preamble_types.txt",
        "num_types": 14,
        # MODEL DETAILS
        "bert_model_name": "bert-base-cased",
        "bert_model_output_dim": 768,
        "num_epochs": 15,
        "save_models_dir": "./models",
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset": Dataset.legaleval_preamble
    },
    # MULTICONER COARSE
    {
        "train_annos_file_path": f"./preprocessed_data/multiconer_train_coarse_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/multiconer_valid_coarse_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/multiconer_train_coarse_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/multiconer_valid_coarse_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/multiconer_train_coarse_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/multiconer_valid_coarse_sample_text.json",
        "types_file_path": "./preprocessed_data/multiconer_train_coarse_types.txt",
        "num_types": 6,
        # MODEL DETAILS
        "bert_model_name": "bert-base-uncased",
        "bert_model_output_dim": 768,
        "num_epochs": 10,
        "save_models_dir": "./models",
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset": Dataset.multiconer_coarse,
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
    },
    # MULTICONER FINE
    {
        "train_annos_file_path": f"./preprocessed_data/multiconer_train_fine_annos.tsv",
        "valid_annos_file_path": f"./preprocessed_data/multiconer_valid_fine_annos.tsv",
        "train_tokens_file_path": f"./preprocessed_data/multiconer_train_fine_tokens.json",
        "valid_tokens_file_path": f"./preprocessed_data/multiconer_valid_fine_tokens.json",
        "train_sample_text_data_file_path": f"./preprocessed_data/multiconer_train_fine_sample_text.json",
        "valid_sample_text_data_file_path": f"./preprocessed_data/multiconer_valid_fine_sample_text.json",
        "types_file_path": "./preprocessed_data/multiconer_train_fine_types.txt",
        "num_types": 33,
        # MODEL DETAILS
        "bert_model_name": "bert-base-uncased",
        "bert_model_output_dim": 768,
        "num_epochs": 10,
        "save_models_dir": "./models",
        "errors_dir": './errors',
        "model_name": "JustBert3Classes",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "dataset": Dataset.multiconer_fine,
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
]