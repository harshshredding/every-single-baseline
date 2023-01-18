from typing import NamedTuple
import yaml


class DatasetConfig(NamedTuple):
    train_annos_file_path: str
    valid_annos_file_path: str
    train_tokens_file_path: str
    valid_tokens_file_path: str
    train_sample_text_data_file_path: str
    valid_sample_text_data_file_path: str
    types_file_path: str
    num_types: int
    dataset_name: str


class ModelConfig(NamedTuple):
    bert_model_name: str
    bert_model_output_dim: int
    num_epochs: int
    save_models_dir: str
    model_name: str
    optimizer: str
    learning_rate: float


def read_dataset_config(config_file_path: str) -> DatasetConfig:
    with open(config_file_path, 'r') as yaml_file:
        dataset_config_raw = yaml.safe_load(yaml_file)
        dataset_config = DatasetConfig(
            train_annos_file_path=dataset_config_raw['train_annos_file_path'],
            valid_annos_file_path=dataset_config_raw['valid_annos_file_path'],
            train_tokens_file_path=dataset_config_raw['train_tokens_file_path'],
            valid_tokens_file_path=dataset_config_raw['valid_tokens_file_path'],
            train_sample_text_data_file_path=dataset_config_raw['train_sample_text_data_file_path'],
            valid_sample_text_data_file_path=dataset_config_raw['valid_sample_text_data_file_path'],
            types_file_path=dataset_config_raw['types_file_path'],
            num_types=int(dataset_config_raw['num_types']),
            dataset_name=dataset_config_raw['dataset_name']
        )
        assert isinstance(dataset_config.num_types, int)
        return dataset_config


def read_model_config(model_config_file_path: str) -> ModelConfig:
    with open(model_config_file_path, 'r') as yaml_file:
        model_config_raw = yaml.safe_load(yaml_file)
        assert len(model_config_raw) == 7, "model config should only have 7 attributes currently"
        model_config = ModelConfig(
            bert_model_name=model_config_raw['bert_model_name'],
            bert_model_output_dim=int(model_config_raw['bert_model_output_dim']),
            num_epochs=int(model_config_raw['num_epochs']),
            save_models_dir=model_config_raw['save_models_dir'],
            model_name=model_config_raw['model_name'],
            optimizer=model_config_raw['optimizer'],
            learning_rate=float(model_config_raw['learning_rate'])
        )
        assert isinstance(model_config.bert_model_output_dim, int)
        assert isinstance(model_config.num_epochs, int)
        assert isinstance(model_config.learning_rate, float)
        return model_config
