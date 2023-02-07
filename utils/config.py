from typing import NamedTuple
from utils.universal import die
import yaml
import glob
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    train_samples_file_path: str
    valid_samples_file_path: str
    test_samples_file_path: str
    types_file_path: str
    num_types: int
    dataset_name: str


@dataclass
class ModelConfig:
    model_config_name: str
    bert_model_name: str
    bert_model_output_dim: int
    num_epochs: int
    save_models_dir: str
    model_name: str
    optimizer: str
    learning_rate: float


class ExperimentConfig(NamedTuple):
    dataset_config: DatasetConfig
    model_config: ModelConfig


def get_experiment_config(model_config_name: str, dataset_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        get_dataset_config_by_name(dataset_name),
        get_model_config_by_name(model_config_name)
    )


def read_dataset_config(config_file_path: str) -> DatasetConfig:
    with open(config_file_path, 'r') as yaml_file:
        dataset_config_raw = yaml.safe_load(yaml_file)
        dataset_config = DatasetConfig(
            train_samples_file_path=dataset_config_raw['train_samples_file_path'],
            valid_samples_file_path=dataset_config_raw['valid_samples_file_path'],
            test_samples_file_path=dataset_config_raw['test_samples_file_path'],
            types_file_path=dataset_config_raw['types_file_path'],
            num_types=int(dataset_config_raw['num_types']),
            dataset_name=dataset_config_raw['dataset_name'],
        )
        assert isinstance(dataset_config.num_types, int)
        return dataset_config


def read_model_config(model_config_file_path: str) -> ModelConfig:
    with open(model_config_file_path, 'r') as yaml_file:
        model_config_raw = yaml.safe_load(yaml_file)
        assert len(model_config_raw) == 8, "model config should only have 7 attributes currently"
        model_config = ModelConfig(
            model_config_name=model_config_raw['model_config_name'],
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


def get_model_config_by_name(model_config_name: str) -> ModelConfig:
    all_config_file_paths = glob.glob('configs/model_configs/*.yaml')
    for config_file_path in all_config_file_paths:
        model_config = read_model_config(config_file_path)
        if model_config.model_config_name == model_config_name:
            return model_config
    die(f"Should have been able to find model config with name {model_config_name}")


def get_dataset_config_by_name(dataset_name: str) -> DatasetConfig:
    all_config_file_paths = glob.glob('configs/dataset_configs/*.yaml')
    for config_file_path in all_config_file_paths:
        dataset_config = read_dataset_config(config_file_path)
        if dataset_config.dataset_name == dataset_name:
            return dataset_config
    die(f"Should have been able to find model config with name {dataset_name}")
