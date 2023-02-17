from typing import NamedTuple, Optional
from utils.universal import die
import yaml
import glob
from dataclasses import dataclass
import importlib


@dataclass
class DatasetConfig:
    train_samples_file_path: str
    valid_samples_file_path: str
    test_samples_file_path: str
    types_file_path: str
    num_types: int
    dataset_name: str
    dataset_config_name: str


@dataclass
class ModelConfig:
    model_config_name: str
    pretrained_model_name: str
    pretrained_model_output_dim: int
    num_epochs: int
    model_name: str
    optimizer: str
    learning_rate: float
    batch_size: int
    # Span Rep Model specific
    max_span_length: Optional[int] = None


def get_large_model_config(
        model_config_name: str,
        model_name: str,
) -> ModelConfig:
    return ModelConfig(
        model_config_name=model_config_name,
        model_name=model_name,
        pretrained_model_name='xlm-roberta-large',
        pretrained_model_output_dim=1024,
        num_epochs=15,
        optimizer='Adam',
        batch_size=4,
        learning_rate=1e-5
    )


def get_small_model_config(
        model_config_name: str,
        model_name: str,
) -> ModelConfig:
    return ModelConfig(
        model_config_name=model_config_name,
        model_name=model_name,
        pretrained_model_name='xlm-roberta-base',
        pretrained_model_output_dim=768,
        num_epochs=15,
        optimizer='Adam',
        batch_size=4,
        learning_rate=1e-5
    )


class ExperimentConfig(NamedTuple):
    dataset_config: DatasetConfig
    model_config: ModelConfig


def get_experiment_config(model_config_module_name: str, dataset_config_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        get_dataset_config_by_name(dataset_config_name),
        get_model_config_from_module(model_config_module_name)
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
            dataset_config_name=dataset_config_raw['dataset_config_name']
        )
        assert isinstance(dataset_config.num_types, int)
        return dataset_config


def read_model_config(model_config_file_path: str) -> ModelConfig:
    with open(model_config_file_path, 'r') as yaml_file:
        model_config_raw = yaml.safe_load(yaml_file)
        assert len(model_config_raw) == 9, "model config should only have 7 attributes currently"
        model_config = ModelConfig(
            model_config_name=model_config_raw['model_config_name'],
            pretrained_model_name=model_config_raw['bert_model_name'],
            pretrained_model_output_dim=int(model_config_raw['bert_model_output_dim']),
            num_epochs=int(model_config_raw['num_epochs']),
            model_name=model_config_raw['model_name'],
            optimizer=model_config_raw['optimizer'],
            learning_rate=float(model_config_raw['learning_rate']),
            batch_size=int(model_config_raw['batch_size'])
        )
        assert isinstance(model_config.pretrained_model_output_dim, int)
        assert isinstance(model_config.num_epochs, int)
        assert isinstance(model_config.learning_rate, float)
        return model_config


def get_model_config_from_module(model_config_module_path: str) -> ModelConfig:
    """
    param:
        model_config_module_path(str): the path to the module in which the model config is defined
    """
    model_config_module = importlib.import_module(f'configs.model_configs.{model_config_module_path}')
    return model_config_module.model_config


def get_dataset_config_by_name(dataset_config_name: str) -> DatasetConfig:
    all_config_file_paths = glob.glob('configs/dataset_configs/*.yaml')
    for config_file_path in all_config_file_paths:
        dataset_config = read_dataset_config(config_file_path)
        if dataset_config.dataset_config_name == dataset_config_name:
            return dataset_config
    die(f"Should have been able to find dataset config with name {dataset_config_name}")


def get_experiment_config_with_smaller_batch(model_config_module: str, dataset_config_name: str):
    experiment_config = get_experiment_config(
        model_config_module_name=model_config_module,
        dataset_config_name=dataset_config_name
    )
    experiment_config.model_config.batch_size = 2
    return experiment_config
