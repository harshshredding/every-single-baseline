from typing import NamedTuple, Optional
from structs import EvaluationType
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
    expected_number_of_train_samples: int
    expected_number_of_valid_samples: int
    expected_number_of_test_samples: int

@dataclass
class ModelConfig:
    model_config_name: str
    pretrained_model_name: str
    pretrained_model_output_dim: int
    model_name: str
    learning_rate: float
    batch_size: int

    # Span Rep Model specific
    max_span_length: Optional[int] = None

    # Tokenization specific options
    use_special_bert_tokens: Optional[bool] = True

    # External Anno type 
    external_feature_type: Optional[str] = None

@dataclass
class PreprocessorConfig:
    preprocessor_config_name: str
    preprocessor_class_path: str
    preprocessor_class_init_params: dict

@dataclass
class ExperimentConfig:
    dataset_config: DatasetConfig
    model_config: ModelConfig
    testing_frequency: int
    optimizer: str = 'Adam'
    num_epochs: int = 20
    evaluation_type: EvaluationType = EvaluationType.f1

def get_large_model_config(
        model_config_name: str,
        model_name: str,
) -> ModelConfig:
    return ModelConfig(
        model_config_name=model_config_name,
        model_name=model_name,
        pretrained_model_name='xlm-roberta-large',
        pretrained_model_output_dim=1024,
        batch_size=4,
        learning_rate=1e-5
    )


def get_large_model_config_bio(
        model_config_name: str,
        model_name: str,
) -> ModelConfig:
    return ModelConfig(
        model_config_name=model_config_name,
        model_name=model_name,
        pretrained_model_name='michiyasunaga/BioLinkBERT-large',
        pretrained_model_output_dim=1024,
        batch_size=4,
        learning_rate=1e-5
    )


def get_large_model_config_bio_bert_cased(
        model_config_name: str,
        model_name: str,
) -> ModelConfig:
    return ModelConfig(
        model_config_name=model_config_name,
        model_name=model_name,
        pretrained_model_name='dmis-lab/biobert-large-cased-v1.1',
        pretrained_model_output_dim=1024,
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
        batch_size=4,
        learning_rate=1e-5
    )


class ExperimentModifier:
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        # modify the experiment config and return in
        raise NotImplementedError("Need to implement the modify method")

class BiggerBatchModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.batch_size = 8
        return experiment_config


class EvenBiggerBatchModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.batch_size = 16
        return experiment_config

class SmallerBatchModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.batch_size = 1
        return experiment_config

class SmallerSpanWidthModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.max_span_length = 32
        return experiment_config


class TinySpanWidthModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.max_span_length = 16
        return experiment_config


class TestEveryEpochModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.testing_frequency = 1
        return experiment_config


class TestFrequencyModifier(ExperimentModifier):
    def __init__(self, frequency: int):
        super().__init__()
        self.frequency = frequency

    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.testing_frequency = self.frequency
        return experiment_config


class Epochs20Modifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.num_epochs = 20
        return experiment_config

class EpochsCustomModifier(ExperimentModifier):
    def __init__(self, num_epochs: int):
        super().__init__()
        self.num_epochs = num_epochs

    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.num_epochs = self.num_epochs
        return experiment_config

class Epochs30Modifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.num_epochs = 30
        return experiment_config

class AccuracyEvaluationModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.evaluation_type = EvaluationType.accuracy
        return experiment_config

class AdamModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:  
        experiment_config.optimizer = 'Adam'
        return experiment_config

class AdafactorModifier(ExperimentModifier):
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:  
        experiment_config.optimizer = 'Adafactor'
        return experiment_config

def get_experiment_config(
        model_config_module_name: str,
        dataset_config_name: str,
        modifiers: list[ExperimentModifier] = []
    ) -> ExperimentConfig:
    experiment_config = ExperimentConfig(
        get_dataset_config_by_name(dataset_config_name),
        get_model_config_from_module(model_config_module_name),
        testing_frequency=4
    )

    if len(modifiers):
        for modifier in modifiers:
            experiment_config = modifier.modify(experiment_config)

    return experiment_config


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
            dataset_config_name=dataset_config_raw['dataset_config_name'],
            expected_number_of_train_samples=dataset_config_raw['expected_number_of_train_samples'],
            expected_number_of_valid_samples=dataset_config_raw['expected_number_of_valid_samples'],
            expected_number_of_test_samples=dataset_config_raw['expected_number_of_test_samples']
        )
        assert isinstance(dataset_config.num_types, int)
        assert ('test' in dataset_config.test_samples_file_path) \
                or ('change_this' == dataset_config.test_samples_file_path)
        assert 'train' in dataset_config.train_samples_file_path
        assert 'valid' in dataset_config.valid_samples_file_path
        return dataset_config


def read_model_config(model_config_file_path: str) -> ModelConfig:
    with open(model_config_file_path, 'r') as yaml_file:
        model_config_raw = yaml.safe_load(yaml_file)
        assert len(model_config_raw) == 9, "model config should only have 7 attributes currently"
        model_config = ModelConfig(
            model_config_name=model_config_raw['model_config_name'],
            pretrained_model_name=model_config_raw['bert_model_name'],
            pretrained_model_output_dim=int(model_config_raw['bert_model_output_dim']),
            model_name=model_config_raw['model_name'],
            learning_rate=float(model_config_raw['learning_rate']),
            batch_size=int(model_config_raw['batch_size'])
        )
        assert isinstance(model_config.pretrained_model_output_dim, int)
        assert isinstance(model_config.learning_rate, float)
        return model_config


def get_model_config_from_module(model_config_module_name: str) -> ModelConfig:
    """
    param:
        model_config_module_path(str): the path to the module in which the model config is defined
    """
    model_config_module = importlib.import_module(f'configs.model_configs.{model_config_module_name}')
    return model_config_module.create_model_config(model_config_module_name)


def get_preprocessor_config(preprocessor_config_module_name: str) -> PreprocessorConfig:
    preprocessor_config_module = importlib.import_module(f'configs.preprocessor_configs.{preprocessor_config_module_name}')
    return preprocessor_config_module.get_preprocessor_config(preprocessor_config_module_name)


def get_dataset_config_by_name(dataset_config_name: str) -> DatasetConfig:
    all_config_file_paths = glob.glob('configs/dataset_configs/*.yaml')
    found_dataset_config = None
    for config_file_path in all_config_file_paths:
        dataset_config = read_dataset_config(config_file_path)
        if dataset_config.dataset_config_name == dataset_config_name:
            assert found_dataset_config is None, f"Duplicate dataset config {dataset_config}"
            found_dataset_config = dataset_config
    assert found_dataset_config is not None, f"Should have been able to find dataset config with name {dataset_config_name}"
    assert 'production' in found_dataset_config.train_samples_file_path
    assert 'production' in found_dataset_config.test_samples_file_path
    assert 'production' in found_dataset_config.valid_samples_file_path
    assert 'json' in found_dataset_config.train_samples_file_path
    assert 'json' in found_dataset_config.test_samples_file_path
    assert 'json' in found_dataset_config.valid_samples_file_path
    return found_dataset_config


def get_experiment_config_with_smaller_batch(model_config_module: str, dataset_config_name: str):
    experiment_config = get_experiment_config(
        model_config_module_name=model_config_module,
        dataset_config_name=dataset_config_name
    )
    experiment_config.model_config.batch_size = 2
    return experiment_config
