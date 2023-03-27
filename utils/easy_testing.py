from transformers import AutoModel
from transformers import AutoTokenizer
from typing import List
from utils.config import get_dataset_config_by_name, ModelConfig
import train_util
from structs import Sample


def get_bert_tokenizer(model_name: str = 'bert-base-cased'):
    """
    Get the bert tokenizer
    """
    return AutoTokenizer.from_pretrained(model_name)

def get_roberta_tokenizer():
    return AutoTokenizer.from_pretrained('xlm-roberta-base')

def get_bert_model():
    return AutoModel.from_pretrained('bert-base-cased')


def get_bert_encoding(bert_tokenizer, tokens=List[str]):
    return bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                          add_special_tokens=False, truncation=True, max_length=512)


def get_train_samples_by_dataset_name(dataset_config_name: str) -> List[Sample]:
    return train_util.get_train_samples(get_dataset_config_by_name(dataset_config_name))


def get_valid_samples_by_dataset_name(dataset_config_name: str) -> List[Sample]:
    return train_util.get_valid_samples(get_dataset_config_by_name(dataset_config_name))


def get_test_samples_by_dataset_name(dataset_config_name: str) -> List[Sample]:
    return train_util.get_test_samples(get_dataset_config_by_name(dataset_config_name))
