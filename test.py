from utils.easy_testing import get_bert_tokenizer
import train_util
from utils.config import get_dataset_config_by_name, DatasetConfig
import util
from structs import Sample
from preamble import *

genia_config = get_dataset_config_by_name('genia')
genia_samples = train_util.get_train_samples(genia_config)

legaleval_preamble_config = get_dataset_config_by_name('legaleval_preamble')
legaleval_preamble_samples = train_util.get_train_samples(legaleval_preamble_config)

legaleval_judgement_config = get_dataset_config_by_name('legaleval_judgement')
legaleval_judgement_samples = train_util.get_train_samples(legaleval_judgement_config)


def ensure_no_sample_is_being_truncated(samples: List[Sample], dataset_config: DatasetConfig):
    for sample in samples:
        tokens = util.get_tokens_from_sample(sample)
        batch_encoding = bert_tokenizer(
            tokens, is_split_into_words=True, truncation=True, return_tensors='pt', add_special_tokens=False
        )
        if batch_encoding['input_ids'].shape[1] == bert_tokenizer.model_max_length:
            print(f"WARN: In dataset {dataset_config.dataset_name}, the sample {sample.id} is being truncated")


bert_tokenizer = get_bert_tokenizer()

for samples, dataset_config in [
    (genia_samples, genia_config),
    (legaleval_preamble_samples, legaleval_preamble_config),
    (legaleval_judgement_samples, legaleval_judgement_config)
]:
    ensure_no_sample_is_being_truncated(samples, dataset_config)
