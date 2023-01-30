from models import SpanBertNounPhrase
from utils.config import get_model_config_by_name, get_dataset_config_by_name
import util

def test_SpanBertNounPhrase():
    dataset_config = get_dataset_config_by_name('multiconer_coarse')
    model_config = get_model_config_by_name("SpanBertNounPhrase")
    all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
    model = SpanBertNounPhrase(all_types, model_config)
    all_samples = util.read_samples(dataset_config.valid_samples_file_path)
    one_sample = all_samples[0]
    model(one_sample)