from utils.config import read_model_config, read_dataset_config


def test_read_model_config():
    model_config = read_model_config('util_tests/test_configs/test_span_rep_model_config.yaml')
    assert model_config.model_name == 'SpanBert'
    assert model_config.num_epochs == 10
    assert model_config.save_models_dir == './models'
    assert model_config.learning_rate == 1e-5


def test_read_dataset_config():
    dataset_config = read_dataset_config('util_tests/test_configs/test_multiconer_fine_dataset_config.yaml')
    assert dataset_config.dataset_name == 'multiconer_fine'
    assert dataset_config.num_types == 6
    assert dataset_config.train_annos_file_path == './preprocessed_data/multiconer_train_fine_annos.tsv'
    assert dataset_config.valid_tokens_file_path == './preprocessed_data/multiconer_valid_fine_tokens.json'
