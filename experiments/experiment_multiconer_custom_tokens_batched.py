from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_large_custom_tokens',
        dataset_config_name='multiconer_fine_tokens'
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_custom_tokenization',
        dataset_config_name='multiconer_fine_tokens'
    ),
    get_experiment_config(
        model_config_module_name='model_seq_base_custom_tokens',
        dataset_config_name='multiconer_fine_tokens'
    ),
    get_experiment_config(
        model_config_module_name='model_span_base_custom_tokenization',
        dataset_config_name='multiconer_fine_tokens'
    ),
]
