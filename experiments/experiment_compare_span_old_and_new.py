from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_no_special_tokens',
        dataset_config_name='multiconer_fine_vanilla'
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_custom_tokenization_no_batch',
        dataset_config_name='multiconer_fine_tokens'
    ),
]
