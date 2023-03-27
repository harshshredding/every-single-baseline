from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='multiconer_fine_vanilla'
    ),
]
