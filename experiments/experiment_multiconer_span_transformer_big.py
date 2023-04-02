from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big',
        dataset_config_name='multiconer_fine_vanilla'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20
