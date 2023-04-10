from utils.config import get_experiment_config


experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='genia_config_vanilla'
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='genia_config_vanilla'
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='genia_config_vanilla'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 15
    experiment.testing_frequency = 1

experiments[1].model_config.max_span_length = 32

experiments[2].model_config.max_span_length = 32
experiments[2].model_config.batch_size = 8
