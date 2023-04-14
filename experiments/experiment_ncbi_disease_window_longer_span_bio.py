from utils.config import get_experiment_config


experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_bio_default',
        dataset_config_name='ncbi_disease_window_longer'
    ), 
    get_experiment_config(
        model_config_module_name='model_span_large_bio_default',
        dataset_config_name='ncbi_disease_window_longer'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20
    experiment.testing_frequency = 1
    experiment.model_config.max_span_length = 32


experiments[1].model_config.batch_size = 8
