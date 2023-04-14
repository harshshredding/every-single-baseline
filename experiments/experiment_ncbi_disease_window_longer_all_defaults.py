from utils.config import get_experiment_config


experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='ncbi_disease_window_longer'
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='ncbi_disease_window_longer'
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='ncbi_disease_window_longer'
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_crf',
        dataset_config_name='ncbi_disease_window_longer'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 15
    experiment.testing_frequency = 1

experiments[0].model_config.max_span_length = 32
experiments[0].model_config.batch_size = 8
