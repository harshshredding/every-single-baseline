from utils.config import get_experiment_config

# In the hope that reducing max-span-width will
# stabilize training.

experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='living_ner_window_combo'
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_crf',
        dataset_config_name='living_ner_window_combo'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20
    experiment.testing_frequency = 1
    experiment.model_config.batch_size = 8
