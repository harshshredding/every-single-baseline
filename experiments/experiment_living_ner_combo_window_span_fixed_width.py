from utils.config import get_experiment_config

# In the hope that reducing max-span-width will
# stabilize training.

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='living_ner_window_combo'
    ),
]

experiments[0].model_config.num_epochs = 20
experiments[0].model_config.max_span_length = 32
