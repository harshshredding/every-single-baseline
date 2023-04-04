from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='living_ner_window'
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='living_ner_window'
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_crf',
        dataset_config_name='living_ner_window'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20

