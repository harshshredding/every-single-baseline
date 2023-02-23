from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_base_default',
        dataset_config_name='chem_drug_ner_window_stride_longer'
    ),
]
