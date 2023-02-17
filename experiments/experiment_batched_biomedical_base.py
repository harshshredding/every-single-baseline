from utils.config import get_experiment_config

experiments = [
    get_experiment_config(model_config_module_name='BatchedBertSpanish', dataset_name='social_dis_ner'),
    get_experiment_config(model_config_module_name='BatchedBertSpanish', dataset_name='living_ner'),
    get_experiment_config(model_config_module_name='BatchedBert', dataset_name='genia'),
]
