from utils.config import get_experiment_config

experiments = [
    get_experiment_config(model_config_name='BatchedBert', dataset_name='genia'),
]
