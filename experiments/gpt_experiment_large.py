from utils.config import get_experiment_config

experiments = [
    get_experiment_config(model_config_name='SeqBatchedRobertaLarge', dataset_config_name='social_dis_ner_gpt'),
    get_experiment_config(model_config_name='SeqBatchedRobertaLarge', dataset_config_name='social_dis_ner'),
]
