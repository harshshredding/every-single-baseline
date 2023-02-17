from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='SeqLabelBatchedNoTokenizationBase',
        dataset_config_name='social_dis_ner_gpt'
    )
]
