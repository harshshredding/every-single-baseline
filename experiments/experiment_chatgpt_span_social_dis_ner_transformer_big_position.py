from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_position',
        dataset_config_name='social_dis_ner_chatgpt'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20
