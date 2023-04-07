from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_external_pos',
        dataset_config_name='social_dis_ner_per_sample_chatgpt'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20
    experiment.model_config.external_feature_type = 'ChatGptDiseasePerSample'
