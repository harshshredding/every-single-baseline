from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_crf_external_pos',
        dataset_config_name='social_dis_ner_umls_external_word_boundaries'
    ),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 20
    experiment.model_config.external_feature_type = 'UmlsExactLoweredWordBoundary'
