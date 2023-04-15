from utils.config import get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier, SmallerSpanWidthModifier

experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_external_pos_bio',
        dataset_config_name='ncbi_umls_lower_word_boundary',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_external_pos_bio',
        dataset_config_name='ncbi_umls_lower_word_boundary',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            BiggerBatchModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_external_pos_bio',
        dataset_config_name='ncbi_umls_lower_word_boundary',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_external_pos_bio',
        dataset_config_name='ncbi_umls_lower_word_boundary',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            BiggerBatchModifier()
        ]
    ),
]


for experiment_config in experiments:
    experiment_config.model_config.external_feature_type = 'UmlsExactLoweredWordBoundary'
    experiment_config.optimizer = 'Adafactor'
