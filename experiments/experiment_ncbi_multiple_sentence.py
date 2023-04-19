from utils.config import BiggerBatchModifier, get_experiment_config, Epochs20Modifier, AccuracyEvaluationModifier, TestEveryEpochModifier, AdafactorModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_large_bio',
        dataset_config_name='ncbi_disease_multiple_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_bio_default',
        dataset_config_name='ncbi_disease_multiple_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_bio',
        dataset_config_name='ncbi_disease_multiple_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdamModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_bio_default',
        dataset_config_name='ncbi_disease_multiple_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            AdamModifier(),
        ]
    ),
]
