from utils.config import AdafactorModifier, AdamModifier, EpochsCustomModifier, SmallerSpanWidthModifier, TinySpanWidthModifier, get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier


experiments = [
    # CRF
    get_experiment_config(
        model_config_module_name='model_seq_large_crf_bio',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_crf_bio',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdamModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_crf_bio',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            BiggerBatchModifier(),
            AdamModifier()
        ]
    ),
]
