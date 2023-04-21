from utils.config import AdafactorModifier, AdamModifier, EpochsCustomModifier, SmallerSpanWidthModifier, TinySpanWidthModifier, get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier


experiments = [
    # CRF
    get_experiment_config(
        model_config_module_name='model_seq_large_crf_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    # Span
    get_experiment_config(
        model_config_module_name='model_span_large_bio_default',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    # Seq
    get_experiment_config(
        model_config_module_name='model_seq_large_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
]
