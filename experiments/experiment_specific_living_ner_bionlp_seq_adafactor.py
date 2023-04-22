from utils.config import AdafactorModifier, AdamModifier, EpochsCustomModifier, SmallerSpanWidthModifier, TinySpanWidthModifier, get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier


experiments = [
    # Seq
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='living_ner_window',
        modifiers=[
            EpochsCustomModifier(num_epochs=12),
            SmallerSpanWidthModifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
]
