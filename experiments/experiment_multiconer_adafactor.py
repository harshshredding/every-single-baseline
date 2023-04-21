from utils.config import AdafactorModifier, AdamModifier, TinySpanWidthModifier, get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier, SmallerSpanWidthModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='multiconer_fine_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='multiconer_fine_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='multiconer_fine_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            AdamModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_default',
        dataset_config_name='multiconer_fine_vanilla',
        modifiers=[
            Epochs20Modifier(),
            SmallerSpanWidthModifier(),
            AdamModifier()
        ]
    ),
]
