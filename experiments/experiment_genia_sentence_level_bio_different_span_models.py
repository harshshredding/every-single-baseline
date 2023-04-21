from utils.config import AdafactorModifier, AdamModifier, get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier, SmallerSpanWidthModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            BiggerBatchModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_position_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_position_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            BiggerBatchModifier(),
            AdafactorModifier()
        ]
    ),

    # ------------- ADAM ---------------
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            AdamModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            BiggerBatchModifier(),
            AdamModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_position_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            AdamModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_transformer_big_position_bio',
        dataset_config_name='genia_config_vanilla',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            BiggerBatchModifier(),
            AdamModifier()
        ]
    ),
]


assert len(experiments) == 8
assert len([experiment for experiment in experiments if experiment.optimizer == 'Adam']) == 4
