from utils.config import AdafactorModifier, BiggerBatchModifier, EpochsCustomModifier, EvenBiggerBatchModifier, TestFrequencyModifier, get_experiment_config, Epochs30Modifier, AccuracyEvaluationModifier, TestEveryEpochModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted_bio',
        dataset_config_name='genia_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestFrequencyModifier(frequency=3),
            AdafactorModifier(),
            AccuracyEvaluationModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted_bio',
        dataset_config_name='genia_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestFrequencyModifier(frequency=3),
            AdafactorModifier(),
            AccuracyEvaluationModifier(),
            BiggerBatchModifier(),
        ]
    ),
]
