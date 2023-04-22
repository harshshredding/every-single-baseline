from utils.config import AdafactorModifier, BiggerBatchModifier, EpochsCustomModifier, EvenBiggerBatchModifier, get_experiment_config, Epochs30Modifier, AccuracyEvaluationModifier, TestEveryEpochModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted',
        dataset_config_name='social_dis_ner_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestEveryEpochModifier(),
            AdafactorModifier(),
            AccuracyEvaluationModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted',
        dataset_config_name='social_dis_ner_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestEveryEpochModifier(),
            AdafactorModifier(),
            AccuracyEvaluationModifier(),
            BiggerBatchModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted',
        dataset_config_name='social_dis_ner_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestEveryEpochModifier(),
            AdamModifier(),
            AccuracyEvaluationModifier(),
            BiggerBatchModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted',
        dataset_config_name='social_dis_ner_meta',
        modifiers=[
            EpochsCustomModifier(num_epochs=15),
            TestEveryEpochModifier(),
            AdamModifier(),
            AccuracyEvaluationModifier(),
            EvenBiggerBatchModifier(),
        ]
    ),
]
