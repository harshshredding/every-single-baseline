from utils.config import BiggerBatchModifier, EvenBiggerBatchModifier, get_experiment_config, Epochs30Modifier, AccuracyEvaluationModifier, TestEveryEpochModifier, AdafactorModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted_bio',
        dataset_config_name='ncbi_sentence_all_mistakes_all_gold',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdamModifier(),
            AccuracyEvaluationModifier(),
            BiggerBatchModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_meta_special_weighted_bio',
        dataset_config_name='ncbi_sentence_all_mistakes_all_gold',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdamModifier(),
            AccuracyEvaluationModifier(),
            EvenBiggerBatchModifier(),
        ]
    ),
]
