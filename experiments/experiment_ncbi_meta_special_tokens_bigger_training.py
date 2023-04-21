from utils.config import get_experiment_config, Epochs30Modifier, AccuracyEvaluationModifier, TestEveryEpochModifier, AdafactorModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_meta_special_tokens_bio',
        dataset_config_name='ncbi_sentence_meta_bigger_special_tokens',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier(),
            AccuracyEvaluationModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_meta_special_tokens_bio',
        dataset_config_name='ncbi_sentence_meta_bigger_special_tokens',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdamModifier(),
            AccuracyEvaluationModifier()
        ]
    ),
]
