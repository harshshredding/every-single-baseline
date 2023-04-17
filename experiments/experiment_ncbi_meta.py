from utils.config import get_experiment_config, Epochs30Modifier, BiggerBatchModifier, TestEveryEpochModifier, AdafactorModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_meta_bio',
        dataset_config_name='ncbi_sentence_meta',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
]
