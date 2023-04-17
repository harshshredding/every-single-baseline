from utils.config import get_experiment_config, Epochs30Modifier, BiggerBatchModifier, TestEveryEpochModifier, AdafactorModifier, AdamModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_large_external_pos_bio',
        dataset_config_name='ncbi_disease_sentence_umls_with_gold',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_external_pos_bio',
        dataset_config_name='ncbi_disease_sentence_umls_with_gold',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            BiggerBatchModifier(),
            AdafactorModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_external_pos_bio',
        dataset_config_name='ncbi_disease_sentence_umls_with_gold',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            AdamModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_external_pos_bio',
        dataset_config_name='ncbi_disease_sentence_umls_with_gold',
        modifiers=[
            Epochs30Modifier(),
            TestEveryEpochModifier(),
            BiggerBatchModifier(),
            AdamModifier()
        ]
    ),
]

for experiment_config in experiments:
    experiment_config.model_config.external_feature_type = 'UmlsDiseaseGold'

