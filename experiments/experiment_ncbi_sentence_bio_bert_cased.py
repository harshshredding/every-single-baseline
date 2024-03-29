from utils.config import get_experiment_config, Epochs20Modifier, BiggerBatchModifier, TestEveryEpochModifier, SmallerSpanWidthModifier


experiments = [
    get_experiment_config(
        model_config_module_name='model_seq_large_bio_bert_cased',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_seq_large_bio_bert_cased',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            BiggerBatchModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_bio_bert_cased',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier()
        ]
    ),
    get_experiment_config(
        model_config_module_name='model_span_large_bio_bert_cased',
        dataset_config_name='ncbi_disease_sentence',
        modifiers=[
            Epochs20Modifier(),
            TestEveryEpochModifier(),
            SmallerSpanWidthModifier(),
            BiggerBatchModifier()
        ]
    ),
]

for experiment_config in experiments:
    experiment_config.optimizer = 'Adafactor'

