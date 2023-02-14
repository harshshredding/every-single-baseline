from utils.config import get_experiment_config

experiments = [
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='multiconer_fine_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanRepSpanBertLarge',
        dataset_config_name='multiconer_fine_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='multiconer_fine_vanilla'
    ),
]
