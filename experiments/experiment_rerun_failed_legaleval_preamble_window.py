from utils.config import get_experiment_config_with_smaller_batch

experiments = [
    # LegalEval Preamble
    get_experiment_config_with_smaller_batch(
        model_config_module='model_span_base_default',
        dataset_config_name='legaleval_preamble_window'
    ),
]
