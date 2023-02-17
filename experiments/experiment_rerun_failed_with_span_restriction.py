from utils.config import get_experiment_config_with_smaller_batch

experiments = [
    # Living NER
    get_experiment_config_with_smaller_batch(
        model_config_module='model_span_large_span_width_restriction',
        dataset_config_name='living_ner_vanilla'
    ),

    # LegalEval Preamble
    get_experiment_config_with_smaller_batch(
        model_config_module='model_span_large_span_width_restriction',
        dataset_config_name='legaleval_preamble_vanilla'
    ),

]
