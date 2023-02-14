from utils.config import get_experiment_config

experiments = [
    # Living NER
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='living_ner_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='living_ner_vanilla'
    ),

    # Multiconer
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='multiconer_fine_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='multiconer_fine_vanilla'
    ),

    # Genia
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='genia_config_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='genia_config_vanilla'
    ),

    # SocialDisNER
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='social_dis_ner_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='social_dis_ner_vanilla'
    ),

    # SocialDisNER GPT
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='social_dis_ner_gpt'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='social_dis_ner_gpt'
    ),

    # LegalEval Preamble
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='legaleval_preamble_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='legaleval_preamble_vanilla'
    ),

    # LegalEval Judgement
    get_experiment_config(
        model_config_name='SeqLabelBatchedNoTokenizationLarge',
        dataset_config_name='legaleval_judgement_vanilla'
    ),

    get_experiment_config(
        model_config_name='SpanBatchedNoTokenizationLargeSpanish',
        dataset_config_name='legaleval_judgement_vanilla'
    ),

]