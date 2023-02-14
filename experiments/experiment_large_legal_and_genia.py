from utils.config import get_experiment_config

experiments = [
    get_experiment_config('SpanLarge', 'genia'),
    get_experiment_config('SpanLarge', 'legaleval_judgement'),
    get_experiment_config('SpanLarge', 'legaleval_preamble'),
    get_experiment_config('SeqLarge', 'genia'),
    get_experiment_config('SeqLarge', 'legaleval_judgement'),
    get_experiment_config('SeqLarge', 'legaleval_preamble'),
]
