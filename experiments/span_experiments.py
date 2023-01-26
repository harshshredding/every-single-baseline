from utils.config import get_experiment_config

experiments = [
    get_experiment_config('SpanBert', 'legaleval_judgement'),
    get_experiment_config('SpanBert', 'legaleval_preamble'),
    get_experiment_config('SpanBert', 'multiconer_coarse'),
    get_experiment_config('SpanBert', 'multiconer_fine')
]
