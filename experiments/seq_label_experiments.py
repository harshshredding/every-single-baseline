from utils.config import get_experiment_config

experiments = [
    get_experiment_config('JustBert3Classes', 'legaleval_judgement'),
    get_experiment_config('JustBert3Classes', 'legaleval_preamble'),
    get_experiment_config('JustBert3Classes', 'multiconer_coarse'),
    get_experiment_config('JustBert3Classes', 'multiconer_fine')
]
