from utils.config import get_experiment_config

experiments = [
    get_experiment_config('JustBert3ClassesCRF', 'legaleval_judgement'),
    get_experiment_config('JustBert3ClassesCRF', 'legaleval_preamble'),
    get_experiment_config('JustBert3ClassesCRF', 'multiconer_coarse'),
    get_experiment_config('JustBert3ClassesCRF', 'multiconer_fine')
]
