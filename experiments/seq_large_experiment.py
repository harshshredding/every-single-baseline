from utils.config import get_experiment_config

experiments = [
    get_experiment_config('SeqLarge', 'legaleval_judgement'),
    get_experiment_config('SeqLarge', 'legaleval_preamble'),
    get_experiment_config('SeqLarge', 'multiconer_fine')
]