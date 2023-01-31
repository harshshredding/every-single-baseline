from utils.config import get_experiment_config

experiments = [
    get_experiment_config('SeqLabelBase', 'legaleval_judgement'),
    get_experiment_config('SeqLabelBase', 'legaleval_preamble'),
    get_experiment_config('SeqLabelBase', 'multiconer_coarse'),
    get_experiment_config('SeqLabelBase', 'multiconer_fine')
]
