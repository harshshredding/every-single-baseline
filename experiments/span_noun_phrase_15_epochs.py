from utils.config import get_experiment_config

experiments = [
    get_experiment_config('SpanBertNounPhrase', 'multiconer_coarse'),
    get_experiment_config('SpanBertNounPhrase', 'multiconer_fine'),
    get_experiment_config('SpanBertNounPhrase', 'legaleval_judgement'),
    get_experiment_config('SpanBertNounPhrase', 'legaleval_preamble'),
]

for experiment in experiments:
    experiment.model_config.num_epochs = 15