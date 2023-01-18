from utils.report import get_experiment_results_dict
from utils.report import Experiment


def test_get_experiment_results_dict():
    experiment_results = get_experiment_results_dict('./util_tests/test_performance_seq_label_semeval.csv')
    assert len(experiment_results) == 4

    result = experiment_results[
        Experiment(experiment_name='seq_label_semeval', dataset_name='legaleval_judgement')
    ]
    assert str(result).startswith("0.856")

    result = experiment_results[
        Experiment(experiment_name='seq_label_semeval', dataset_name='legaleval_preamble')
    ]
    assert str(result).startswith("0.791")

    result = experiment_results[
        Experiment(experiment_name='seq_label_semeval', dataset_name='multiconer_coarse')
    ]
    assert str(result).startswith("0.758")

    result = experiment_results[
        Experiment(experiment_name='seq_label_semeval', dataset_name='multiconer_fine')
    ]
    assert str(result).startswith("0.456")

