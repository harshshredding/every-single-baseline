from typing import NamedTuple, List
import matplotlib.pyplot as plt
import csv


class Experiment(NamedTuple):
    experiment_name: str
    model_name: str
    dataset_name: str


class ExperimentResult(NamedTuple):
    experiment: Experiment
    score: float


def get_experiment_results(performance_csv_file_path) -> List[ExperimentResult]:
    performance_dict: dict[Experiment, float] = {}
    with open(performance_csv_file_path, 'r') as perf_file:
        reader = csv.DictReader(perf_file)
        row: dict
        for row in reader:
            experiment_name = row.get('experiment_name', 'n/a')
            model_name = row.get('model_name', 'n/a')
            dataset_name = row.get('dataset_name', 'n/a')
            curr_score = float(row['f1_score'])
            experiment = Experiment(
                experiment_name=experiment_name,
                model_name=model_name,
                dataset_name=dataset_name
            )
            if (experiment not in performance_dict) or (performance_dict[experiment] < curr_score):
                performance_dict[experiment] = curr_score

    assert len(performance_dict)
    ret = []
    for experiment in performance_dict:
        ret.append(ExperimentResult(experiment, performance_dict[experiment]))
    return ret


def generate_table_pdf(data: List[List[str]], column_names: List[str], pdf_path: str):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.table(
        cellText=data,
        colLabels=column_names,
    )
    fig.tight_layout()
    plt.savefig(pdf_path, bbox_inches='tight')
