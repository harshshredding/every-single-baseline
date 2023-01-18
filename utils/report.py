from typing import NamedTuple, List
import matplotlib.pyplot as plt
import csv


class Experiment(NamedTuple):
    experiment_name: str
    dataset_name: str


class ExperimentResult(NamedTuple):
    experiment: Experiment
    max_performance: float


def get_experiment_results_dict(performance_csv_file_path):
    experiment_result_dict = {}
    with open(performance_csv_file_path, 'r') as perf_file:
        reader = csv.DictReader(perf_file)
        for row in reader:
            experiment = Experiment(row['experiment_name'], row['dataset_name'])
            curr_score = float(row['f1_score'])
            if experiment not in experiment_result_dict:
                experiment_result_dict[experiment] = curr_score
            else:
                max_perf = experiment_result_dict[experiment]
                if curr_score > max_perf:
                    experiment_result_dict[experiment] = curr_score
    return experiment_result_dict


def generate_table_pdf(data: List[List[str]], column_names: List[str], pdf_path: str):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.table(cellText=data,
             colLabels=column_names,
             loc='center')
    fig.tight_layout()
    plt.savefig(pdf_path, bbox_inches='tight')
