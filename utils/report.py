from typing import NamedTuple, List
import util
from utils import dropbox as dropbox_util
import matplotlib.pyplot as plt
import csv
import glob


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
            curr_score = row['f1_score'].split(',')
            if len(curr_score) == 1:
                curr_score = float(curr_score[0])
            else:
                curr_score = float(curr_score[0][1:])
            experiment = Experiment(
                experiment_name=experiment_name,
                model_name=model_name,
                dataset_name=dataset_name
            )
            if (experiment not in performance_dict) or (performance_dict[experiment] < curr_score):
                performance_dict[experiment] = curr_score

    if not len(performance_dict):
        print(f"WARN: file {performance_csv_file_path} does not have any results")

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


def list_all_performance_files() -> List[str]:
    all_file_names = dropbox_util.get_all_file_names_in_folder()
    all_performance_files = [file_name for file_name in all_file_names if ("performance" in file_name)]
    return all_performance_files


def download_all_performance_files():
    util.create_directory_structure('./performance_files')
    performance_files_to_download = list_all_performance_files()
    for performance_file in performance_files_to_download:
        dropbox_path = f'/{performance_file}'
        local_download_path = f'./performance_files/{performance_file}'
        dropbox_util.download_file(dropbox_path, local_download_path)


def generate_performance_analysis_file():
    all_downloaded_performance_files = glob.glob('./performance_files/*.csv')
    with open('./performance_analysis.csv', 'w') as performance_analysis_file:
        writer = csv.writer(performance_analysis_file, delimiter='\t')
        for performance_file in all_downloaded_performance_files:
            experiment_results = get_experiment_results(performance_file)
            for result in experiment_results:
                writer.writerow([result.experiment.experiment_name,
                                 result.experiment.model_name,
                                 result.experiment.dataset_name,
                                 result.score])
