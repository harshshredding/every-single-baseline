import util
from utils.report import get_experiment_results, ExperimentResult
from utils.dropbox import get_all_performance_files
from preamble import *
import utils.dropbox as dropbox
import glob
from tabulate import tabulate


def download_all_performance_files():
    util.create_directory_structure('./downloaded_results')
    performance_files_to_download =  
    for performance_file in performance_files_to_download:
        dropbox_path = f'/{performance_file}'
        local_download_path = f'./downloaded_results/{performance_file}'
        dropbox.download_file(dropbox_path, local_download_path)


def find_experiment_result(
        experiment_results: List[ExperimentResult],
        dataset_name: str,
        experiment_name: str,
        model_name: str
):
    return [result for result in experiment_results
            if (result.experiment.experiment_name == experiment_name)
            and (result.experiment.dataset_name == dataset_name)
            and (result.experiment.model_name == model_name)
            ]


def main():
    download_all_performance_files()
    all_performance_file_paths = glob.glob('./downloaded_results/*.csv')
    all_experiment_results: List[ExperimentResult] = []
    for performance_file in all_performance_file_paths:
        all_experiment_results.extend(get_experiment_results(performance_file))
    all_datasets = set([experiment_result.experiment.dataset_name for experiment_result in all_experiment_results])
    all_experiment_model_pairs = set(
        [(experiment_result.experiment.experiment_name, experiment_result.experiment.model_name)
         for experiment_result in all_experiment_results]
    )
    all_experiment_model_pairs = sorted(list(all_experiment_model_pairs))
    all_experiment_names = [experiment_model_pair[0] for experiment_model_pair in all_experiment_model_pairs]
    all_model_names = [experiment_model_pair[1] for experiment_model_pair in all_experiment_model_pairs]
    assert len(all_experiment_names) == len(all_model_names)
    column_names_header = ['dataset_name'] + all_experiment_names
    model_names_header = ['-'] + all_model_names
    table_data = [model_names_header]
    for dataset_name in all_datasets:
        row = [dataset_name]
        for experiment_name, model_name in all_experiment_model_pairs:
            experiment_result = find_experiment_result(all_experiment_results, dataset_name, experiment_name,
                                                       model_name)
            if len(experiment_result):
                assert len(experiment_result) == 1
                experiment_result = experiment_result[0]
                max_performance = str(experiment_result.score)
                row.append(max_performance[:4]
                           if len(max_performance) > 4
                           else max_performance)
            else:
                row.append('n/a')
        table_data.append(row)
    with open('table.txt', 'w') as table_file:
        print(tabulate(table_data, headers=column_names_header), file=table_file)


if __name__ == '__main__':
    main()
