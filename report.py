import util
from utils.report import get_experiment_results_dict, Experiment, ExperimentResult, generate_table_pdf
import utils.dropbox as dropbox
import glob


def download_all_performance_files():
    util.create_directory_structure('./downloaded_results')
    semeval_files = [
        'performance_span_noun_phrase_15_epochs.csv',
        'performance_seq_label_semeval.csv',
        'performance_semeval_spanner.csv',
        'performance_crf_seq_label_semeval.csv'
    ]
    for semeval_performance_file in semeval_files:
        dropbox_path = f'/{semeval_performance_file}'
        local_download_path = f'./downloaded_results/{semeval_performance_file}'
        dropbox.download_file(dropbox_path, local_download_path)


def main():
    download_all_performance_files()
    all_performance_file_paths = glob.glob('./downloaded_results/*.csv')
    experiment_results = []
    for performance_file in all_performance_file_paths:
        for experiment, score in get_experiment_results_dict(performance_file).items():
            experiment_results.append(
                ExperimentResult(Experiment(experiment.experiment_name, experiment.dataset_name), score)
            )
    all_datasets = ['legaleval_judgement', 'legaleval_preamble', 'multiconer_coarse', 'multiconer_fine']
    column_names = ['dataset_name', 'seq_label_semeval', 'semeval_spanner', 'crf_seq_label_semeval', 'span_noun_phrase_15_epochs']
    table_data = []
    for dataset_name in all_datasets:
        row = [dataset_name]
        for experiment_name in column_names[1:]:
            for experiment_result in experiment_results:
                if experiment_result.experiment.experiment_name == experiment_name \
                        and experiment_result.experiment.dataset_name == dataset_name:
                    row.append(experiment_result.max_performance)
        assert len(row) == 5
        table_data.append(row)
    generate_table_pdf(table_data, column_names, './table.pdf')


if __name__ == '__main__':
    main()
