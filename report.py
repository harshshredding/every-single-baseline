import util
from utils.report import get_experiment_results_dict, Experiment, ExperimentResult, generate_table_pdf
from preamble import *
import utils.dropbox as dropbox
import glob
from tabulate import tabulate


def download_all_performance_files():
    util.create_directory_structure('./downloaded_results')
    performance_files_to_download = [
        'performance_span_noun_phrase_15_epochs.csv',
        'performance_seq_label_semeval.csv',
        'performance_semeval_spanner.csv',
        'performance_crf_seq_label_semeval.csv',
        'performance_submit_heuristic.csv',
        'performance_submit_span_width_embed_large.csv',
        'performance_seq_large_experiment.csv',
        'performance_submit_multiconer.csv',
        'performance_large_legal_and_genia.csv',
        'performance_multiconer_experiment_after_submission.csv',
        'performance_gpt_experiment_large_special_token_span.csv',
        'performance_gpt_experiment_base_special_span.csv',
        'performance_social_dis_ner_no_custom_tokenization.csv',
        'performance_gpt_experiment_large_special_token.csv',
        'performance_gpt_experiment_base_special_token.csv',
        'performance_gpt_experiment_large.csv',
        'performance_gpt_experiment_base.csv',
        'performance_batched_biomedical_base.csv'
    ]
    for performance_file in performance_files_to_download:
        dropbox_path = f'/{performance_file}'
        local_download_path = f'./downloaded_results/{performance_file}'
        dropbox.download_file(dropbox_path, local_download_path)


def find_experiment_result(experiment_results: List[ExperimentResult], dataset_name: str, experiment_name: str):
    return [result for result in experiment_results
            if (result.experiment.experiment_name == experiment_name)
            and (result.experiment.dataset_name == dataset_name)
            ]


def main():
    download_all_performance_files()
    all_performance_file_paths = glob.glob('./downloaded_results/*.csv')
    experiment_results = []
    for performance_file in all_performance_file_paths:
        for experiment, score in get_experiment_results_dict(performance_file).items():
            experiment_results.append(
                ExperimentResult(experiment, score)
            )
    all_datasets = ['legaleval_judgement', 'legaleval_preamble', 'multiconer_coarse', 'multiconer_fine', 'genia']
    column_names = ['dataset_name', 'seq_label_semeval', 'semeval_spanner', 'crf_seq_label_semeval',
                    'span_noun_phrase_15_epochs', 'submit_multiconer',
                    'submit_heuristic', 'submit_span_width_embed_large', 'seq_large_experiment', 'large_legal_and_genia'
                    ]
    table_data = []
    for dataset_name in all_datasets:
        row = [dataset_name]
        for experiment_name in column_names[1:]:
            experiment_result = find_experiment_result(experiment_results, dataset_name, experiment_name)
            if len(experiment_result):
                assert len(experiment_result) == 1
                experiment_result = experiment_result[0]
                max_performance = str(experiment_result.max_performance)
                row.append(max_performance[:4]
                           if len(max_performance) > 4
                           else max_performance)
            else:
                row.append('n/a')
        assert len(row) == 10
        table_data.append(row)
    with open('table.txt', 'w') as table_file:
        print(tabulate(table_data, headers=column_names), file=table_file)


if __name__ == '__main__':
    main()
