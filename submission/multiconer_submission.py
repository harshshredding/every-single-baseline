import util
import pandas as pd
from structs import Anno, SampleId
from typing import List, Dict
from preprocessors.multiconer_preprocessor import read_raw_data, read_raw_data_list

test_data_file_path = "./multiconer-data-raw/public_data/EN-English/en_test.conll"


def read_multiconer_predictions() -> dict[str, List[Anno]]:
    predictions_file_path = './submission/predictions/' \
                            'submit_multiconer_multiconer_fine_test_results_epoch_9_predictions.tsv'
    df = pd.read_csv(predictions_file_path, sep='\t')
    sample_to_annos = {}
    for _, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        annos_list.append(
            Anno(
                begin_offset=int(row['begin']),
                end_offset=int(row['end']),
                label_type=row['type'],
                extraction=row['extraction'],
            )
        )
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos


def not_the_same_anno(curr_anno: Anno, other_anno: Anno):
    return not ((other_anno.begin_offset == curr_anno.begin_offset)
                and (other_anno.end_offset == curr_anno.end_offset)
                )


def curr_is_contained(curr_anno: Anno, other_anno: Anno):
    return (other_anno.begin_offset <= curr_anno.begin_offset) and (curr_anno.end_offset <= other_anno.end_offset)


def remove_nesting(sample_annos: List[Anno]) -> List[Anno]:
    to_remove = []
    for curr_anno in sample_annos:
        containers = [
            other_anno
            for other_anno in sample_annos
            if not_the_same_anno(curr_anno, other_anno) and curr_is_contained(curr_anno=curr_anno,
                                                                              other_anno=other_anno)
        ]
        to_remove.extend(containers)
    return [anno for anno in sample_annos if anno not in to_remove]


def get_predictions_without_nesting() -> dict[str, List[Anno]]:
    predictions = read_multiconer_predictions()
    return {sample_id: remove_nesting(annos) for sample_id, annos in predictions.items()}


def get_test_token_dict():
    return read_raw_data(test_data_file_path)


def get_test_data_list():
    return read_raw_data_list(test_data_file_path)


def print_duplicate_test_data():
    test_data_list = get_test_data_list()
    all_sentences_dict = {}
    for sample_id, tokens in test_data_list:
        if sample_id in all_sentences_dict:
            all_sentences_dict[sample_id].append(tokens)
        else:
            all_sentences_dict[sample_id] = [tokens]
    for sample_id in all_sentences_dict:
        if len(all_sentences_dict[sample_id]) > 1:
            print("----------------------")
            print(f"SAMPLE_ID: {sample_id}")
            for tokens in all_sentences_dict[sample_id]:
                print(tokens)
                print()


def get_test_sample_ids():
    return set([line.strip()[5:] for line in open(test_data_file_path) if line.startswith("#")])


def main():
    test_data_list = get_test_data_list()
    predictions = get_predictions_without_nesting()
    with open('./submission/en.pred.conll', 'w') as submission_file_output:
        for sample_id, tokens in test_data_list:
            print(f'# id {sample_id}', file=submission_file_output)
            sample_annos = predictions.get(sample_id, [])
            curr_offset = 0
            for token_string, _ in tokens:
                token_begin = curr_offset
                token_end = curr_offset + len(token_string)
                containers = [anno for anno in sample_annos if (anno.begin_offset <= token_begin)
                              and (token_end <= anno.end_offset)]
                if len(containers):
                    if len(containers) > 1:
                        print(f"WARN: sample_id: {sample_id} token {token_string} overlapped by more than one")
                    token_anno = containers[0]
                    if int(token_anno.begin_offset) == int(token_begin):
                        print(f"B-{token_anno.label_type}", file=submission_file_output)
                    else:
                        print(f"I-{token_anno.label_type}", file=submission_file_output)
                else:
                    print('O', file=submission_file_output)
                curr_offset = curr_offset + len(token_string) + 1
            print('', file=submission_file_output)
    test_file_length = sum(1 for line in open('./multiconer-data-raw/public_data/EN-English/en_test.conll'))
    submission_file_length = sum(1 for line in open('./submission/en.pred.conll'))
    assert test_file_length == submission_file_length, f"length not equal testfile:{test_file_length} " \
                                                       f"submission_file:{submission_file_length}"


if __name__ == '__main__':
    main()
