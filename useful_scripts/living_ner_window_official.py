from utils.general import read_predictions_file
from collections import defaultdict
from structs import Anno, Sample
from utils.easy_testing import get_test_samples_by_dataset_name
from utils.evaluation import get_f1_score_from_sets
import csv


def get_original_id_info_from_window_id(sample_id: str) -> tuple:
    separator = '_subsample_'
    assert separator in sample_id
    separator_beginning = sample_id.find(separator)
    original_sample_id = sample_id[:separator_beginning]
    offset = sample_id[(separator_beginning + len(separator)):]
    offset = int(offset)
    assert offset % 50 == 0
    return original_sample_id, offset


def get_original_gold_annos_from_window_annos(window_annos_dict: dict[str, list[Anno]]) -> dict[str, list[Anno]]:
    original_annos_dict = defaultdict(list)
    special_separator = ' [SEP] '

    for window_sample_id, window_annos in window_annos_dict.items():
        original_sample_id, offset = get_original_id_info_from_window_id(window_sample_id)
        assert offset % 50 == 0
        for window_anno in window_annos:
            original_annos_list = original_annos_dict[original_sample_id]
            new_begin_offset = window_anno.begin_offset + offset - len(special_separator)
            new_end_offset = window_anno.end_offset + offset - len(special_separator)

            if offset >= 100:
                new_begin_offset -= 100
                new_end_offset -= 100
            elif offset >= 50:
                new_begin_offset -= 50
                new_end_offset -= 50

            original_annos_list.append(
                Anno(
                    begin_offset=new_begin_offset,
                    end_offset=new_end_offset,
                    label_type=window_anno.label_type,
                    extraction=window_anno.extraction
                )
            )
    return original_annos_dict


def remove_duplicate_annos_position(annos: list[Anno]) -> list[Anno]:
    ret = []
    annos_dict = {}
    for anno in annos:
        annos_dict[(anno.begin_offset, anno.end_offset)] = anno
    for _, anno in annos_dict.items():
        ret.append(anno)
    return ret


def get_annos_set(annos_dict: dict[str, list[Anno]]):
    ret = []
    for sample_id, annos in annos_dict.items():
        annos = [(sample_id, anno.label_type, anno.begin_offset, anno.end_offset)
                 for anno in annos] 
        ret.extend(annos)
    return set(ret)


def get_gold_annos_set(samples: list[Sample]) -> set[tuple[str, str, int, int]]:
    gold_annos = []
    for sample in samples:
        annos = [(sample.id, anno.label_type, anno.begin_offset, anno.end_offset)
                 for anno in sample.annos.gold]
        gold_annos.extend(annos)
    return set(gold_annos)


predictions_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/bionlp_living_ner_majority_all.tsv'
output_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/official_living_majority_all.tsv'


predictions_dict = read_predictions_file(predictions_file_path=predictions_file_path)
original_predictions_dict = get_original_gold_annos_from_window_annos(predictions_dict)
original_predictions_dict_no_duplicates = {sample_id: remove_duplicate_annos_position(annos) 
                                           for sample_id, annos in original_predictions_dict.items()}
assert len(original_predictions_dict) == len(original_predictions_dict_no_duplicates)
predicted_annos_set = get_annos_set(original_predictions_dict_no_duplicates)


test_gold_samples = get_test_samples_by_dataset_name('living_ner_vanilla')
gold_annos_set = get_gold_annos_set(test_gold_samples)

print(get_f1_score_from_sets(gold_set=gold_annos_set, predicted_set=predicted_annos_set))


with open(output_file_path, 'w') as output_tsv: 
    writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
    writer.writerow(['filename', 'mark', 'label', 'off0', 'off1', 'span'])
    for i, predicted_anno in enumerate(predicted_annos_set):
        writer.writerow([str(predicted_anno[0]),
                         f"T{i}",
                         str(predicted_anno[1]),
                         predicted_anno[2],
                         predicted_anno[3],
                         "extraction"])
