from structs import Sample
import statistics
from enum import Enum
from utils.general import read_predictions_file

class EvaluationType(Enum):
    f1 = 0
    accuracy = 1

def f1(TP, FP, FN) -> tuple[float, float, float]:
    if (TP + FP) == 0:
        precision = None
    else:
        precision = TP / (TP + FP)
    if (FN + TP) == 0:
        recall = None
    else:
        recall = TP / (FN + TP)
    if (precision is None) or (recall is None) or ((precision + recall) == 0):
        return 0, 0, 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall

def get_f1_score(gold_set: set, predicted_set: set):
    true_positives = len(gold_set.intersection(predicted_set))
    false_positives = len(predicted_set.difference(gold_set))
    false_negatives = len(gold_set.difference(predicted_set))
    return f1(TP=true_positives, FP=false_positives, FN=false_negatives)

def get_all_entity_types_set(samples: list[Sample]) -> set[str]:
    all_entity_types = set()
    for sample in samples:
        for gold_anno in sample.annos.gold:
            all_entity_types.add(gold_anno.label_type)
    return all_entity_types

def get_gold_annos_for_entity_type(samples: list[Sample], entity_type: str) -> set[tuple[str, str, int, int]]:
    gold_annos = []
    for sample in samples:
        annos_of_type = [(sample.id, anno.label_type, anno.begin_offset, anno.end_offset)
                         for anno in sample.annos.gold if anno.label_type == entity_type]
        gold_annos.extend(annos_of_type)
    return set(gold_annos)

def get_gold_annos_set(samples: list[Sample]) -> set[tuple[str, str, int, int]]:
    gold_annos = []
    for sample in samples:
        annos = [(sample.id, anno.label_type, anno.begin_offset, anno.end_offset)
                 for anno in sample.annos.gold]
        gold_annos.extend(annos)
    return set(gold_annos)

def get_predicted_annos_for_entity_type(predictions_file_path: str, entity_type: str) -> set[tuple[str, str, int, int]]:
    predictions_dict = read_predictions_file(predictions_file_path=predictions_file_path)
    predicted_annos = []
    for sample_id, annos in predictions_dict.items():
        predicted_annos.extend(
                [(sample_id, anno.label_type, anno.begin_offset, anno.end_offset)
                 for anno in annos if anno.label_type == entity_type]
        )
    return set(predicted_annos)

def get_predicted_annos_set(predictions_file_path: str) -> set[tuple[str, str, int, int]]:
    predictions_dict = read_predictions_file(predictions_file_path=predictions_file_path)
    predicted_annos = []
    for sample_id, annos in predictions_dict.items():
        predicted_annos.extend(
                [(sample_id, anno.label_type, anno.begin_offset, anno.end_offset)
                 for anno in annos]
        )
    return set(predicted_annos)

def get_macro_f1(predictions_file_path: str, samples: list[Sample]):
    all_entity_types = get_all_entity_types_set(samples)
    assert len(all_entity_types)
    f1_per_entity_type = []
    for entity_type in all_entity_types:
        gold_annos = get_gold_annos_for_entity_type(
            samples=samples, 
            entity_type=entity_type
        )
        predicted_annos = get_predicted_annos_for_entity_type(
            predictions_file_path=predictions_file_path,
            entity_type=entity_type
        )
        f1, _, _ = get_f1_score(gold_set=gold_annos, predicted_set=predicted_annos)
        print(entity_type, f1)
        f1_per_entity_type.append(f1)
    assert len(f1_per_entity_type) == len(all_entity_types)
    return statistics.fmean(f1_per_entity_type)


def get_micro_f1(predictions_file_path: str, samples: list[Sample]):
    gold_annos = get_gold_annos_set(
        samples=samples, 
    )
    predicted_annos = get_predicted_annos_set(
        predictions_file_path=predictions_file_path,
    )
    f1, _, _ = get_f1_score(gold_set=gold_annos, predicted_set=predicted_annos)
    return f1
