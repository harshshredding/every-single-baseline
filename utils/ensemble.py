from collections import defaultdict
from utils.general import read_predictions_file
from structs import SampleAnnotation, DatasetSplit
from utils.evaluation import get_gold_annos_set, get_f1_score_from_sets


def get_majority_vote_predictions(prediction_file_paths: list[str]):
    """
    Combine the predictions of agents using the majority voting strategy i.e 
    only keep those predictions that the majority of agents voted 'yes' on.
    """

    votes: defaultdict[SampleAnnotation, int] = defaultdict(lambda: 0)

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                votes[SampleAnnotation(sample_id, anno.label_type, anno.begin_offset, anno.end_offset)] += 1

    test_annos_majority_votes = {anno: count for anno, count in votes.items() if count > len(prediction_file_paths)//2 }


    majority_predictions = set([anno for anno in test_annos_majority_votes])
    assert len(majority_predictions) == len(test_annos_majority_votes)

    return majority_predictions



def get_union_predictions(prediction_file_paths: list[str]) -> set[SampleAnnotation]:
    union_predictions: set[SampleAnnotation] = set()

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                union_predictions.add(SampleAnnotation(sample_id, anno.label_type, anno.begin_offset, anno.end_offset))

    return union_predictions



def union_results(dataset_config_name: str, test_prediction_file_paths: list[str]):
    gold_predictions = get_gold_annos_set(dataset_config_name=dataset_config_name, split=DatasetSplit.test)
    union_predictions = get_union_predictions(test_prediction_file_paths)
    print(get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=union_predictions))



def get_majority_voting_results(dataset_config_name: str, test_prediction_file_paths: list[str]):
    gold_predictions = get_gold_annos_set(dataset_config_name=dataset_config_name, split=DatasetSplit.test)
    majority_prediction_counts = get_majority_vote_predictions(test_prediction_file_paths)
    majority_predictions = set([anno for anno in majority_prediction_counts])
    assert len(majority_predictions) == len(majority_prediction_counts)
    print(get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=majority_predictions))
