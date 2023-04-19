from collections import defaultdict
from utils.general import read_predictions_file
from structs import SampleAnno


def get_majority_vote_predictions(prediction_file_paths: list[str]):
    """
    Combine the predictions of agents using the majority voting strategy i.e 
    only keep those predictions that the majority of agents voted 'yes' on.
    """

    votes: defaultdict[SampleAnno, int] = defaultdict(lambda: 0)

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                votes[SampleAnno(sample_id, anno.label_type, anno.begin_offset, anno.end_offset)] += 1

    test_annos_majority_votes = {anno: count for anno, count in votes.items() if count > len(prediction_file_paths)//2 }

    return test_annos_majority_votes



def get_union_predictions(prediction_file_paths: list[str]) -> set[SampleAnno]:
    union_predictions: set[SampleAnno] = set()

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                union_predictions.add(SampleAnno(sample_id, anno.label_type, anno.begin_offset, anno.end_offset))

    return union_predictions

