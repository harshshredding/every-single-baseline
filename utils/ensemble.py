from collections import defaultdict
from utils.general import read_predictions_file
from structs import SampleAnno

def get_majority_vote_predictions(prediction_file_paths: list[str]):
    votes: defaultdict[SampleAnno, int] = defaultdict(lambda: 0)

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                votes[SampleAnno(sample_id, 'Disease', anno.begin_offset, anno.end_offset)] += 1

    test_annos_majority_votes = {anno: count for anno, count in votes.items() if count > len(prediction_file_paths)//2 }

    return test_annos_majority_votes

