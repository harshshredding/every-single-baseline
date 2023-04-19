from utils.evaluation import get_f1_score_from_sets, get_gold_annos_set
from structs import DatasetSplit
from utils.ensemble import get_majority_vote_predictions


def majority_vote_f1_ncbi_several():
    gold_predictions = get_gold_annos_set(dataset_config_name='ncbi_disease_sentence', split=DatasetSplit.test)

    test_prediction_file_paths = []
    for i in range(10, 14):
        test_prediction_file_paths.append(
            f'/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_test_epoch_{i}_predictions.tsv'
        )

    for i in range(15, 19):
        test_prediction_file_paths.append(
            f'/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_test_epoch_{i}_predictions.tsv'
        )
    assert len(test_prediction_file_paths) == 8
    majority_prediction_counts = get_majority_vote_predictions(test_prediction_file_paths)

    majority_predictions = set([anno for anno in majority_prediction_counts])
    assert len(majority_predictions) == len(majority_prediction_counts)
    print(get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=majority_predictions))


def majority_vote_f1_ncbi_special_models():
    gold_predictions = get_gold_annos_set(dataset_config_name='ncbi_disease_sentence', split=DatasetSplit.test)

    seq_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_test_epoch_10_predictions.tsv'
    span_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_test_epoch_15_predictions.tsv'
    test_prediction_file_paths = [seq_predictions_file_path, span_predictions_file_path]

    majority_prediction_counts = get_majority_vote_predictions(test_prediction_file_paths)

    majority_predictions = set([anno for anno in majority_prediction_counts])
    assert len(majority_predictions) == len(majority_prediction_counts)
    print(get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=majority_predictions))
