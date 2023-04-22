from utils.evaluation import get_f1_score_from_sets, get_gold_annos_set, evaluate_predictions
from structs import DatasetSplit
from utils.ensemble import get_majority_vote_predictions, get_union_predictions, get_majority_voting_results, union_results
import csv

# ************ 
# NCBI
# ************
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


def majority_vote_f1_ncbi_best_models():
    gold_predictions = get_gold_annos_set(dataset_config_name='ncbi_disease_sentence', split=DatasetSplit.test)

    seq_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_test_epoch_10_predictions.tsv'
    span_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_test_epoch_15_predictions.tsv'
    test_prediction_file_paths = [seq_predictions_file_path, span_predictions_file_path]

    majority_prediction_counts = get_majority_vote_predictions(test_prediction_file_paths)

    majority_predictions = set([anno for anno in majority_prediction_counts])
    assert len(majority_predictions) == len(majority_prediction_counts)
    print(get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=majority_predictions))


def union_f1_ncbi_best_models():
    gold_predictions = get_gold_annos_set(dataset_config_name='ncbi_disease_sentence', split=DatasetSplit.test)

    seq_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_test_epoch_10_predictions.tsv'
    span_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_test_epoch_15_predictions.tsv'
    test_prediction_file_paths = [seq_predictions_file_path, span_predictions_file_path]

    union_predictions = get_union_predictions(test_prediction_file_paths)

    print(get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=union_predictions))

def ncbi_evaluate_prediction_files():
    seq_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_test_epoch_10_predictions.tsv'
    span_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_test_epoch_15_predictions.tsv'
    evaluate_predictions(gold_dataset_config='ncbi_disease_sentence', predictions_file_path=seq_predictions_file_path, dataset_split=DatasetSplit.test)
    evaluate_predictions(gold_dataset_config='ncbi_disease_sentence', predictions_file_path=span_predictions_file_path, dataset_split=DatasetSplit.test)







# ************ 
# LivingNER
# ************
def union_living_ner_window_combo():
    prediction_file_paths = [
        '/Users/harshverma/every-single-baseline/baseline_paper/predictions/living_ner/experiment_living_ner_combo_seq_more_testing_bigger_batch_living_ner_window_combo_model_seq_large_default_test_epoch_4_predictions.tsv',
        '/Users/harshverma/every-single-baseline/baseline_paper/predictions/living_ner/experiment_living_ner_combo_window_span_fixed_width_more_testing_bigger_batch_living_ner_window_combo_model_span_large_default_test_epoch_4_predictions.tsv'
    ]
    union_results(dataset_config_name='living_ner_window_combo', test_prediction_file_paths=prediction_file_paths)


def majority_living_ner_window_combo():
    prediction_file_paths = [
            '/Users/harshverma/every-single-baseline/baseline_paper/predictions/living_ner/experiment_living_ner_combo_seq_more_testing_bigger_batch_living_ner_window_combo_model_seq_large_default_test_epoch_4_predictions.tsv',
            '/Users/harshverma/every-single-baseline/baseline_paper/predictions/living_ner/experiment_living_ner_combo_window_span_fixed_width_more_testing_bigger_batch_living_ner_window_combo_model_span_large_default_test_epoch_4_predictions.tsv'
    ]
    get_majority_voting_results(dataset_config_name='living_ner_window_combo', test_prediction_file_paths=prediction_file_paths)




# ****************
# Social Dis NER
# ****************
def union_social_dis_ner():
    test_prediction_file_paths = [
        "/Users/harshverma/meta_bionlp/social_dis_ner/test/Apps/harshv_research_nlp/experiment_social_dis_ner_bionlp_all_0_social_dis_ner_vanilla_model_span_large_default_test_epoch_16_predictions.tsv",
        "/Users/harshverma/meta_bionlp/social_dis_ner/test/Apps/harshv_research_nlp/experiment_social_dis_ner_bionlp_all_2_social_dis_ner_vanilla_model_seq_large_default_test_epoch_5_predictions.tsv"
    ]
    union_predictions = get_union_predictions(test_prediction_file_paths)
    output_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/social_union_bionlp_adafactor.tsv'
    with open(output_file_path, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['tweets_id', 'begin', 'end', 'type', 'extraction'])
        for union_prediction in union_predictions:
            writer.writerow([union_prediction.sample_id,
                             str(union_prediction.begin_offset),
                             str(union_prediction.end_offset),
                             union_prediction.type_string,
                             "extraction"])


def majority_social_dis_ner():
    test_prediction_file_paths = [
        "/Users/harshverma/meta_bionlp/social_dis_ner/test/Apps/harshv_research_nlp/experiment_social_dis_ner_bionlp_all_0_social_dis_ner_vanilla_model_span_large_default_test_epoch_16_predictions.tsv",
        "/Users/harshverma/meta_bionlp/social_dis_ner/test/Apps/harshv_research_nlp/experiment_social_dis_ner_bionlp_all_2_social_dis_ner_vanilla_model_seq_large_default_test_epoch_5_predictions.tsv"
    ]
    majority_prediction_counts = get_majority_vote_predictions(test_prediction_file_paths)
    majority_predictions = set([anno for anno in majority_prediction_counts])
    assert len(majority_predictions) == len(majority_prediction_counts)

    output_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/social_majority_bionlp_adafactor.tsv'
    with open(output_file_path, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['tweets_id', 'begin', 'end', 'type', 'extraction'])
        for majority_prediction in majority_predictions:
            writer.writerow([majority_prediction.sample_id,
                             str(majority_prediction.begin_offset),
                             str(majority_prediction.end_offset),
                             majority_prediction.type_string,
                             "extraction"])



# ****************
# MultiCoNER
# ****************
def union_multiconer_custom_tokenization_batched():
    test_prediction_file_paths = [
"/Users/harshverma/every-single-baseline/meta/multiconer/test/custom_tokenization_batched/harshv_research_nlp/experiment_multiconer_custom_tokens_batched_multiconer_fine_tokens_seq_large_custom_tokenization_test_epoch_7_predictions.tsv",
"/Users/harshverma/every-single-baseline/meta/multiconer/test/custom_tokenization_batched/harshv_research_nlp/experiment_multiconer_custom_tokens_batched_multiconer_fine_tokens_span_large_custom_tokenization_test_epoch_7_predictions.tsv"
    ]
    union_predictions = get_union_predictions(test_prediction_file_paths)

    output_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/multiconer_union_custom_tokenization_batched.tsv'
    with open(output_file_path, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['sample_id', 'begin', 'end', 'type', 'extraction'])
        for union_prediction in union_predictions:
            writer.writerow([union_prediction.sample_id,
                             str(union_prediction.begin_offset),
                             str(union_prediction.end_offset),
                             union_prediction.type_string,
                             "extraction"])


def majority_multiconer_custom_tokenization_batched():
    test_prediction_file_paths = [
"/Users/harshverma/every-single-baseline/meta/multiconer/test/custom_tokenization_batched/harshv_research_nlp/experiment_multiconer_custom_tokens_batched_multiconer_fine_tokens_seq_large_custom_tokenization_test_epoch_7_predictions.tsv",
"/Users/harshverma/every-single-baseline/meta/multiconer/test/custom_tokenization_batched/harshv_research_nlp/experiment_multiconer_custom_tokens_batched_multiconer_fine_tokens_span_large_custom_tokenization_test_epoch_7_predictions.tsv"
    ]
    majority_prediction_counts = get_majority_vote_predictions(test_prediction_file_paths)
    majority_predictions = set([anno for anno in majority_prediction_counts])
    assert len(majority_predictions) == len(majority_prediction_counts)

    output_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/multiconer_majority_custom_tokenization_batched.tsv'
    with open(output_file_path, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['sample_id', 'begin', 'end', 'type', 'extraction'])
        for majority_prediction in majority_predictions:
            writer.writerow([majority_prediction.sample_id,
                             str(majority_prediction.begin_offset),
                             str(majority_prediction.end_offset),
                             majority_prediction.type_string,
                             "extraction"])



# ************ 
# Genia 
# ************
def union_genia_sentence():
    prediction_file_paths = [
'/Users/harshverma/every-single-baseline/meta/genia/test/experiment_genia_sentence_level_bio_genia_config_vanilla_model_seq_large_bio_test_epoch_4_predictions.tsv',
'/Users/harshverma/every-single-baseline/meta/genia/test/experiment_genia_sentence_level_bio_genia_config_vanilla_model_span_large_bio_default_test_epoch_7_predictions.tsv'
    ]
    union_results(dataset_config_name='genia_config_vanilla', test_prediction_file_paths=prediction_file_paths)


def majority_genia_sentence():
    prediction_file_paths = [
'/Users/harshverma/every-single-baseline/meta/genia/test/experiment_genia_sentence_level_bio_genia_config_vanilla_model_seq_large_bio_test_epoch_4_predictions.tsv',
'/Users/harshverma/every-single-baseline/meta/genia/test/experiment_genia_sentence_level_bio_genia_config_vanilla_model_span_large_bio_default_test_epoch_7_predictions.tsv'
    ]
    get_majority_voting_results(dataset_config_name='genia_config_vanilla', test_prediction_file_paths=prediction_file_paths)
