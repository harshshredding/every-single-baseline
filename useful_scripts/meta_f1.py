from utils.easy_testing import get_test_samples_by_dataset_name
from utils.evaluation import get_f1_score_from_sets
from utils.general import read_predictions_file
import pandas as pd
from structs import SampleAnnotation, Annotation
import csv
from collections import defaultdict


def read_meta_predictions_file(predictions_file_path, entity_type: 'str') -> set[SampleAnnotation]:
    df = pd.read_csv(predictions_file_path, sep='\t')
    ret = set()
    num_removed = 0
    for _, row in df.iterrows():
        sample_id = str(row['sample_id'])
        original_sample_id, start, end = sample_id.split('@@@')
        label = row['label']
        assert label in ['correct', 'incorrect']
        if label == 'correct':
            ret.add(SampleAnnotation(str(original_sample_id), entity_type, int(start), int(end)))
        else:
            num_removed += 1
    print(f"removed {num_removed} predictions")
    return ret


def read_meta_predictions_file_with_type_information(predictions_file_path) -> set[SampleAnnotation]:
    df = pd.read_csv(predictions_file_path, sep='\t')
    ret = set()
    num_removed = 0
    for _, row in df.iterrows():
        sample_id = str(row['sample_id'])
        original_sample_id, start, end, entity_type = sample_id.split('@@@')
        label = row['label']
        assert label in ['correct', 'incorrect']
        if label == 'correct':
            ret.add(SampleAnnotation(str(original_sample_id), entity_type, int(start), int(end)))
        else:
            num_removed += 1
    print(f"removed {num_removed} predictions")
    return ret


def meta_genia():
    meta_predictions_file_path = '/Users/harshverma/meta_bionlp/genia/filter/experiment_genia_meta_0_genia_meta_model_meta_special_weighted_bio_test_epoch_11_predictions.tsv'
    meta_predictions_set = read_meta_predictions_file_with_type_information(meta_predictions_file_path)

    gold_samples = get_test_samples_by_dataset_name('genia_config_vanilla')
    gold_predictions: set[SampleAnnotation] = set()
    for gold_sample in gold_samples:
        for gold_anno in gold_sample.annos.gold:
            gold_predictions.add(SampleAnnotation(str(gold_sample.id), gold_anno.label_type, int(gold_anno.begin_offset), int(gold_anno.end_offset)))

    print(get_f1_score_from_sets(gold_predictions, meta_predictions_set))


def evaluate_meta_predictions(meta_predictions_file_path: str, dataset_config_name: str):
    """
    Given meta's predictions in `meta_predictions_file_path` for a dataset
    corresponding to `dataset_config_name`, evaluate meta's performance.
    """
    meta_predictions_set = read_meta_predictions_file_with_type_information(meta_predictions_file_path)
    gold_samples = get_test_samples_by_dataset_name(dataset_config_name)
    gold_predictions: set[SampleAnnotation] = set()
    for gold_sample in gold_samples:
        for gold_anno in gold_sample.annos.gold:
            gold_predictions.add(SampleAnnotation(str(gold_sample.id), gold_anno.label_type, int(gold_anno.begin_offset), int(gold_anno.end_offset)))

    print(get_f1_score_from_sets(gold_predictions, meta_predictions_set))




def meta_f1():
    raise NotImplementedError()
    for i in range(30):
        predictions_file_path = f'/Users/harshverma/every-single-baseline/meta/ncbi/predictions/combined/adam_all_mistakes_all_gold_batch_16/Apps/harshv_research_nlp/experiment_ncbi_meta_weighted_all_mistakes_all_gold_bigger_batch_1_ncbi_sentence_all_mistakes_all_gold_model_meta_special_weighted_bio_test_epoch_{i}_predictions.tsv'
        meta_predictions = read_meta_predictions_file(predictions_file_path=predictions_file_path, entity_type='Disease')
        gold_predictions = set()

        gold_samples = get_test_samples_by_dataset_name('ncbi_disease_sentence')
        gold_samples_dict = {sample.id: sample for sample in gold_samples}
        for meta_prediction in meta_predictions:
            assert meta_prediction[0] in gold_samples_dict

        for gold_sample in gold_samples:
            for gold_anno in gold_sample.annos.gold:
                gold_predictions.add((str(gold_sample.id), int(gold_anno.begin_offset), int(gold_anno.end_offset), 'Disease'))

        assert len(gold_predictions) and len(meta_predictions)
        
        print(i, get_f1_score_from_sets(gold_predictions, meta_predictions))


def get_living_ner_all_test_predictions() -> defaultdict[str, list[Annotation]]:
    test_prediction_file_paths = [
'/Users/harshverma/meta_bionlp/living_ner/test/experiment_specific_living_ner_bionlp_seq_adafactor_0_living_ner_window_model_seq_large_default_test_epoch_9_predictions.tsv',
'/Users/harshverma/meta_bionlp/living_ner/test/experiment_specific_living_ner_bionlp_span_adafactor_0_living_ner_window_model_span_large_default_test_epoch_8_predictions.tsv'
    ]
    all_predictions_dict = defaultdict(list)
    assert len(test_prediction_file_paths) == 2
    for prediction_file_path in test_prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            all_predictions_dict[sample_id].extend(annos)
    return all_predictions_dict


def living_ner_meta():
    print("hello")
    filter_path = '/Users/harshverma/meta_bionlp/living_ner/filter/experiment_living_ner_meta_0_living_ner_meta_model_meta_special_weighted_test_epoch_8_predictions.tsv'
    filtered_predictions = read_meta_predictions_file(predictions_file_path=filter_path, entity_type='DEFAULT')
    filtered_predictions = set([(pred.sample_id, pred.begin_offset, pred.end_offset) for pred in filtered_predictions])
    filtered_predictions_with_types: set[SampleAnnotation] = set()

    all_test_predicitons = get_living_ner_all_test_predictions()

    for filtered_prediction in filtered_predictions:
        sample_id = filtered_prediction[0]
        corresponding_anno = [
         anno 
         for anno in all_test_predicitons[sample_id] 
         if anno.begin_offset == filtered_prediction[1] and anno.end_offset == filtered_prediction[2]
        ]
        assert len(corresponding_anno)
        filtered_predictions_with_types.add(
                SampleAnnotation(
                    sample_id=sample_id,
                    type_string=corresponding_anno[0].label_type,
                    begin_offset=corresponding_anno[0].begin_offset,
                    end_offset=corresponding_anno[0].end_offset
                )
        )

    gold_samples = get_test_samples_by_dataset_name('living_ner_window')

    gold_predictions: set[SampleAnnotation] = set()

    for gold_sample in gold_samples:
        for gold_anno in gold_sample.annos.gold:
            gold_predictions.add(SampleAnnotation(str(gold_sample.id), gold_anno.label_type, int(gold_anno.begin_offset), int(gold_anno.end_offset)))

    print(get_f1_score_from_sets(gold_predictions, filtered_predictions_with_types))

    out_file_path = '/Users/harshverma/every-single-baseline/useful_scripts/bionlp_living_ner_meta.tsv'

    write_predictions(filtered_predictions_with_types, output_file_path=out_file_path)


def write_predictions(predictions: set[SampleAnnotation], output_file_path: str):
    with open(output_file_path, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['sample_id', 'begin', 'end', 'type', 'extraction'])
        for prediction in predictions:
            writer.writerow([prediction.sample_id,
                             str(prediction.begin_offset),
                             str(prediction.end_offset),
                             prediction.type_string,
                             "extraction"])



def social_dis_ner_meta():
    predictions_file_path = '/Users/harshverma/meta_bionlp/social_dis_ner/combined/Apps/harshv_research_nlp/experiment_social_dis_ner_meta_0_social_dis_ner_meta_model_meta_special_weighted_test_epoch_8_predictions.tsv'
    output_file_path = '/Users/harshverma/meta_bionlp/social_dis_ner/submission/social_meta_adafactor_epoch_8.tsv'

    meta_predictions = read_meta_predictions_file(predictions_file_path=predictions_file_path, entity_type='ENFERMEDAD')
    with open(output_file_path, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['tweets_id', 'begin', 'end', 'type', 'extraction'])
        for union_prediction in meta_predictions:
            writer.writerow([union_prediction.sample_id,
                             str(union_prediction.begin_offset),
                             str(union_prediction.end_offset),
                             union_prediction.type_string,
                             "extraction"])

