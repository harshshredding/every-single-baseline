from utils.easy_testing import get_test_samples_by_dataset_name
from util import get_f1_score
import pandas as pd

def read_meta_predictions_file(predictions_file_path) -> set:
    df = pd.read_csv(predictions_file_path, sep='\t')
    ret = set()
    num_removed = 0
    for _, row in df.iterrows():
        sample_id = str(row['sample_id'])
        original_sample_id, start, end = sample_id.split('@@@')
        label = row['label']
        assert label in ['correct', 'incorrect']
        if label == 'correct':
            ret.add((str(original_sample_id), int(start), int(end), 'Disease'))
        else:
            num_removed += 1
    print(f"removed {num_removed} predictions")
    return ret

def meta_f1():
    for i in range(30):
        predictions_file_path = f'/Users/harshverma/every-single-baseline/meta/ncbi/predictions/combined/special_tokens_adam/Apps/harshv_research_nlp/experiment_ncbi_meta_special_tokens_bigger_training_1_ncbi_sentence_meta_bigger_special_tokens_model_meta_special_tokens_bio_test_epoch_{i}_predictions.tsv'
        meta_predictions = read_meta_predictions_file(predictions_file_path=predictions_file_path)
        gold_predictions = set()

        gold_samples = get_test_samples_by_dataset_name('ncbi_disease_sentence')
        gold_samples_dict = {sample.id: sample for sample in gold_samples}
        for meta_prediction in meta_predictions:
            assert meta_prediction[0] in gold_samples_dict

        for gold_sample in gold_samples:
            for gold_anno in gold_sample.annos.gold:
                gold_predictions.add((str(gold_sample.id), int(gold_anno.begin_offset), int(gold_anno.end_offset), 'Disease'))

        assert len(gold_predictions) and len(meta_predictions)
        
        print(i, get_f1_score(gold_predictions, meta_predictions))
