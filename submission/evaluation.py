from preprocessors.multiconer_preprocessor import read_raw_data
from seqeval.metrics import classification_report
from pathlib import Path
from preamble import *


def evaluate(pred_conll_file_path, output_file):
    print("-"*30, file=output_file)
    print(Path(pred_conll_file_path).name, file=output_file)
    gold_tokens = read_raw_data('multiconer2023/EN-English/en_test.conll')
    pred_tokens = read_raw_data(pred_conll_file_path)

    all_tags_gold = []
    all_tags_pred = []
    for sample_id in gold_tokens:
        assert len(gold_tokens[sample_id]) == len(pred_tokens[sample_id])
        gold_tags = [tag for _, tag in gold_tokens[sample_id]]
        pred_tags = [tag for _, tag in pred_tokens[sample_id]]
        all_tags_gold.append(gold_tags)
        all_tags_pred.append(pred_tags)
    assert len(all_tags_gold) == len(all_tags_pred)
    print(classification_report(all_tags_gold, all_tags_pred, zero_division=1), file=output_file)


with open('submission/multiconer_paper_table_base.txt', 'w') as output_file:
    evaluate('submission/my_submission_preds/span_base_special.conll', output_file)
    evaluate('submission/my_submission_preds/seq_base_special.conll', output_file)

# Span large special tokens Epoch 7
# Seq large special tokens Epoch 7
