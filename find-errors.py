from train_annos import get_annos_dict
from util import *
validation_file_path = './validation_predictions.tsv'
error_file_path = './errors.tsv'
gold_annos = get_annos_dict(args['gold_file_path'])
pred_annos = get_annos_dict(validation_file_path)
validation_ids = get_validation_ids()
with open(error_file_path, 'w') as error_file:
    error_file.write("tweets_id\tbegin\tend\terror_type")
    for sample_id in validation_ids:
        tweet_gold_annos = gold_annos.get(sample_id, [])
        tweet_gold_annos = [(anno['begin'], anno['end']) for anno in tweet_gold_annos]
        gold_set = set(tweet_gold_annos)
        tweet_pred_annos = pred_annos.get(sample_id, [])
        tweet_pred_annos = [(anno['begin'], anno['end']) for anno in tweet_pred_annos]
        pred_set = set(tweet_pred_annos)
        FP = pred_set.difference(gold_set)
        FN = gold_set.difference(pred_set)
        for span in FP:
            start_offset = span[0]
            end_offset = span[1]
            error_file.write(f"\n{sample_id}\t{int(start_offset)}\t{int(end_offset)}\tFP")
        for span in FN:
            start_offset = span[0]
            end_offset = span[1]
            error_file.write(f"\n{sample_id}\t{int(start_offset)}\t{int(end_offset)}\tFN")