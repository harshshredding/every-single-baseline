from evaluation_util import score_task2
from util import get_tweet_data
tweet_data = get_tweet_data('./socialdisner-data/train-valid-txt'
                            '-files/validation')
pred_file_path = './submissions/validation_predictions_encoder_light_9.tsv'
gold_file_path = './validation_entities.tsv'
score_task2(pred_file_path, gold_file_path, tweet_data, './results.txt')