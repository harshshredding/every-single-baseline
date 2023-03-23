from utils.openai import get_all_chatgpt_predictions_social_dis_ner
from preamble import *
import train_util
from utils.easy_testing import get_dataset_config_by_name

train_samples = train_util.get_train_samples(get_dataset_config_by_name('social_dis_ner_vanilla'))
get_all_chatgpt_predictions_social_dis_ner(
    samples=train_samples,
    output_file_path="./chatgpt_social_dis_ner_train.json"
)