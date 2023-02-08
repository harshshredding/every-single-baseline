from utils.openai import get_diseases_social_dis_ner, get_diseases_social_dis_ner_spanish_version
import util
import train_util
from utils.config import get_dataset_config_by_name

valid_samples = train_util.get_valid_samples(get_dataset_config_by_name('social_dis_ner'))
# print(get_diseases_social_dis_ner(tweet))
