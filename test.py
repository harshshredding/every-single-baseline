from utils.config import get_model_config_by_name, get_dataset_config_by_name
from utils.universal import *
from models import JustBert3ClassesCRF
import util
import train_util

model_config = get_model_config_by_name('JustBert3Classes')
dataset_config = get_dataset_config_by_name('multiconer_coarse')
all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
token_data_dict = train_util.get_valid_token_data_dict(dataset_config)
annos_dict = train_util.get_valid_annos_dict(dataset_config)
sample_token_data = token_data_dict['5239d808-f300-46ea-aa3b-5093040213a3']
sample_annos = annos_dict['5239d808-f300-46ea-aa3b-5093040213a3']
model = JustBert3ClassesCRF(all_types, model_config, dataset_config)
model.forward(sample_token_data, sample_annos, model_config)
pass