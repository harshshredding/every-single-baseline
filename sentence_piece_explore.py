import train_util
from utils.easy_testing import get_dataset_config_by_name
from train_util import prepare_model
from utils.config import get_experiment_config
from preamble import *

test_samples_new = train_util.get_test_samples(get_dataset_config_by_name('multiconer_fine_vanilla'))

seq_new_experiment = get_experiment_config(
    model_config_module_name='model_seq_large_default',
    dataset_config_name='multiconer_fine_vanilla'
)

new_seq_model = prepare_model(seq_new_experiment.model_config, seq_new_experiment.dataset_config)

print(type(new_seq_model))

for sample in test_samples_new:
    loss_new, predictions_new = new_seq_model([sample])
