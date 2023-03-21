import train_util
from utils.easy_testing import get_dataset_config_by_name
from train_util import prepare_model
from utils.config import get_experiment_config

valid_samples_new = train_util.get_valid_samples(get_dataset_config_by_name('multiconer_fine_vanilla'))
valid_samples_old = train_util.get_valid_samples(get_dataset_config_by_name('multiconer_fine_tokens'))
first_sample_new = valid_samples_new[0]
first_sample_old = valid_samples_old[0]

seq_new_experiment = get_experiment_config(
    model_config_module_name='model_seq_base_no_special_tokens',
    dataset_config_name='multiconer_fine_vanilla'
)

seq_old_experiment = get_experiment_config(
    model_config_module_name='model_seq_base_custom_tokenization_no_batch',
    dataset_config_name='multiconer_fine_tokens'
)
new_seq_model = prepare_model(seq_new_experiment.model_config, seq_new_experiment.dataset_config)
old_seq_model = prepare_model(seq_old_experiment.model_config, seq_old_experiment.dataset_config)

print(type(new_seq_model))
print(type(old_seq_model))

for sample_new, sample_old in zip(valid_samples_new, valid_samples_old):
    collect_new = []
    collect_old = []
    loss_new, predictions_new = new_seq_model([sample_new], collect_new)
    loss_old, predictions_old = old_seq_model([sample_old], collect_old)
    assert all((collect_new[0].input_ids == collect_old[0].input_ids).numpy()[0]),\
        f"Input ids dont match:\nnew{collect_new[0].input_ids}\nold{collect_old[0].input_ids}"
    if collect_new[1] != collect_old[1]:
        print()
        print()
        print(
            f"Gold bio labels dont match:\nnew{collect_new[1]}\nold{collect_old[1]}"
            f"\ntokens:{collect_old[0].tokens()}"
            f"\ntokens:{collect_old[0].word_ids()}"
            f"\ntokens:{collect_old[0].input_ids.numpy()}"
            f"\ntokens:{list(zip(collect_old[0].tokens(), collect_new[1]))}"
            f"\nsample_id:{sample_new.id}"
        )
