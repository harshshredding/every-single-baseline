import json

train_samples_file_path = './preprocessed_data/living_ner_train_window_production_samples.json'
valid_samples_file_path = './preprocessed_data/living_ner_valid_window_production_samples.json'
test_samples_file_path = './preprocessed_data/living_ner_test_window_production_samples.json'

new_train_samples_file_path = './preprocessed_data/living_ner_train_window_combo_production_samples.json'
new_valid_samples_file_path = './preprocessed_data/living_ner_valid_window_combo_production_samples.json'
new_test_samples_file_path = './preprocessed_data/living_ner_test_window_combo_production_samples.json'

with open(train_samples_file_path, 'r') as train_file:
    train_sample_list_raw = json.load(train_file)
    print("num orig train samples", len(train_sample_list_raw))

with open(valid_samples_file_path, 'r') as valid_file:
    valid_sample_list_raw = json.load(valid_file)
    print("num orig valid samples", len(valid_sample_list_raw))

with open(test_samples_file_path, 'r') as test_file:
    test_sample_list_raw = json.load(test_file)
    print("num orig test samples", len(test_sample_list_raw))

new_train_list_raw = train_sample_list_raw + valid_sample_list_raw
assert len(new_train_list_raw) == len(train_sample_list_raw) + len(valid_sample_list_raw)

with open(new_train_samples_file_path, 'w') as train_out_file,\
     open(new_valid_samples_file_path, 'w') as valid_out_file,\
     open(new_test_samples_file_path, 'w') as test_out_file:
    json.dump(new_train_list_raw, train_out_file, default=vars)
    json.dump(valid_sample_list_raw, valid_out_file, default=vars)
    json.dump(test_sample_list_raw, test_out_file, default=vars)

