import util

sample_to_token_data = util.get_train_data()
annos_dict = util.get_train_annos_dict()
token_data_list = sample_to_token_data['0f4ec629-df4c-451e-b4ef-bc51f8608b17']
print(util.get_token_strings(token_data_list))
print(util.get_label_idx_dicts())