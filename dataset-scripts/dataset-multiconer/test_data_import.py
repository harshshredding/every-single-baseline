import util
from train_annos import *

sample_to_token_data = util.get_train_data()
annos_dict = util.get_train_annos_dict()
token_data_list = sample_to_token_data['dc6c6d00-daf8-454b-9d4f-f5e36f223365']
for token_data in token_data_list:
    print(token_data)
print(util.get_token_strings(token_data_list))
for anno in annos_dict['dc6c6d00-daf8-454b-9d4f-f5e36f223365']:
    print(anno)
util.create_gate_file("genia", sample_to_token_data, annos_dict, 100)

#create_gate_file("multiconer", sample_to_token_data, annos_dict, 100)