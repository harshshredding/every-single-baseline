from nn_utils import *
from util import *
from train_annos import *
sample_to_token_data = get_train_data()
annos_dict = get_train_annos_dict()
for sample_id in list(sample_to_token_data.keys())[:5]:
    sample_data = sample_to_token_data[sample_id]
    print_list(sample_data)
    text_string = ''.join(get_token_strings(sample_data))
    print(text_string)
    print(annos_dict[sample_id])