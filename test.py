from nn_utils import *
from util import *
from train_annos import *
sample_to_token_data = get_train_data()
annos_dict = get_train_annos_dict()
for sample_id in list(sample_to_token_data.keys())[:5]:
    sample_data = sample_to_token_data[sample_id]
    sample_annos = annos_dict[sample_id]
    print("DATA:")
    print_list(sample_data)
    print("ANNOS:")
    print(sample_annos)
    text_string = ' '.join(get_token_strings(sample_data))
    print("TEXT:")
    print(text_string)
    print("TOKEN LABELS")
    print(get_label_strings(sample_data, sample_annos))
    print("BIO TOKEN LABELS")
    print(get_labels_bio(sample_data, sample_annos))