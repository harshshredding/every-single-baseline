from util import *
from train_annos import *

sample_to_token_data = get_train_data()
annos_dict = get_train_annos_dict()
create_gate_file("genia", sample_to_token_data, annos_dict, 100)
