from read_gate_output import *
sample_to_token_data = get_sample_to_token_data('train-2.json')
sample_id = '1425026916625666065'
print(list(zip(get_token_strings(sample_to_token_data[sample_id]), get_labels(sample_to_token_data[sample_id]))))
