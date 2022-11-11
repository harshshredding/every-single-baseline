import util
from args import args

def create_gate_input_mistakes():
    util.create_directory_structure(args['gate_input_folder_path'])
    sample_to_token_data = util.get_valid_data()
    annos_dict = util.get_mistakes_annos('submissions/multiconer_bert_errors.tsv')
    util.create_gate_file("multiconer_errors", sample_to_token_data, annos_dict, 10000)

create_gate_input_mistakes()
