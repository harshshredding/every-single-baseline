import util
from structs import Dataset
from args import args, curr_dataset
assert curr_dataset == Dataset.legaleval


util.create_directory_structure(args['gate_input_folder_path'])
util.create_gate_input_mistakes('submissions/errors_legaleval_judgement_bert_epoch_7.tsv',\
                             f"{args['gate_input_folder_path']}/legaleval_errors_judgement.bdocjs")
