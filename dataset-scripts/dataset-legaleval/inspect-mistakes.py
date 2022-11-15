import util
from args import args

def create_gate_input_mistakes():
    util.create_directory_structure(args['gate_input_folder_path'])
    sample_to_token_data = util.get_valid_data()
    gold_annos_dict = util.get_valid_annos_dict()
    mistake_annos_dict = util.get_mistakes_annos('submissions/errors_legaleval_judgement_bert_epoch_7.tsv')
    combined_annos_dict = {}
    for sample_id in gold_annos_dict:
        gold_annos_list = gold_annos_dict[sample_id]
        mistake_annos_list = mistake_annos_dict.get(sample_id, [])
        combined_list = gold_annos_list + mistake_annos_list
        for anno in combined_list:
            anno.begin_offset = int(anno.begin_offset)
            anno.end_offset = int(anno.end_offset)
        combined_annos_dict[sample_id] = combined_list 
    util.create_gate_file("legaleval_errors_judgement", sample_to_token_data, combined_annos_dict)

create_gate_input_mistakes()
