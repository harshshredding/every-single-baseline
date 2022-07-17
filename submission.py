from util import *
from transformers import AutoTokenizer
from read_gate_output import *
from args import args
from args import device
import time
sample_to_token_data_test = get_test_data(args['test_data_folder_path'])
model_path = ''
model = prepare_model()
model.load_state_dict(torch.load(model_path))
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
raw_test_data = get_raw_test_data()
# Validation starts
model.eval()
with open('test_predictions.tsv', 'w') as predictions_file:
    predictions_file.write('\t'.join(['tweets_id', 'begin', 'end', 'type', 'extraction']))
    with torch.no_grad():
        validation_start_time = time.time()
        sample_ids = list(sample_to_token_data_test.keys())
        if args['testing_mode']:
            sample_ids = sample_ids[:10]
        for sample_id in sample_ids:
            raw_text = raw_test_data[sample_id]
            sample_data = sample_to_token_data_test[sample_id]
            tokens = get_token_strings(sample_data)
            offsets_list = get_token_offsets(sample_data)
            batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
            model_input = prepare_model_input(batch_encoding, sample_data)
            output = model(*model_input)
            pred_labels_expanded = torch.argmax(output, dim=1).cpu().detach().numpy()
            pred_spans_token_index = get_spans_from_seq_labels(pred_labels_expanded, batch_encoding)
            pred_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                                       pred_spans_token_index]
            pred_spans_set = set(pred_spans_char_offsets)
            for span in pred_spans_set:
                start_offset = span[0]
                end_offset = span[1]
                extraction = raw_text[start_offset: end_offset]
                predictions_file.write(
                    '\n' + '\t'.join([sample_id, str(start_offset), str(end_offset), 'ENFERMEDAD', extraction]))