from os.path import dirname, realpath
import sys
import json
root_dir = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root_dir)
from structs import TokenData

def read_raw_data():
    with open(f'{root_dir}/multiconer-data-raw/train_dev/en-dev.conll', 'r') as dev_file:
        samples_dict = {}
        curr_sample_id = None
        for line in list(dev_file.readlines()):
            line = line.strip()
            if len(line):
                if line.startswith('#'):
                    sample_info = line.split()
                    assert len(sample_info) == 4
                    sample_id = sample_info[2]
                    curr_sample_id = sample_id
                else:
                    assert curr_sample_id is not None
                    token_data_split = line.split(" _ _ ")
                    assert len(token_data_split) == 2
                    tokens_list = samples_dict.get(curr_sample_id, [])
                    tokens_list.append((token_data_split[0], token_data_split[1]))
                    samples_dict[curr_sample_id] = tokens_list
        assert len(samples_dict) == 871
        return samples_dict

def create_types_file():
    validation_samples = read_raw_data()
    print(validation_samples['5239d808-f300-46ea-aa3b-5093040213a3'])
    all_labels = set()
    for sample_id in validation_samples:
        tokens_data = validation_samples[sample_id]
        labels = [label for token_string, label in tokens_data]
        all_labels.update(labels)
    print("num all labels", len(all_labels))
    top_level_labels = [label[2:] for label in all_labels if len(label) > 2]
    print("num all top level lables", len(top_level_labels))
    assert len(top_level_labels) == (len(all_labels) - 1)
    top_level_labels_set = set(top_level_labels)
    assert len(top_level_labels_set) == (len(top_level_labels)//2)
    print("num top level label set", len(top_level_labels_set))
    with open('./datasets/multiconer/types.txt', 'w') as types_file:
        for label in top_level_labels_set:
            print(label, file=types_file)

def create_input_file():
    all_tokens_json = []
    sample_data = read_raw_data()
    for sample_id in sample_data:
        token_data_list = sample_data[sample_id]
        token_offset = 0
        for (token_string, token_label) in token_data_list:
            token_json = {'Token': [{"string": token_string, "startOffset": token_offset,
                                     "endOffset": token_offset + len(token_string), "length": len(token_string)}],
                          'Sample': [{"id": sample_id, "startOffset": 0}],
                          'Span': [{"type": token_label, "id": None}]
                          }
            all_tokens_json.append(token_json)
            token_offset += len(token_string) 
    with open('./datasets/multiconer/input-files/dev.json', 'w') as output_file:
        json.dump(all_tokens_json, output_file, indent=4)

def create_annos_file():
    sample_data = read_raw_data()
    annos_dict = {}
    for sample_id in sample_data:
        token_data_list = sample_data[sample_id]
        token_offset = 0
        curr_span_start, curr_span_type, curr_span_text = None, None, None
        spans = []
        for token_string, token_label in token_data_list:
            if token_label == 'O' or token_label.startswith('B-'):
                if curr_span_start is not None:
                    spans.append((curr_span_start, token_offset, curr_span_type, curr_span_text))
                    curr_span_start, curr_span_type, curr_span_text = None, None, None
            if token_label.startswith("B-"):
                curr_span_start = token_offset
                curr_span_type = token_label[2:]
                curr_span_text = token_string
            elif token_label.startswith("I-"):
                curr_span_text = " ".join([curr_span_text, token_string])
            token_offset += len(token_string)
        if curr_span_start is not None:
            spans.append((curr_span_start, token_offset, curr_span_type, curr_span_text))
            curr_span_start, curr_span_type, curr_span_text = None, None, None
        annos_dict[sample_id] = spans
    print(annos_dict['5239d808-f300-46ea-aa3b-5093040213a3'])
    print(annos_dict['d7d47dfc-7e5d-48e8-9390-019a3e9476c1'])
    print(annos_dict['8a8e516d-e4ba-42e3-bf62-f2994db69d55'])


#create_types_file()
#create_input_file()
create_annos_file()