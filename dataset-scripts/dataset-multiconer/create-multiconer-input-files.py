from pathlib import Path
import json
import csv
from structs import Anno
import util
from collections import Counter

coarse_to_fine = {
     'Coarse_Location':['Facility', 'OtherLOC', 'HumanSettlement', 'Station'],
     'Coarse_Creative_Work': ['VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software', 'OtherCW'],
     'Coarse_Group': ['MusicalGRP', 'PublicCorp', 'PrivateCorp', 'OtherCorp', 'AerospaceManufacturer', 'SportsGRP', 'CarManufacturer', 'TechCorp', 'ORG'],
     'Coarse_Person': ['Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric', 'SportsManager', 'OtherPER'],
     'Coarse_Product': ['Clothing', 'Vehicle', 'Food', 'Drink', 'OtherPROD'],
     'Coarse_Medical': ['Medication/Vaccine', 'MedicalProcedure', 'AnatomicalStructure', 'Symptom', 'Disease']
}

def get_all_fine_grained_labels():
    ret = []
    for coarse_label in coarse_to_fine:
        for fine_label in coarse_to_fine[coarse_label]:
            ret.append(fine_label)
    return ret

def get_fine_to_coarse_dict():
    ret = {}
    for coarse in coarse_to_fine:
        for fine in coarse_to_fine[coarse]:
            ret[fine] = coarse
    return ret

def read_raw_data(train=True):
    with open(f"./multiconer-data-raw/train_dev/en-{'train' if train else 'dev'}.conll", 'r') as dev_file:
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
        assert len(samples_dict) == (16767 if train else 871), len(samples_dict)
        return samples_dict

def create_types_file():
    samples = read_raw_data()
    #print(samples['5239d808-f300-46ea-aa3b-5093040213a3'])
    fine_labels_set = set()
    fine_label_occurences = []
    for sample_id in samples:
        tokens_data = samples[sample_id]
        fine_labels = [label[2:] for _, label in tokens_data if len(label) > 2]
        fine_labels_set.update(fine_labels)
        fine_label_occurences.extend(fine_labels)
    predefined_fine_grain_labels = set(get_all_fine_grained_labels())
    assert predefined_fine_grain_labels.difference(fine_labels_set) == {'TechCorp', 'OtherCW', 'OtherCorp'}
    fine_label_occurence_count = Counter(fine_label_occurences)
    print("top level occurence count")
    print(json.dumps(fine_label_occurence_count,indent=4))
    with open('./datasets/multiconer/info.txt', 'w') as info_file:
        print("top level occurence count", file=info_file)
        print(json.dumps(fine_label_occurence_count,indent=4), file=info_file)
    print("num fine labels", len(fine_labels_set)) 
    Path("./datasets/multiconer").mkdir(parents=True, exist_ok=True)
    with open('./datasets/multiconer/types.txt', 'w') as types_file:
        for fine_label in fine_labels_set:
            print(fine_label, file=types_file)

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
            token_offset += (len(token_string) + 1) # add one for one space between tokens
    
    Path("./datasets/multiconer/input-files").mkdir(parents=True, exist_ok=True)
    with open('./datasets/multiconer/input-files/train.json', 'w') as output_file:
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
                    spans.append(Anno(curr_span_start, token_offset - 1, curr_span_type, curr_span_text))
                    curr_span_start, curr_span_type, curr_span_text = None, None, None
            if token_label.startswith("B-"):
                curr_span_start = token_offset
                curr_span_type = token_label[2:]
                curr_span_text = token_string
            elif token_label.startswith("I-"):
                curr_span_text = " ".join([curr_span_text, token_string])
            token_offset += (len(token_string) + 1) # add one for one space between tokens 
        if curr_span_start is not None:
            spans.append(Anno(curr_span_start, token_offset - 1, curr_span_type, curr_span_text))
            curr_span_start, curr_span_type, curr_span_text = None, None, None
        annos_dict[sample_id] = spans

    Path("./datasets/multiconer/gold-annos").mkdir(parents=True, exist_ok=True)
    with open('./datasets/multiconer/gold-annos/annos.tsv', 'w') as annos_file:
        writer = csv.writer(annos_file, delimiter='\t')
        header = ['sample_id', 'begin', 'end', 'type', 'extraction']
        writer.writerow(header)
        for sample_id in annos_dict:
            for anno in annos_dict[sample_id]:
                row = [sample_id, anno.begin_offset, anno.end_offset, anno.label_type, anno.extraction]
                writer.writerow(row)

    #print(annos_dict['5239d808-f300-46ea-aa3b-5093040213a3'])
    #print(annos_dict['d7d47dfc-7e5d-48e8-9390-019a3e9476c1'])
    #print(annos_dict['8a8e516d-e4ba-42e3-bf62-f2994db69d55'])

def create_gate_input_file():
    sample_to_token_data = util.get_train_data()
    annos_dict = util.get_train_annos_dict()
    fine_to_coarse_dict = get_fine_to_coarse_dict()
    for sample_id in annos_dict:
        gold_annos = annos_dict[sample_id]
        coarse_annos = [Anno(anno.begin_offset, anno.end_offset, fine_to_coarse_dict[anno.label_type], anno.extraction) for anno in gold_annos]
        gold_annos.extend(coarse_annos)
    Path("./datasets/multiconer/gate-input").mkdir(parents=True, exist_ok=True)
    util.create_gate_file("multiconer", sample_to_token_data, annos_dict, 10000)

create_types_file()
create_input_file()
create_annos_file()
create_gate_input_file()