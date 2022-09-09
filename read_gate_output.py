import json
from args import *
import os
from typing import List, Dict
from structs import *


def parse_token_data(token_data_raw):
    """
    in: a json object representing a token
    out: a TokenData object parsed from the input json object
    """
    if curr_dataset == Dataset.few_nerd:
        return TokenData(
            token_data_raw['Sample'][0]['id'],
            token_data_raw['Sample'][0]['startOffset'],
            token_data_raw['Token'][0]['string'],
            token_data_raw['Token'][0]['length'],
            token_data_raw['Token'][0]['startOffset'],
            token_data_raw['Token'][0]['endOffset'],
            token_data_raw['Span'][0]['type'] if 'Span' in token_data_raw else None
        )
    else:
        raise NotImplementedError('implement parsing for social-dis-ner data')


def read_data_from_folder(data_folder) -> Dict[str, List[TokenData]]:
    sample_to_tokens = {}
    data_files_list = os.listdir(data_folder)
    for filename in data_files_list:
        data_file_path = os.path.join(data_folder, filename)
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        for token_data in data:
            parsed_token_data = parse_token_data(token_data)
            sample_id = str(parsed_token_data.sample_id)
            sample_tokens = sample_to_tokens.get(sample_id, [])
            sample_tokens.append(parsed_token_data)
            sample_to_tokens[sample_id] = sample_tokens
        return sample_to_tokens


def get_train_data() -> Dict[str, List[TokenData]]:
    return read_data_from_folder(args['training_data_folder_path'])


def get_valid_data() -> Dict[str, List[TokenData]]:
    return read_data_from_folder(args['validation_data_folder_path'])


def get_test_data() -> Dict[str, List[TokenData]]:
    return read_data_from_folder(args['test_data_folder_path'])


def get_token_strings(sample_data: List[TokenData]):
    only_token_strings = []
    for token_data in sample_data:
        only_token_strings.append(token_data.token_string)
    return only_token_strings


def get_labels(sample_data: List[TokenData]):
    tags = []
    for token_data in sample_data:
        if token_data.label is not None:
            tags.append(token_data.label)
        else:
            tags.append(OUTSIDE_LABEL_STRING)
    return tags


def get_labels_rich(sample_data: List[TokenData], annos: List[Anno]) -> List[Label]:
    labels = get_labels(sample_data)
    offsets = get_token_offsets(sample_data)
    new_labels = []
    for (label, curr_offset) in zip(labels, offsets):
        if label != OUTSIDE_LABEL_STRING:
            anno_same_start = [anno for anno in annos if anno.begin_offset == curr_offset[0]]
            in_anno = [anno for anno in annos if
                       (curr_offset[0] >= anno.begin_offset) and (curr_offset[1] <= anno.end_offset)]
            if len(anno_same_start) > 0:
                new_labels.append(Label(label, BioTag.begin))
            else:
                # avoid DiseaseMid without a DiseaseStart
                if (len(new_labels) > 0) and (new_labels[-1].bio_tag != BioTag.out) \
                        and (label == new_labels[-1].label_type) and len(in_anno):
                    new_labels.append(Label(label, BioTag.inside))
                else:
                    new_labels.append(Label.get_outside_label())
        else:
            new_labels.append(Label.get_outside_label())
    return new_labels


def get_token_offsets(sample_data: List[TokenData]):
    offsets_list = []
    for token_data in sample_data:
        sample_start = token_data.sample_start_offset
        offsets_list.append((token_data.token_start_offset - sample_start,
                             token_data.token_end_offset - sample_start))
    return offsets_list


def get_umls_data(sample_data):
    raise NotImplementedError()
    # umls_tags = []
    # for token_data in sample_data:
    #     if 'UMLS' in token_data:
    #         umls_tags.append(token_data['UMLS'])
    #     else:
    #         umls_tags.append('o')
    # return umls_tags


def get_dis_gaz_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'DisGaz' in token_data:
    #         output.append('DisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_dis_gaz_one_hot(sample_data):
    # dis_labels = get_dis_gaz_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_umls_diz_gaz_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'UMLS_Disease' in token_data:
    #         output.append('UmlsDisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_umls_dis_gaz_one_hot(sample_data):
    # dis_labels = get_umls_diz_gaz_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_silver_dis_one_hot(sample_data):
    # dis_labels = get_silver_dis_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_silver_dis_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'SilverDisGaz' in token_data:
    #         output.append('SilverDisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_pos_data(sample_data):
    # pos_tags = []
    # for token_data in sample_data:
    #     pos_tags.append(token_data['Token'][0]['category'])
    # return pos_tags
    raise NotImplementedError()


def get_umls_indices(sample_data, umls_key_to_index):
    # umls_data = get_umls_data(sample_data)
    # umls_keys = [default_key if umls == 'o' else umls[0]['CUI'] for umls in umls_data]
    # default_index = umls_key_to_index[default_key]
    # umls_indices = [umls_key_to_index.get(key, default_index) for key in umls_keys]
    # return umls_indices
    raise NotImplementedError()


def get_pos_indices(sample_data, pos_key_to_index):
    # pos_tags = get_pos_data(sample_data)
    # default_index = pos_key_to_index[default_key]
    # pos_indices = [pos_key_to_index.get(tag, default_index) for tag in pos_tags]
    # return pos_indices
    raise NotImplementedError()
