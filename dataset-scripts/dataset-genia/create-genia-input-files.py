from bs4 import BeautifulSoup
import csv
from dataclasses import dataclass
from typing import List
import json


@dataclass
class Label:
    string: str
    id: str


@dataclass
class Token:
    string: str
    start: int
    end: int
    labels: List[Label]
    sample_id: str


def get_token_data_from_tag(bs_tag, sample_id, offset, label_id) -> List[Token]:
    ret = []
    tag_label = bs_tag['sem'] if (bs_tag.name == 'cons' and ('sem' in bs_tag.attrs)) else None
    for child in bs_tag.contents:
        if child.name == 'cons':
            ret.extend(get_token_data_from_tag(child, sample_id, offset, label_id))
        elif child.string is not None:
            start_offset = offset[0]
            end_offset = start_offset + len(child.string)
            ret.append(Token(child.string, start_offset, end_offset, [], sample_id=sample_id))
            offset[0] = end_offset
        else:
            NotImplementedError(f'cannot handle tag {child.name}')
    if (tag_label is not None) and (get_parent_label(tag_label) is not None):
        for token in ret:
            token.labels.insert(0, Label(string=get_parent_label(tag_label), id=str(label_id[0])))
        label_id[0] += 1
    return ret


def get_parent_label(label: str) -> str | None:
    if label.startswith('G#DNA'):
        return 'dna'
    elif label.startswith('G#protein'):
        return 'protein'
    elif label.startswith('G#RNA'):
        return 'rna'
    elif label.startswith('G#cell_line'):
        return 'cell_line'
    elif label.startswith('G#cell_type'):
        return 'cell_type'
    else:
        return None


def create_training_data(soup):
    all_tokens_json = []
    for sent_id, sent in enumerate(soup.find_all('sentence')):
        for token in get_token_data_from_tag(sent, sample_id=sent_id, label_id=[0], offset=[0]):
            token_json = {'Token': [{"string": token.string, "startOffset": token.start,
                                     "endOffset": token.end, "length": len(token.string)}],
                          'Sample': [{"id": token.sample_id, "startOffset": 0}],
                          'Span': [{"type": label.string, "id": label.id} for label in token.labels]
                          }
            all_tokens_json.append(token_json)
    with open('../../datasets/genia-dataset/input-files/train.json', 'w') as output_file:
        json.dump(all_tokens_json, output_file)


def get_labels_from_tag(bs_tag, offset=None):
    if offset is None:
        offset = [0]
    ret = []
    label_start = offset[0]
    for child in bs_tag.contents:
        if child.name == 'cons':
            ret.extend(get_labels_from_tag(child, offset))
        else:
            offset[0] += len(child.string)
    label_end = offset[0]
    if (bs_tag.name == 'cons') and ('sem' in bs_tag.attrs):
        ret.append((bs_tag['sem'], label_start, label_end))
    return ret


def move_offset(bs_tag, offset):
    assert offset is not None
    for child in bs_tag.contents:
        if child.name == 'cons':
            move_offset(child, offset)
        else:
            offset[0] += len(child.string)


def get_surface_labels_from_sentence(sentence_tag, offset=None):
    if offset is None:
        offset = [0]
    ret = []
    for child in sentence_tag.contents:
        if child.name == 'cons':
            start_offset = offset[0]
            move_offset(child, offset)
            end_offset = offset[0]
            ret.append((child['sem'], start_offset, end_offset))
        else:
            offset[0] += len(child.string)
    return ret


def extract_non_complex(complex_label, non_complex_set):
    for non_complex in non_complex_set:
        if non_complex in complex_label:
            return non_complex
    raise Exception(f'no non-complex label found in given complex {complex_label}')


def extract_tokens(tag, start, end, offset=None):
    if offset is None:
        offset = [0]
    ret = []
    for child in tag.contents:
        if child.name == 'cons':
            ret.extend(extract_tokens(child, start, end, offset))
        elif child.string is not None:
            token_start = offset[0]
            token_end = token_start + len(child.string)
            if (token_start >= start) and (token_end <= end):
                ret.append(child.string)
            offset[0] = token_end
        else:
            NotImplementedError(f'cannot handle tag {child.name}')
    return ret


def create_gold_annos_file(soup, complex_set, non_complex_set, surface_labels_only=True):
    with open('../../datasets/genia-dataset/gold-annos/annos.tsv', 'w') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        header = ['sample_id', 'begin', 'end', 'type', 'extraction']
        writer.writerow(header)
        for sent_id, sent in enumerate(soup.find_all('sentence')):
            if surface_labels_only:
                spans = get_surface_labels_from_sentence(sent)
            else:
                spans = get_labels_from_tag(sent)
            spans_non_complex = []
            for label_name, start, end in spans:
                if label_name in complex_set:
                    non_complex_label_name = extract_non_complex(label_name, non_complex_set)
                    spans_non_complex.append((non_complex_label_name, start, end))
                else:
                    spans_non_complex.append((label_name, start, end))
            assert len(spans) == len(spans_non_complex)
            spans_cleaned = [(get_parent_label(label), start, end) for (label, start, end) in spans_non_complex
                             if get_parent_label(label) is not None]
            for span in spans_cleaned:
                row = [sent_id, span[1], span[2], span[0], ' '.join(extract_tokens(sent, span[1], span[2]))]
                writer.writerow(row)


with open('../../GENIA_term_3.02/GENIAcorpus3.02.xml', 'r') as genia_file:
    soup = BeautifulSoup(genia_file, 'xml')
    labels_set = set()
    print("num abstracts", len(soup.find_all('abstract')))
    print("num sentences", len(soup.find_all('sentence')))
    label_count = 0
    non_complex_count = 0
    complex_count = 0
    non_complex_set = set()
    complex_set = set()
    for sent in soup.find_all('sentence'):
        label_string_list = [label[0] for label in get_labels_from_tag(sent)]
        label_count += len(label_string_list)
        non_complex_list = [label_string for label_string in label_string_list
                            if (label_string[0:2] == 'G#')]
        non_complex_set.update(non_complex_list)
        non_complex_count += len(non_complex_list)
        complex_list = [label_string for label_string in label_string_list if label_string[0:2] != 'G#']
        complex_count = complex_count + len(complex_list)
        complex_set.update(complex_list)
    print("num labels", label_count)
    print("num non-complex count", non_complex_count)
    print("num complex count", complex_count)
    print("non-complex set", non_complex_set)
    print("non-complex set size", len(non_complex_set))
    print("complex set", complex_set)
    print("complex set size", len(complex_set))
    for complex_type in complex_set:
        found = False
        count = 0
        for non_complex_type in non_complex_set:
            if non_complex_type in complex_type:
                found = True
                count += 1
        assert found and (count == 1)
    # Creating annotation file
    create_gold_annos_file(soup=soup, complex_set=complex_set, non_complex_set=non_complex_set)
    # Create Token data
    create_training_data(soup=soup)
