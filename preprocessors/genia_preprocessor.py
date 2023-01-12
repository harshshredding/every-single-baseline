from typing import List

from preprocess import Preprocessor
from structs import Anno
from bs4 import BeautifulSoup


def get_first_sample_soup():
    with open('GENIA_term_3.02/GENIAcorpus3.02.xml', 'r') as genia_file:
        soup = BeautifulSoup(genia_file, 'xml')
        return list(soup.find_all('sentence'))[0]


def get_text_from_tag():
    first_sample_soup = get_first_sample_soup()
    print(first_sample_soup)


def get_text(tag):
    ret = ''
    for child in tag.contents:
        if child.name == 'cons':
            ret += get_text(child)
        elif child.string is not None:
            ret += child.string
        else:
            NotImplementedError(f'cannot handle tag {child.name}')
    return ret


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


def get_annos(tag, offset=None, token_strings=None) -> List[Anno]:
    if offset is None:
        offset = [0]
        assert token_strings is None
        token_strings = []
    ret = []
    anno_start_offset = offset[0]
    token_begin_idx = len(token_strings)
    for child in tag.contents:
        if child.name == 'cons':
            ret.extend(get_annos(child, offset, token_strings))
        else:
            offset[0] += len(child.string)
            token_strings.append(child.string)
    anno_end_offset = offset[0]
    if (tag.name == 'cons') and ('sem' in tag.attrs):
        ret.append(
            Anno(
                anno_start_offset,
                anno_end_offset,
                tag['sem'],
                ''.join(token_strings[token_begin_idx:]),
                {}))
    return ret

