from typing import List, Tuple

from preprocess import Preprocessor
from structs import Anno, Sample, DatasetSplit
from bs4 import BeautifulSoup
import util


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
    if (tag.name == 'cons') and ('sem' in tag.attrs) and (tag['sem'][0:2] == 'G#'):
        ret.append(
            Anno(
                anno_start_offset,
                anno_end_offset,
                tag['sem'],
                ''.join(token_strings[token_begin_idx:]),
                {}))
    return ret


def get_parent_label_from_anno(anno: Anno) -> str | None:
    anno_label_type = anno.label_type.lower()
    if anno_label_type.startswith('G#DNA'.lower()):
        return 'dna'
    elif anno_label_type.startswith('G#protein'.lower()):
        return 'protein'
    elif anno_label_type.startswith('G#RNA'.lower()):
        return 'rna'
    elif anno_label_type.startswith('G#cell_line'.lower()):
        return 'cell_line'
    elif anno_label_type.startswith('G#cell_type'.lower()):
        return 'cell_type'
    else:
        return None


def get_parent_annos(anno_list: List[Anno]) -> List[Anno]:
    ret = []
    for anno in anno_list:
        if get_parent_label_from_anno(anno) is not None:
            anno.label_type = get_parent_label_from_anno(anno)
            ret.append(anno)
    return ret


def get_split_range(split: DatasetSplit) -> Tuple:
    if split == DatasetSplit.train:
        return 0, 9273
    if split == DatasetSplit.valid:
        return 9273, 13909
    if split == DatasetSplit.test:
        return 13909, 18546
    raise Exception("should not reach here")


def get_samples(split: DatasetSplit) -> List[Sample]:
    split_range = get_split_range(split)
    with open('GENIA_term_3.02/GENIAcorpus3.02.xml', 'r') as genia_file:
        soup = BeautifulSoup(genia_file, 'xml')
        ret = []
        for sample_id, sent_tag in enumerate(soup.find_all('sentence')):
            if split_range[0] <= sample_id < split_range[1]:
                sample_text = get_text(sent_tag)
                sample_annos = get_parent_annos(get_annos(sent_tag))
                ret.append(Sample(sample_text, str(sample_id), sample_annos))
        return ret


class PreprocessGenia(Preprocessor):

    def __init__(
            self,
            raw_data_folder_path: str,
            entity_type_file_path: str,
            annotations_file_path: str,
            visualization_file_path: str,
            tokens_file_path: str,
            sample_text_file_path: str,
            dataset_split: DatasetSplit,
    ) -> None:
        super().__init__(
            raw_data_folder_path,
            entity_type_file_path,
            annotations_file_path,
            visualization_file_path,
            tokens_file_path,
            sample_text_file_path
        )
        self.dataset_split = dataset_split

    def get_samples(self) -> List[Sample]:
        return get_samples(self.dataset_split)

    def create_entity_types_file(self) -> None:
        all_types_set = set()
        for sample in self.get_samples():
            all_types_set.update([anno.label_type for anno in sample.annos])
        assert len(all_types_set) == 5
        with util.open_make_dirs(self.entity_type_file_path, 'w') as types_file:
            for type_name in all_types_set:
                print(type_name, file=types_file)


if __name__ == '__main__':
    genia_preproc = PreprocessGenia(
        raw_data_folder_path='./GENIA_term_3.02',
        entity_type_file_path='./preprocessed_data/genia_valid_types.txt',
        annotations_file_path='./preprocessed_data/genia_valid_annos.tsv',
        visualization_file_path='./preprocessed_data/genia_valid_visualisation.bdocjs',
        tokens_file_path='./preprocessed_data/genia_valid_tokens.json',
        sample_text_file_path="./preprocessed_data/genia_valid_text.json",
        dataset_split=DatasetSplit.valid,
    )
    genia_preproc.run()

    genia_preproc = PreprocessGenia(
        raw_data_folder_path='./GENIA_term_3.02',
        entity_type_file_path='./preprocessed_data/genia_train_types.txt',
        annotations_file_path='./preprocessed_data/genia_train_annos.tsv',
        visualization_file_path='./preprocessed_data/genia_train_visualisation.bdocjs',
        tokens_file_path='./preprocessed_data/genia_train_tokens.json',
        sample_text_file_path="./preprocessed_data/genia_train_text.json",
        dataset_split=DatasetSplit.train,
    )
    genia_preproc.run()
