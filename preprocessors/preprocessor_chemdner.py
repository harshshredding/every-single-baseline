from bs4 import BeautifulSoup
from preamble import *
from structs import Anno, Sample, AnnotationCollection, DatasetSplit, Dataset
from preprocess import Preprocessor, PreprocessorRunType
from annotators import Annotator


def get_sample_text_from_passage(passage_soup: BeautifulSoup) -> str:
    children_names = [child.name for child in passage_soup.children]
    assert 'text' in children_names
    sample_text = passage_soup.find('text').text
    return sample_text


def get_annotation_type(anno_soup: BeautifulSoup) -> str:
    type_info = [info for info in anno_soup.find_all('infon') if info['key'] == 'class'][0]
    return type_info.text


def get_annos_from_passage(passage_soup: BeautifulSoup, passage_offset: int) -> List[Anno]:
    ret = []
    anno_soups = passage_soup.find_all('annotation')
    for anno_soup in anno_soups:
        anno_type = get_annotation_type(anno_soup)
        anno_start = int(anno_soup.location['offset'])
        anno_end = int(anno_soup.location['length']) + anno_start
        anno_start -= passage_offset
        anno_end -= passage_offset
        anno_text = anno_soup.find('text').text
        ret.append(
            Anno(
                begin_offset=anno_start,
                end_offset=anno_end,
                label_type=anno_type,
                extraction=anno_text
            )
        )
    return ret


def get_samples_from_bioc_file(bioc_xml_file_path: str) -> List[Sample]:
    with open(bioc_xml_file_path, 'r') as cdr_xml_file:
        cdr_raw_xml_data = cdr_xml_file.read()
    cdr_soup = BeautifulSoup(cdr_raw_xml_data, features='xml')
    all_documents = cdr_soup.find_all('document')
    ret = []
    for cdr_document in all_documents:
        cdr_document_id = cdr_document.id.text
        for passage_idx, passage_soup in enumerate(cdr_document.find_all('passage')):
            sample_text = get_sample_text_from_passage(passage_soup)
            passage_offset = int(passage_soup.offset.text)
            gold_annos = get_annos_from_passage(passage_soup, passage_offset)
            ret.append(
                Sample(
                    text=sample_text,
                    annos=AnnotationCollection(gold=gold_annos, external=[]),
                    id=(cdr_document_id + str(passage_idx))
                )
            )
    return ret


def get_samples(dataset_split: DatasetSplit) -> List[Sample]:
    match dataset_split:
        case DatasetSplit.valid:
            raw_data_file_path = 'chemdner_corpus/development.bioc.xml'
        case DatasetSplit.train:
            raw_data_file_path = 'chemdner_corpus/training.bioc.xml'
        case _:
            raise RuntimeError("split not supported")
    return get_samples_from_bioc_file(raw_data_file_path)


class PreprocessChemD(Preprocessor):
    """
    A preprocessor for the ChemD dataset.
    """

    def __init__(
            self,
            dataset_split: DatasetSplit,
            preprocessor_type: str,
            dataset: Dataset,
            annotators: List[Annotator] = [],
            run_mode: PreprocessorRunType = PreprocessorRunType.production
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset=dataset,
            annotators=annotators,
            dataset_split=dataset_split,
            run_mode=run_mode,
        )

    def get_samples(self) -> List[Sample]:
        return get_samples(self.dataset_split)

    def get_entity_types(self) -> List[str]:
        all_types_set = set()
        for sample in self.get_samples():
            all_types_set.update([anno.label_type for anno in sample.annos.gold])
        assert len(all_types_set) == 8, f"all_types: {all_types_set}"
        return list(all_types_set)
