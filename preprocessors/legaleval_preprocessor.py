import json
from structs import Annotation, Sample, DatasetSplit, AnnotationCollection, Dataset
from preprocess import Preprocessor
import util
from enum import Enum
from preamble import *
from annotators import Annotator


class LegalSection(Enum):
    PREAMBLE = 0
    JUDGEMENT = 1


class PreprocessLegal(Preprocessor):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator],
            legal_section: LegalSection,
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            annotators=annotators,
            dataset_split=dataset_split,
            dataset=(
                Dataset.legaleval_preamble
                if legal_section == LegalSection.PREAMBLE
                else Dataset.legaleval_judgement
            )
        )
        self.dataset_split = dataset_split
        assert (dataset_split == DatasetSplit.train) or (
                dataset_split == DatasetSplit.valid)
        self.legal_section = legal_section

    def get_data_file_path(self) -> str:
        return f'./legal_raw/{self.dataset_split.name}_{self.legal_section.name}.json'

    def __get_raw_annos(self, sample):
        return sample['annotations'][0]['result']

    def __parse_anno_raw(self, anno_raw) -> Annotation:
        anno_raw = anno_raw['value']
        return Annotation(anno_raw['start'], anno_raw['end'], anno_raw['labels'][0], anno_raw['text'])

    def get_entity_types(self) -> List[str]:
        types_set = set()
        judgement_raw_file_path = f"./legal_raw/{DatasetSplit.train.name}_{LegalSection.JUDGEMENT.name}.json"
        judgement_samples = self.__get_samples(judgement_raw_file_path)
        for sample in judgement_samples:
            types_set.update([anno.label_type for anno in sample.annos.gold])
        preamble_raw_file_path = f"./legal_raw/{DatasetSplit.train.name}_{LegalSection.PREAMBLE.name}.json"
        preamble_samples = self.__get_samples(preamble_raw_file_path)
        for sample in preamble_samples:
            types_set.update([anno.label_type for anno in sample.annos.gold])
        print("num types: ", len(types_set))
        assert len(types_set) == 14
        print(util.pretty_string(list(types_set)))
        return list(types_set)

    def get_samples(self) -> List[Sample]:
        return self.__get_samples(self.get_data_file_path())

    def __get_samples(self, raw_file_path) -> List[Sample]:
        ret = []
        with open(raw_file_path, 'r') as raw_file_handler:
            json_data = json.load(raw_file_handler)
            print("num samples ", len(json_data), raw_file_path)
            for raw_sample in json_data:
                assert 'id' in raw_sample
                sample_id = raw_sample['id']
                sample_text = raw_sample['data']['text']
                assert 'annotations' in raw_sample
                assert len(raw_sample['annotations']) == 1
                assert 'result' in raw_sample['annotations'][0]
                raw_annos = self.__get_raw_annos(raw_sample)
                parsed_annos = [self.__parse_anno_raw(
                    raw_anno) for raw_anno in raw_annos]
                ret.append(Sample(sample_text, sample_id, AnnotationCollection(parsed_annos, [])))
        return ret


class PreprocessLegalJudgement(PreprocessLegal):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator],
    ):
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            annotators=annotators,
            legal_section=LegalSection.JUDGEMENT
        )


class PreprocessLegalPreamble(PreprocessLegal):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator],
    ):
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            annotators=annotators,
            legal_section=LegalSection.PREAMBLE
        )


# def train_judgement():
#     annotators: List[Annotator] = [NounPhraseAnnotator()]
#     train_judgement_preproc = PreprocessLegal(
#         raw_data_folder_path='./legal_raw',
#         entity_type_file_path='./preprocessed_data/legaleval_train_judgement_types.txt',
#         annotations_file_path='./preprocessed_data/legaleval_train_judgement_annos.tsv',
#         visualization_file_path='./preprocessed_data/legaleval_train_judgement_visualisation.bdocjs',
#         tokens_file_path='./preprocessed_data/legaleval_train_judgement_tokens.json',
#         sample_text_file_path="./preprocessed_data/legaleval_train_judgement_sample_text.json",
#         dataset_split=DatasetSplit.train,
#         legal_section=LegalSection.JUDGEMENT,
#         samples_file_path="./preprocessed_data/legaleval_train_judgement_samples.json",
#         annotators=annotators
#     )
#     train_judgement_preproc.run()
#     train_annos_dict = util.get_annos_dict(
#         train_judgement_preproc.annotations_file_path)
#     assert len(train_annos_dict['90d9a97c7b7749ec8a4f460fda6f937e']) == 2
#
#
# def valid_judgement():
#     annotators: List[Annotator] = [NounPhraseAnnotator()]
#     valid_judgement_preproc = PreprocessLegal(
#         raw_data_folder_path='./legal_raw',
#         entity_type_file_path='./preprocessed_data/legaleval_valid_judgement_types.txt',
#         annotations_file_path='./preprocessed_data/legaleval_valid_judgement_annos.tsv',
#         visualization_file_path='./preprocessed_data/legaleval_valid_judgement_visualisation.bdocjs',
#         tokens_file_path='./preprocessed_data/legaleval_valid_judgement_tokens.json',
#         sample_text_file_path="./preprocessed_data/legaleval_valid_judgement_sample_text.json",
#         dataset_split=DatasetSplit.valid,
#         legal_section=LegalSection.JUDGEMENT,
#         samples_file_path="./preprocessed_data/legaleval_valid_judgement_samples.json",
#         annotators=annotators
#     )
#     valid_judgement_preproc.run()
#     # Tests
#     valid_annos_dict = util.get_annos_dict(
#         valid_judgement_preproc.annotations_file_path)
#     valid_token_data = util.get_tokens_from_file(
#         valid_judgement_preproc.tokens_file_path)
#     valid_texts = util.get_texts(valid_judgement_preproc.sample_text_file_path)
#     assert len(valid_annos_dict['03f3901e95ed493b866bd7807f623bc0']) == 3
#     assert len(valid_annos_dict['b0311cba3aac4d909eec6e156c059617']) == 1
#     util.assert_tokens_contain(
#         valid_token_data['03f3901e95ed493b866bd7807f623bc0'],
#         ['Union', 'of', 'India', 'VIII', 'Cooper']
#     )
#     util.assert_tokens_contain(
#         valid_token_data['b0311cba3aac4d909eec6e156c059617'],
#         ['Singh', 'Statutory', 'Principles']
#     )
#     assert ", our Constitution has no 'due process' clause or the VIII Amendment" in valid_texts[
#         '03f3901e95ed493b866bd7807f623bc0']
#
#
# def train_preamble():
#     annotators: List[Annotator] = [NounPhraseAnnotator()]
#     train_preamble_preproc = PreprocessLegal(
#         raw_data_folder_path='./legal_raw',
#         entity_type_file_path='./preprocessed_data/legaleval_train_preamble_types.txt',
#         annotations_file_path='./preprocessed_data/legaleval_train_preamble_annos.tsv',
#         visualization_file_path='./preprocessed_data/legaleval_train_preamble_visualisation.bdocjs',
#         tokens_file_path='./preprocessed_data/legaleval_train_preamble_tokens.json',
#         sample_text_file_path="./preprocessed_data/legaleval_train_preamble_sample_text.json",
#         dataset_split=DatasetSplit.train,
#         legal_section=LegalSection.PREAMBLE,
#         samples_file_path="./preprocessed_data/legaleval_train_preamble_samples.json",
#         annotators=annotators
#     )
#     train_preamble_preproc.run()
#     train_annos_dict = util.get_annos_dict(
#         train_preamble_preproc.annotations_file_path)
#     assert len(train_annos_dict['d79fb7f965a74e418212458285c7c213']) == 8
#
#
# def valid_preamble():
#     annotators: List[Annotator] = [NounPhraseAnnotator()]
#     valid_preamble_preproc = PreprocessLegal(
#         raw_data_folder_path='./legal_raw',
#         entity_type_file_path='./preprocessed_data/legaleval_valid_preamble_types.txt',
#         annotations_file_path='./preprocessed_data/legaleval_valid_preamble_annos.tsv',
#         visualization_file_path='./preprocessed_data/legaleval_valid_preamble_visualisation.bdocjs',
#         tokens_file_path='./preprocessed_data/legaleval_valid_preamble_tokens.json',
#         sample_text_file_path="./preprocessed_data/legaleval_valid_preamble_sample_text.json",
#         dataset_split=DatasetSplit.valid,
#         legal_section=LegalSection.PREAMBLE,
#         samples_file_path="./preprocessed_data/legaleval_valid_preamble_samples.json",
#         annotators=annotators
#     )
#     valid_preamble_preproc.run()
#     valid_annos_dict = util.get_annos_dict(
#         valid_preamble_preproc.annotations_file_path)
#     assert len(valid_annos_dict['1ea9dd0b69b64720851a2234a689082f']) == 13
#
#
# valid_judgement()
# valid_preamble()
# train_judgement()
# train_preamble()

# preproc.create_anno_file(DatasetSplit.train, LegalSection.PREAMBLE)
# preproc.create_anno_file(DatasetSplit.valid, LegalSection.PREAMBLE)
# preproc.create_gate_file(DatasetSplit.train, LegalSection.PREAMBLE)
# preproc.create_gate_file(DatasetSplit.valid, LegalSection.PREAMBLE)

# preproc.create_anno_file(DatasetSplit.train, LegalSection.JUDGEMENT)
# preproc.create_anno_file(DatasetSplit.valid, LegalSection.JUDGEMENT)
# preproc.create_gate_file(DatasetSplit.train, LegalSection.JUDGEMENT)
# preproc.create_gate_file(DatasetSplit.valid, LegalSection.JUDGEMENT)

# preproc.create_types_file()

# preproc.create_input_file(DatasetSplit.valid, LegalSection.PREAMBLE)
# preproc.create_input_file(DatasetSplit.valid, LegalSection.JUDGEMENT)
# preproc.create_input_file(DatasetSplit.train, LegalSection.PREAMBLE)
# preproc.create_input_file(DatasetSplit.train, LegalSection.JUDGEMENT)
