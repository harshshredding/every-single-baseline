import json
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset, AnnotationCollection
from preprocess import Preprocessor
from typing import List, Dict
import util
from enum import Enum
from collections import Counter
import spacy
from utils.universal import die
import benepar
from annotators import NounPhraseAnnotator, Annotator
benepar.download('benepar_en3')

class Granularity(Enum):
    coarse = 0
    fine = 1


class PreprocessMulticoner(Preprocessor):
    def __init__(
            self,
            raw_data_folder_path: str,
            entity_type_file_path: str,
            annotations_file_path: str,
            visualization_file_path: str,
            tokens_file_path: str,
            sample_text_file_path: str,
            samples_file_path: str,
            dataset_split: DatasetSplit,
            granularity: Granularity,
            annotators: List[Annotator]
    ) -> None:
        super().__init__(
            raw_data_folder_path=raw_data_folder_path,
            entity_type_file_path=entity_type_file_path,
            annotations_file_path=annotations_file_path,
            visualization_file_path=visualization_file_path,
            tokens_file_path=tokens_file_path,
            sample_text_file_path=sample_text_file_path,
            samples_file_path=samples_file_path,
            annotators=annotators
        )
        self.dataset_split = dataset_split
        assert (dataset_split == DatasetSplit.train) or (dataset_split == DatasetSplit.valid)
        self.granularity = granularity
        self.coarse_to_fine = {
            'Coarse_Location': ['Facility', 'OtherLOC', 'HumanSettlement', 'Station'],
            'Coarse_Creative_Work': ['VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software', 'OtherCW'],
            'Coarse_Group': ['MusicalGRP', 'PublicCorp', 'PrivateCorp', 'OtherCorp', 'AerospaceManufacturer',
                             'SportsGRP', 'CarManufacturer', 'TechCorp', 'ORG'],
            'Coarse_Person': ['Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric', 'SportsManager', 'OtherPER'],
            'Coarse_Product': ['Clothing', 'Vehicle', 'Food', 'Drink', 'OtherPROD'],
            'Coarse_Medical': ['Medication/Vaccine', 'MedicalProcedure', 'AnatomicalStructure', 'Symptom', 'Disease'],
            'O': ['O']
        }
        # benepar constituency parsing spacy pipeline
        nlp_benepar = spacy.load('en_core_web_md')
        nlp_benepar.add_pipe("benepar", config={"model": "benepar_en3"})
        self.nlp_benepar = nlp_benepar

    def __get_all_fine_grained_labels(self):
        ret = []
        for coarse_label in self.coarse_to_fine:
            for fine_label in self.coarse_to_fine[coarse_label]:
                ret.append(fine_label)
        return ret

    def create_entity_types_file(self) -> None:
        samples = self.__get_samples(f"{self.raw_data_folder_path}/en-train.conll")
        train_labels_set = set()
        train_labels_occurences = []
        for sample in samples:
            labels_list = [anno.label_type for anno in sample.annos.gold]
            train_labels_set.update(labels_list)
            train_labels_occurences.extend(labels_list)
        predefined_labels = set(self.__get_all_fine_grained_labels()) \
            if self.granularity == Granularity.fine \
            else set(list(self.coarse_to_fine.keys()))
        if self.granularity == Granularity.fine:
            assert predefined_labels.difference(train_labels_set) == {'TechCorp', 'OtherCW', 'OtherCorp', 'O'}
        else:
            assert predefined_labels.difference(train_labels_set) == {'O'}
        label_occurence_count = Counter(train_labels_occurences)
        print("top level occurence count")
        print(json.dumps(label_occurence_count, indent=4))
        print("num fine labels", len(train_labels_set))
        with util.open_make_dirs(self.entity_type_file_path, 'w') as types_file:
            for fine_label in train_labels_set:
                print(fine_label, file=types_file)

    def get_samples(self) -> List[Sample]:
        if self.dataset_split == DatasetSplit.valid:
            return self.__get_samples(f"{self.raw_data_folder_path}/en-dev.conll")
        elif self.dataset_split == DatasetSplit.train:
            return self.__get_samples(f"{self.raw_data_folder_path}/en-train.conll")
        else:
            die("Cannot handle a dataset other than train or valid")
            

    def __get_text(self, tokens: List[tuple]) -> str:
        return ' '.join([token_string for token_string, _ in tokens])

    def __read_raw_data(self, raw_file_path: str) -> Dict[SampleId, List[tuple]]:
        with open(raw_file_path, 'r') as dev_file:
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
                        token_string, token_label = line.split(" _ _ ")
                        tokens_list = samples_dict.get(curr_sample_id, [])
                        tokens_list.append((token_string, token_label))
                        samples_dict[curr_sample_id] = tokens_list
            return samples_dict

    def __remove_bio(self, label_string):
        return label_string[2:] if len(label_string) > 2 else label_string

    def __get_fine_to_coarse_dict(self):
        ret = {}
        for coarse in self.coarse_to_fine:
            for fine in self.coarse_to_fine[coarse]:
                ret[fine] = coarse
        return ret

    def __build_annos_dict(self, sample_to_tokens) -> Dict[SampleId, List[Anno]]:
        fine_to_coarse = self.__get_fine_to_coarse_dict()
        annos_dict = {}
        for sample_id in sample_to_tokens:
            token_data_list = sample_to_tokens[sample_id]
            token_offset = 0
            curr_span_start, curr_span_type, curr_span_text = None, None, None
            spans = []
            for token_string, token_label in token_data_list:
                if token_label == 'O' or token_label.startswith('B-'):
                    if curr_span_start is not None:
                        assert curr_span_type is not None
                        assert curr_span_text is not None
                        spans.append(Anno(curr_span_start, token_offset - 1, curr_span_type, curr_span_text))
                        curr_span_start, curr_span_type, curr_span_text = None, None, None
                if token_label.startswith("B-"):
                    curr_span_start = token_offset
                    curr_span_type = self.__remove_bio(token_label) if self.granularity == Granularity.fine \
                        else fine_to_coarse[self.__remove_bio(token_label)]
                    curr_span_text = token_string
                elif token_label.startswith("I-"):
                    assert curr_span_text is not None
                    curr_span_text = " ".join([curr_span_text, token_string])
                token_offset += (len(token_string) + 1)  # add one for one space between tokens
            if curr_span_start is not None:
                assert curr_span_type is not None
                assert curr_span_text is not None
                spans.append(Anno(curr_span_start, token_offset - 1, curr_span_type, curr_span_text))
                curr_span_start, curr_span_type, curr_span_text = None, None, None
            annos_dict[sample_id] = spans
        return annos_dict

    def __get_samples(self, raw_file_path) -> List[Sample]:
        ret: List[Sample] = []
        sample_to_tokens = self.__read_raw_data(raw_file_path)
        annos_dict = self.__build_annos_dict(sample_to_tokens)
        for sample_id in sample_to_tokens:
            tokens = sample_to_tokens[sample_id]
            sample_text = self.__get_text(tokens)
            sample_gold_annos = annos_dict.get(sample_id, [])
            ret.append(Sample(sample_text, sample_id, AnnotationCollection(sample_gold_annos, [])))
        return ret


# prefix = f"{Dataset.multiconer.name}_{DatasetSplit.train.name}_{Granularity.coarse.name}"
# train_coarse_preproc = PreprocessMulticoner(
#     raw_data_folder_path='./multiconer-data-raw/train_dev',
#     entity_type_file_path=f'./preprocessed_data/{prefix}_types.txt',
#     annotations_file_path=f'./preprocessed_data/{prefix}_annos.tsv',
#     visualization_file_path=f'./preprocessed_data/{prefix}_visualization.bdocjs',
#     tokens_file_path=f'./preprocessed_data/{prefix}_tokens.json',
#     sample_text_file_path=f"./preprocessed_data/{prefix}_sample_text.json",
#     dataset_split=DatasetSplit.train,
#     granularity=Granularity.coarse
# )
# train_coarse_preproc.run()

# prefix = f"{Dataset.multiconer.name}_{DatasetSplit.train.name}_{Granularity.fine.name}"
# train_fine_preproc = PreprocessMulticoner(
#     raw_data_folder_path='./multiconer-data-raw/train_dev',
#     entity_type_file_path=f'./preprocessed_data/{prefix}_types.txt',
#     annotations_file_path=f'./preprocessed_data/{prefix}_annos.tsv',
#     visualization_file_path=f'./preprocessed_data/{prefix}_visualization.bdocjs',
#     tokens_file_path=f'./preprocessed_data/{prefix}_tokens.json',
#     sample_text_file_path=f"./preprocessed_data/{prefix}_sample_text.json",
#     dataset_split=DatasetSplit.train,
#     granularity=Granularity.fine
# )
# train_fine_preproc.run()


annotators: List[Annotator] = [NounPhraseAnnotator()]
prefix = f"{Dataset.multiconer_coarse.name}_{DatasetSplit.valid.name}_{Granularity.coarse.name}"
valid_coarse_preproc = PreprocessMulticoner(
    raw_data_folder_path='/Users/harshverma/Downloads/train_dev',
    entity_type_file_path=f'./preprocessed_data/{prefix}_types.txt',
    annotations_file_path=f'./preprocessed_data/{prefix}_annos.tsv',
    visualization_file_path=f'./preprocessed_data/{prefix}_visualization.bdocjs',
    tokens_file_path=f'./preprocessed_data/{prefix}_tokens.json',
    sample_text_file_path=f"./preprocessed_data/{prefix}_sample_text.json",
    samples_file_path=f"./preprocessed_data/{prefix}_samples.json",
    dataset_split=DatasetSplit.valid,
    granularity=Granularity.coarse,
    annotators=annotators
)
valid_coarse_preproc.run()

# prefix = f"{Dataset.multiconer.name}_{DatasetSplit.valid.name}_{Granularity.fine.name}"
# valid_fine_preproc = PreprocessMulticoner(
#     raw_data_folder_path='./multiconer-data-raw/train_dev',
#     entity_type_file_path=f'./preprocessed_data/{prefix}_types.txt',
#     annotations_file_path=f'./preprocessed_data/{prefix}_annos.tsv',
#     visualization_file_path=f'./preprocessed_data/{prefix}_visualization.bdocjs',
#     tokens_file_path=f'./preprocessed_data/{prefix}_tokens.json',
#     sample_text_file_path=f"./preprocessed_data/{prefix}_sample_text.json",
#     dataset_split=DatasetSplit.valid,
#     granularity=Granularity.fine
# )
# valid_fine_preproc.run()
