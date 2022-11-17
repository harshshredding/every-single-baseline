from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict

OUTSIDE_LABEL_STRING = 'o'

class BioTag(Enum):
    out = 0
    begin = 1
    inside = 2

class Dataset(Enum):
    social_dis_ner = 1
    few_nerd = 2
    genia = 3
    living_ner = 4
    multiconer = 5
    legaleval = 6

class DatasetSplit(Enum):
    train = 0
    valid = 1
    test = 2

class Label:
    def __init__(self, label_type, bio_tag):
        self.label_type = label_type
        self.bio_tag = bio_tag

    def __key(self):
        return self.label_type, self.bio_tag

    def __str__(self):
        if self.bio_tag == BioTag.begin:
            return self.label_type + '-BEGIN'
        elif self.bio_tag == BioTag.inside:
            return self.label_type + '-INSIDE'
        else:
            return OUTSIDE_LABEL_STRING

    def __repr__(self) -> str:
        if self.bio_tag == BioTag.begin:
            return self.label_type + '-BEGIN'
        elif self.bio_tag == BioTag.inside:
            return self.label_type + '-INSIDE'
        else:
            return OUTSIDE_LABEL_STRING

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Label):
            return (other.label_type == self.label_type) and (other.bio_tag == self.bio_tag)
        raise NotImplementedError()

    @staticmethod
    def get_outside_label():
        return Label(OUTSIDE_LABEL_STRING, BioTag.out)

@dataclass
class Anno:
    begin_offset: int
    end_offset: int
    label_type: str
    extraction: str
    features: dict = field(default_factory=dict)

@dataclass
class TokenData:
    sample_id: str
    token_string: str
    token_len: int
    token_start_offset: int
    token_end_offset: int

@dataclass
class Sample:
    text: str
    id: str
    annos: List[Anno]

class Preprocessor(ABC):
    """
    An abstraction which allows standardizing the preprocessing
    for every NER dataset. Such standardization makes it possible to quickly 
    run our machine learning models on new datasets.
    """

    def __init__(
        self, 
        raw_data_folder_path: str, 
        entity_type_file_path: str, 
        annotations_file_path: str,
        visualization_file_path: str,
        tokens_file_path: str
    ) -> None:
        """
        Creates a preprocessor configured with some file paths
        that represent its output locations.

        Args:
            raw_data_folder_path: str
                the folder in which the raw data
                (provided by the organizers) is located.
            entity_type_file_path: str
                the file(.txt formatted) in which all the entity types of 
                the dataset are going to be listed(one per line).
            annotations_file_path: str
                the file(.tsv formatted) in which all the gold annotations of the
                dataset are going to be stored.
            visualization_file_path: str
                the file(.bdocjs formatted) that is going to be
                used by GATE developer to visualize the annotations.
            tokens_file_path: str
                the file(.json formatted) that will store the tokens
                of each sample of this dataset.
        """
        super().__init__()
        self.raw_data_folder_path = raw_data_folder_path
        self.entity_type_file_path = entity_type_file_path
        self.visualization_file_path = visualization_file_path
        self.annotations_file_path = annotations_file_path
        self.tokens_file_path = tokens_file_path

    @abstractmethod
    def get_samples(self) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """
        pass

    @abstractmethod
    def create_entity_types_file(self) -> None:
        """
        Create a .txt file that lists all possible entity types -- one per line.

        For eg. the below mock txt file lists entity types ORG, PER, and LOC.
        <<< file start
        ORG
        PER
        LOC
        <<< file end
        """
        pass

    @abstractmethod
    def create_annotations_file(self) -> None:
        """
        Create a TSV(tab seperated values) formatted file that contains all the gold annotations for the dataset.

        The tsv file should have the following columns:
            - sample_id: The ID of the sample.
            - begin: An integer representing the beginning offset of the annotation in the sample.
            - end: An integer representing the end offset of the annotation in the sample.
            - type: The entity type of the annotation.
            - extraction: The text being annotated.
        """
        pass

    @abstractmethod
    def create_visualization_file(self) -> None:
        """
        Create a .bdocjs formatted file which can me directly imported into gate developer.
        """
        pass

    @abstractmethod
    def create_tokens_file(self) -> None:
        """
        Create a json file that stores the tokens(and other corresponding information) 
        for each sample.
        """
        pass

    def run(self) -> None:
        """
        Execute the preprocessing steps that generate files which
        can be used to train models.
        """
        self.create_entity_types_file()
        self.create_annotations_file()
        self.create_tokens_file()
        self.create_visualization_file()


SampleAnnotations = Dict[str, List[Anno]]