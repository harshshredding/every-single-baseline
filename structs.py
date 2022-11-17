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
    @abstractmethod
    def get_samples(self, **kwargs) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        Args:
            kwargs: variable num of dataset-specific arguments
        """
        pass

    @abstractmethod
    def create_entity_types_file(self, output_file_path) -> None:
        """
        Create a .txt file that lists all possible entity types -- one per line.

        For eg. the below txt file lists entity types ORG, PER, and LOC.
        <<< file start
        ORG
        PER
        LOC
        <<< file end

        Args:
            output_file_path: the path of the entity types file to create
        """
        pass

    @abstractmethod
    def create_annotations_file(self, output_file_path, **kwargs) -> None:
        """
        Create a TSV(tab seperated values) format file that contains all the gold annotations for the dataset.

        The tsv file should have the following columns:
            - sample_id: The ID of the sample.
            - begin: An integer representing the beginning offset of the annotation in the sample.
            - end: An integer representing the end offset of the annotation in the sample.
            - type: The entity type of the annotation.
            - extraction: The text being annotated.

        Args:
            output_file_path: the path of the annotations file to create 
            kwargs: variable num of dataset-specific arguments
        """

    @abstractmethod
    def create_visualization_file()

SampleAnnotations = Dict[str, List[Anno]]