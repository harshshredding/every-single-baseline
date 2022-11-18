from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict
import spacy
from gatenlp import Document
import json

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

SampleAnnotations = Dict[str, List[Anno]]
SampleId = str