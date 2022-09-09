from enum import Enum

OUTSIDE_LABEL_STRING = 'o'


class BioTag(Enum):
    out = 0
    begin = 1
    inside = 2


class Dataset(Enum):
    social_dis_ner = 1
    few_nerd = 2


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

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Label):
            return (other.label_type == self.label_type) and (other.bio_tag == self.bio_tag)
        raise NotImplementedError()

    @staticmethod
    def get_outside_label():
        return Label(OUTSIDE_LABEL_STRING, BioTag.out)


class Anno:
    def __init__(self, begin_offset, end_offset, label_type, extraction):
        self.begin_offset = begin_offset
        self.end_offset = end_offset
        self.label_type = label_type
        self.extraction = extraction


class TokenData:
    def __init__(self, sample_id, sample_start_offset, token_string, token_len, token_start, token_end, label):
        self.sample_id = sample_id
        self.sample_start_offset = sample_start_offset
        self.token_string = token_string
        self.token_len = token_len
        self.token_start_offset = token_start
        self.token_end_offset = token_end
        self.label = label
