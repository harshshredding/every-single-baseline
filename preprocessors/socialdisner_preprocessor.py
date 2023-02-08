"""
Preprocessing for SocialDisNER dataset.
"""
import os
from typing import List, Dict
import pandas as pd
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset, AnnotationCollection
from preprocess import Preprocessor
from preamble import *
from annotators import Annotator


class PreprocessSocialDisNer(Preprocessor):
    """
    The SocialDisNER dataset preprocessor.
    """

    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator]
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            dataset=Dataset.social_dis_ner,
            annotators=annotators
        )
        assert (dataset_split == DatasetSplit.train) or (dataset_split == DatasetSplit.valid)

    @staticmethod
    def __get_tweet_annos_dict() -> Dict[SampleId, List[Anno]]:
        """
        Read annotations for each sample from the given file and return
        a dict from sample_ids to corresponding annotations.
        """
        annos_file_path = "socialdisner-data/mentions.tsv"
        df = pd.read_csv(annos_file_path, sep='\t')
        sample_to_annos = {}
        for i, row in df.iterrows():
            annos_list = sample_to_annos.get(str(row['tweets_id']), [])
            annos_list.append(
                Anno(row['begin'], row['end'], row['type'], row['extraction']))
            sample_to_annos[str(row['tweets_id'])] = annos_list
        return sample_to_annos

    def get_samples(self) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """
        match self.dataset_split:
            case DatasetSplit.train:
                raw_data_folder_path = 'socialdisner-data/train-valid-txt-files/training'
            case DatasetSplit.valid:
                raw_data_folder_path = 'socialdisner-data/train-valid-txt-files/validation'
            case _:
                raise RuntimeError('only train and valid are supported')
        ret = []
        data_files_list = os.listdir(raw_data_folder_path)
        tweet_to_annos = self.__get_tweet_annos_dict()
        for filename in data_files_list:
            data_file_path = os.path.join(raw_data_folder_path, filename)
            with open(data_file_path, 'r') as f:
                data = f.read()
                new_str = str()
                for char in data:
                    if ord(char) < 2047:
                        new_str = new_str + char
                    else:
                        new_str = new_str + ' '
                data = new_str
            twitter_id = filename[:-4]
            tweet_annos = tweet_to_annos.get(twitter_id, [])
            ret.append(
                Sample(
                    text=data,
                    id=twitter_id,
                    annos=AnnotationCollection(tweet_annos, [])
                )
            )
        return ret

    def get_entity_types(self) -> List[str]:
        """
        Create a .txt file that lists all possible entity types -- one per line.

        For eg. the below mock txt file lists entity types ORG, PER, and LOC.
        <<< file start
        ORG
        PER
        LOC
        <<< file end
        """
        return ['ENFERMEDAD']
