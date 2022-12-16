"""
Preprocessing for SocialDisNER dataset.
"""
import os
from typing import List, Dict
import pandas as pd
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset
from preprocess import Preprocessor


class SocialDisNerPreprocessor(Preprocessor):
    """
    The SocialDisNER dataset preprocessor.
    """

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
        assert (dataset_split == DatasetSplit.train) or (
            dataset_split == DatasetSplit.valid)

    def __get_tweet_annos_dict(self) -> Dict[SampleId, List[Anno]]:
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
        ret = []
        data_files_list = os.listdir(self.raw_data_folder_path)
        tweet_to_annos = self.__get_tweet_annos_dict()
        for filename in data_files_list:
            data_file_path = os.path.join(self.raw_data_folder_path, filename)
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
            ret.append(Sample(data, twitter_id, tweet_annos))
        return ret

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
        with open(self.entity_type_file_path, 'w') as output_file:
            print('ENFERMEDAD', file=output_file)


prefix = f"{Dataset.social_dis_ner.name}_{DatasetSplit.valid.name}"
valid_preproc = SocialDisNerPreprocessor(
    raw_data_folder_path='socialdisner-data/train-valid-txt-files/validation',
    entity_type_file_path=f'preprocessed_data/{prefix}_types.txt',
    annotations_file_path=f'preprocessed_data/{prefix}_annos.tsv',
    visualization_file_path=f'preprocessed_data/{prefix}_visualization.bdocjs',
    tokens_file_path=f'preprocessed_data/{prefix}_tokens.json',
    sample_text_file_path=f"preprocessed_data/{prefix}_sample_text.json",
    dataset_split=DatasetSplit.valid,
)
valid_preproc.run()


prefix = f"{Dataset.social_dis_ner.name}_{DatasetSplit.train.name}"
train_preproc = SocialDisNerPreprocessor(
    raw_data_folder_path='socialdisner-data/train-valid-txt-files/training',
    entity_type_file_path=f'preprocessed_data/{prefix}_types.txt',
    annotations_file_path=f'preprocessed_data/{prefix}_annos.tsv',
    visualization_file_path=f'preprocessed_data/{prefix}_visualization.bdocjs',
    tokens_file_path=f'preprocessed_data/{prefix}_tokens.json',
    sample_text_file_path=f"preprocessed_data/{prefix}_sample_text.json",
    dataset_split=DatasetSplit.train,
)
train_preproc.run()
