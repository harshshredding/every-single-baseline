"""
Preprocessing for SocialDisNER dataset.
"""
import os
from typing import List, Dict
import pandas as pd
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset, AnnotationCollection
from utils.preprocess import Preprocessor, PreprocessorRunType
from preamble import *
from annotators import Annotator
import json


class PreprocessSocialDisNer(Preprocessor):
    """
    The SocialDisNER dataset preprocessor.
    """

    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            dataset=Dataset.social_dis_ner,
            annotators=annotators,
            run_mode=run_mode
        )
        assert (dataset_split == DatasetSplit.train) or (dataset_split == DatasetSplit.valid)

    def __get_tweet_annos_dict(self) -> Dict[SampleId, List[Anno]]:
        """
        Read annotations for each sample from the given file and return
        a dict from sample_ids to corresponding annotations.
        """
        if self.dataset_split == DatasetSplit.test:
            return {}
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
            case DatasetSplit.test:
                raw_data_folder_path = 'social_dis_ner_test_data/test-data/test-data-txt-files'
            case _:
                raise RuntimeError(f"Split {self.dataset_split.name} not supported yet")

        ret = []
        data_files_list = os.listdir(raw_data_folder_path)
        tweet_to_annos = self.__get_tweet_annos_dict()
        for filename in data_files_list:
            data_file_path = os.path.join(raw_data_folder_path, filename)
            with open(data_file_path, 'r') as f:
                tweet_data = f.read()
                new_str = str()
                for char in tweet_data:
                    if ord(char) < 2047:
                        new_str = new_str + char
                    else:
                        new_str = new_str + ' '
                tweet_data = new_str
            twitter_id = filename[:-4]
            tweet_annos = tweet_to_annos.get(twitter_id, [])
            ret.append(
                Sample(
                    text=tweet_data,
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


class PreprocessSocialDisNerGPT(Preprocessor):
    """
    The SocialDisNER dataset preprocessor.
    """

    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            dataset=Dataset.social_dis_ner,
            annotators=annotators,
            run_mode=run_mode
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

        gpt_predictions_dict = self.get_gpt_predictions()

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
            gpt_predictions_prefix = gpt_predictions_dict[twitter_id] + ' [SEP] '
            for anno in tweet_annos:
                anno.begin_offset = anno.begin_offset + len(gpt_predictions_prefix)
                anno.end_offset = anno.end_offset + len(gpt_predictions_prefix)
            ret.append(
                Sample(
                    text=gpt_predictions_prefix + data,
                    id=twitter_id,
                    annos=AnnotationCollection(tweet_annos, [])
                )
            )
        return ret

    def get_gpt_predictions(self):
        match self.dataset_split:
            case DatasetSplit.valid:
                gpt_predictions_file_path = './social_dis_ner_openai_output_valid.json'
            case DatasetSplit.train:
                gpt_predictions_file_path = './social_dis_ner_openai_output_train.json'
            case __:
                raise RuntimeError("Can only support train and valid")

        with open(gpt_predictions_file_path, 'r') as gpt_predictions_file:
            gpt_predictions = json.load(gpt_predictions_file)

        print("Got predictions", len(gpt_predictions))

        gpt_predictions_dict = {
            sample_id: ','.join(diseases)
            for sample_id, diseases in gpt_predictions
        }

        match self.dataset_split:
            case DatasetSplit.valid:
                assert len(gpt_predictions_dict) == 2500
            case DatasetSplit.train:
                assert len(gpt_predictions_dict) == 5000
            case _:
                raise RuntimeError("Can only support train and valid")

        print("dict len", len(gpt_predictions_dict))
        return gpt_predictions_dict

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


class PreprocessSocialDisNerChatGPT(PreprocessSocialDisNerGPT):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: List[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            annotators=annotators,
            run_mode=run_mode
        )

    def get_gpt_predictions(self):
        match self.dataset_split:
            case DatasetSplit.valid:
                chatgpt_predictions_file_path = './chatgpt_social_dis_ner_valid.json'
            case DatasetSplit.train:
                chatgpt_predictions_file_path = './chatgpt_social_dis_ner_train.json'
            case __:
                raise RuntimeError("Can only support train and valid")

        with open(chatgpt_predictions_file_path, 'r') as gpt_predictions_file:
            chat_gpt_predictions = json.load(gpt_predictions_file)

        print("Got predictions", len(chat_gpt_predictions))

        gpt_predictions_dict = {
            sample_id: diseases
            for sample_id, diseases in chat_gpt_predictions
        }

        match self.dataset_split:
            case DatasetSplit.valid:
                assert len(gpt_predictions_dict) == 2500
            case DatasetSplit.train:
                assert len(gpt_predictions_dict) == 5000
            case _:
                raise RuntimeError("Can only support train and valid")

        print("dict len", len(gpt_predictions_dict))
        return gpt_predictions_dict
