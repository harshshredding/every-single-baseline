"""
Preprocessing for LivingNER dataset. https://temu.bsc.es/livingner/
"""
import os
from typing import List, Dict
import pandas as pd
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset
from preprocess import Preprocessor
import util
from annotators import Annotator


class LivingNerPreprocessor(Preprocessor):
    """
    The LivingNER dataset preprocessor.
    """

    def __init__(
            self,
            dataset_split: DatasetSplit,
            preprocessor_type: str,
            annotators: List[Annotator]
    ) -> None:
        super().__init__(
            dataset_split=dataset_split,
            preprocessor_type=preprocessor_type,
            dataset=Dataset.living_ner,
            annotators=annotators
        )
        match dataset_split:
            case DatasetSplit.train:
                self.raw_data_folder_path = './livingner_raw/training_valid_test_background_multilingual/training'
            case DatasetSplit.valid:
                self.raw_data_folder_path = './livingner_raw/training_valid_test_background_multilingual/valid'
            case _:
                raise RuntimeError("Can only support train and valid")

    def __get_annos_dict(self) -> Dict[SampleId, List[Anno]]:
        """
        Read annotations for each sample from the given file and return
        a dict from sample_ids to corresponding annotations.
        """
        if self.dataset_split == DatasetSplit.valid:
            annos_file_path = f"{self.raw_data_folder_path}/subtask1-NER/validation_entities_subtask1.tsv"
        else:
            annos_file_path = f"{self.raw_data_folder_path}/subtask1-NER/training_entities_subtask1.tsv"
        data_frame = pd.read_csv(annos_file_path, sep='\t')
        sample_to_annos = {}
        for _, row in data_frame.iterrows():
            annos_list = sample_to_annos.get(str(row['filename']), [])
            annos_list.append(
                Anno(row['off0'], row['off1'], row['label'], row['span']))
            sample_to_annos[str(row['filename'])] = annos_list
        return sample_to_annos

    def get_samples(self) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """
        document_samples = []
        raw_text_files_folder_path = f"{self.raw_data_folder_path}/text-files"
        data_files_list = os.listdir(raw_text_files_folder_path)
        annos_dict = self.__get_annos_dict()
        for filename in data_files_list:
            data_file_path = os.path.join(raw_text_files_folder_path, filename)
            with open(data_file_path, 'r', encoding="utf-8") as f:
                data = f.read()
                new_str = str()
                for char in data:
                    if ord(char) < 2047:
                        new_str = new_str + char
                    else:
                        new_str = new_str + ' '
                data = new_str
            sample_id = filename[:-4]
            sample_annos = annos_dict.get(sample_id, [])
            document_samples.append(Sample(data, sample_id, sample_annos))
        sentence_samples = []
        for doc_sample in document_samples:
            sentence_samples.extend(util.make_sentence_samples(doc_sample, self.nlp))
        return sentence_samples

    def get_entity_types(self) -> List[str]:
        return ['HUMAN', 'SPECIES']
