"""
Preprocessing for LivingNER dataset. https://temu.bsc.es/livingner/
"""
import os
from typing import List, Dict
import pandas as pd
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset
from preprocess import Preprocessor


class LivingNerPreprocessor(Preprocessor):
    """
    The LivingNER dataset preprocessor.
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
        assert dataset_split in (DatasetSplit.valid, DatasetSplit.train)

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
        ret = []
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
            ret.append(Sample(data, sample_id, sample_annos))
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
            print('HUMAN', file=output_file)
            print('SPECIES', file=output_file)


prefix = f"{Dataset.living_ner.name}_{DatasetSplit.valid.name}"
processor = LivingNerPreprocessor(
    raw_data_folder_path='../livingner_raw/training_valid_test_background_multilingual/valid',
    entity_type_file_path=f'../preprocessed_data/{prefix}_types.txt',
    annotations_file_path=f'../preprocessed_data/{prefix}_annos.tsv',
    visualization_file_path=f'../preprocessed_data/{prefix}_visualization.bdocjs',
    tokens_file_path=f'../preprocessed_data/{prefix}_tokens.json',
    sample_text_file_path=f"../preprocessed_data/{prefix}_sample_text.json",
    dataset_split=DatasetSplit.valid,
)
processor.run()


prefix = f"{Dataset.living_ner.name}_{DatasetSplit.train.name}"
processor = LivingNerPreprocessor(
    raw_data_folder_path='../livingner_raw/training_valid_test_background_multilingual/training',
    entity_type_file_path=f'../preprocessed_data/{prefix}_types.txt',
    annotations_file_path=f'../preprocessed_data/{prefix}_annos.tsv',
    visualization_file_path=f'../preprocessed_data/{prefix}_visualization.bdocjs',
    tokens_file_path=f'../preprocessed_data/{prefix}_tokens.json',
    sample_text_file_path=f"../preprocessed_data/{prefix}_sample_text.json",
    dataset_split=DatasetSplit.train,
)
processor.run()


# prefix = f"{Dataset.social_dis_ner.name}_{DatasetSplit.train.name}"
# train_preproc = SocialDisNerPreprocessor(
#     raw_data_folder_path='../socialdisner-data/train-valid-txt-files/training',
#     entity_type_file_path=f'../preprocessed_data/{prefix}_types.txt',
#     annotations_file_path=f'../preprocessed_data/{prefix}_annos.tsv',
#     visualization_file_path=f'../preprocessed_data/{prefix}_visualization.bdocjs',
#     tokens_file_path=f'../preprocessed_data/{prefix}_tokens.json',
#     sample_text_file_path=f"../preprocessed_data/{prefix}_sample_text.json",
#     dataset_split=DatasetSplit.train,
# )
# train_preproc.run()