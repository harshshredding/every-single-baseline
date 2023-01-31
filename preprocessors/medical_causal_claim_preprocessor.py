from typing import List
from structs import Anno, Sample, DatasetSplit, AnnotationCollection
from preprocess import Preprocessor
import csv
import json


def parse_annos(annotation_json: str, sample_text: str) -> List[Anno]:
    parsed_json = json.loads(annotation_json)
    assert len(parsed_json) == 1
    assert len(parsed_json[0]) == 1
    assert 'crowd-entity-annotation' in parsed_json[0]
    assert len(parsed_json[0]['crowd-entity-annotation']) == 1
    assert 'entities' in parsed_json[0]['crowd-entity-annotation']
    raw_annos = parsed_json[0]['crowd-entity-annotation']['entities']
    result_annos = []
    for raw_anno in raw_annos:
        start_offset = raw_anno['startOffset']
        end_offset = raw_anno['endOffset']
        label_type = raw_anno['label']
        extraction = sample_text[start_offset:end_offset]
        result_annos.append(Anno(start_offset, end_offset, label_type, extraction))
    return result_annos


class MedicalCausalClaimPreprocessor(Preprocessor):
    def __init__(
            self,
            name: str,
            entity_type_file_path: str,
            annotations_file_path: str,
            visualization_file_path: str,
            tokens_file_path: str,
            sample_text_file_path: str,
            dataset_split: DatasetSplit,
            samples_file_path: str,
    ) -> None:
        super().__init__(
            name=name,
            entity_type_file_path=entity_type_file_path,
            annotations_file_path=annotations_file_path,
            visualization_file_path=visualization_file_path,
            tokens_file_path=tokens_file_path,
            sample_text_file_path=sample_text_file_path,
            samples_file_path=samples_file_path
        )
        self.dataset_split = dataset_split
        assert (dataset_split == DatasetSplit.train) or \
               (dataset_split == DatasetSplit.valid) or \
               (dataset_split == DatasetSplit.test)

    def get_samples(self) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """
        if self.dataset_split == DatasetSplit.train:
            raw_data_file_path = "./medical_claim_subtask_1_raw/st1_train_inc_text.csv"
        elif self.dataset_split == DatasetSplit.valid:
            raw_data_file_path = "./medical_claim_subtask_1_raw/st1_train_inc_text.csv"
        elif self.dataset_split == DatasetSplit.test:
            raw_data_file_path = "./medical_claim_subtask_1_raw/st1_test_inc_text.csv"
        else:
            raise RuntimeError("no other dataset split possible")

        with open(raw_data_file_path, 'r') as data_csv_file:
            reader = csv.DictReader(data_csv_file)
            ret = []
            for row in reader:
                post_id = row['post_id']
                sample_text = row['text']
                gold_annos = []
                if (self.dataset_split == DatasetSplit.train) or (self.dataset_split == DatasetSplit.valid):
                    labels_json = row['stage1_labels']
                    gold_annos = parse_annos(labels_json, sample_text)
                ret.append(Sample(sample_text, post_id, AnnotationCollection(gold_annos, [])))
            if self.dataset_split == DatasetSplit.train:
                percent_85 = int(len(ret) * 0.85)
                ret = ret[:percent_85]
            elif self.dataset_split == DatasetSplit.valid:
                percent_85 = int(len(ret) * 0.85)
                ret = ret[percent_85:]
            elif self.dataset_split == DatasetSplit.test:
                pass
            return ret

    def create_entity_types_file(self) -> None:
        with open(self.entity_type_file_path, 'w') as output_file:
            print('claim', file=output_file)
            print('question', file=output_file)
            print('claim_per_exp', file=output_file)
            print('per_exp', file=output_file)


def train_preprocess():
    train_preprocessor = MedicalCausalClaimPreprocessor(
        name="Medical Causal Claim Train",
        entity_type_file_path=f'preprocessed_data/medical_causal_train_types.txt',
        annotations_file_path=f'preprocessed_data/medical_causal_train_annos.tsv',
        visualization_file_path=f'preprocessed_data/medical_causal_train_visualization.bdocjs',
        tokens_file_path=f'preprocessed_data/medical_causal_train_tokens.json',
        sample_text_file_path=f"preprocessed_data/medical_causal_train_sample_text.json",
        dataset_split=DatasetSplit.train,
        samples_file_path="preprocessed_data/medical_causal_train_samples.json"
    )
    train_preprocessor.run()


def valid_preprocess():
    valid_preprocessor = MedicalCausalClaimPreprocessor(
        name="Medical Causal Claim Valid",
        entity_type_file_path=f'preprocessed_data/medical_causal_valid_types.txt',
        annotations_file_path=f'preprocessed_data/medical_causal_valid_annos.tsv',
        visualization_file_path=f'preprocessed_data/medical_causal_valid_visualization.bdocjs',
        tokens_file_path=f'preprocessed_data/medical_causal_valid_tokens.json',
        sample_text_file_path=f"preprocessed_data/medical_causal_valid_sample_text.json",
        dataset_split=DatasetSplit.valid,
        samples_file_path="preprocessed_data/medical_causal_valid_samples.json"
    )
    valid_preprocessor.run()


def test_preprocess():
    test_preprocessor = MedicalCausalClaimPreprocessor(
        name="Medical Causal Claim Test",
        entity_type_file_path=f'preprocessed_data/medical_causal_test_types.txt',
        annotations_file_path=f'preprocessed_data/medical_causal_test_annos.tsv',
        visualization_file_path=f'preprocessed_data/medical_causal_test_visualization.bdocjs',
        tokens_file_path=f'preprocessed_data/medical_causal_test_tokens.json',
        sample_text_file_path=f"preprocessed_data/medical_causal_test_sample_text.json",
        dataset_split=DatasetSplit.test,
        samples_file_path="preprocessed_data/medical_causal_test_samples.json"
    )
    test_preprocessor.run()


def main():
    train_preprocess()
    valid_preprocess()
    test_preprocess()


if __name__ == '__main__':
    main()
