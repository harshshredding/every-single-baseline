from typing import List
from structs import Anno, Sample, DatasetSplit, SampleId, Dataset
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
            entity_type_file_path: str,
            annotations_file_path: str,
            visualization_file_path: str,
            tokens_file_path: str,
            sample_text_file_path: str,
            dataset_split: DatasetSplit,
    ) -> None:
        super().__init__(
            entity_type_file_path,
            annotations_file_path,
            visualization_file_path,
            tokens_file_path,
            sample_text_file_path
        )
        self.dataset_split = dataset_split
        assert (dataset_split == DatasetSplit.train) or (
                dataset_split == DatasetSplit.valid)

    def get_samples(self) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """
        with open('/home/harsh/every-single-baseline/causal_claims_raw/st1_train_sample_inc_text.csv',
                  'r') as data_csv_file:
            reader = csv.DictReader(data_csv_file)
            ret = []
            for row in reader:
                post_id = row['post_id']
                sample_text = row['text']
                labels_json = row['stage1_labels']
                annos = parse_annos(labels_json, sample_text)
                ret.append(Sample(sample_text, post_id, annos))
            return ret

    def create_entity_types_file(self) -> None:
        with open(self.entity_type_file_path, 'w') as output_file:
            print('claim', file=output_file)
            print('question', file=output_file)
            print('claim_per_exp', file=output_file)
            print('per_exp', file=output_file)


def main():
    train_preprocessor = MedicalCausalClaimPreprocessor(
        entity_type_file_path=f'preprocessed_data/test_medical_causal_types.txt',
        annotations_file_path=f'preprocessed_data/test_medical_causal_annos.tsv',
        visualization_file_path=f'preprocessed_data/test_medical_causal_visualization.bdocjs',
        tokens_file_path=f'preprocessed_data/test_medical_causal_tokens.json',
        sample_text_file_path=f"preprocessed_data/test_medical_causal_sample_text.json",
        dataset_split=DatasetSplit.train,
    )
    train_preprocessor.run()


if __name__ == '__main__':
    main()
