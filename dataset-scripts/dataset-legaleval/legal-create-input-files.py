import json
import csv
import spacy
from structs import Anno, Preprocessor, Sample, DatasetSplit
from typing import List
import util
from gatenlp import Document
from enum import Enum
import os
import shutil

class LegalSection(Enum):
    PREAMBLE = 0
    JUDGEMENT = 1

class PreprocessLegal(Preprocessor):

    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    def __get_raw_annos(self, sample):
        return sample['annotations'][0]['result']

    def __parse_anno_raw(self, anno_raw) -> Anno:
        anno_raw = anno_raw['value']
        return Anno(anno_raw['start'], anno_raw['end'], anno_raw['labels'][0], anno_raw['text'])

    def clean_up(self):
        if os.path.isdir('./datasets/legaleval'):
            print("removing dir")
            shutil.rmtree('./datasets/legaleval')
    
    def create_types_file(self):
        types_set = set()
        judgement_samples = self.get_samples(DatasetSplit.train, LegalSection.JUDGEMENT)
        for sample in judgement_samples:
            types_set.update([anno.label_type for anno in sample.annos])
        preamble_samples = self.get_samples(DatasetSplit.train, LegalSection.PREAMBLE)
        for sample in preamble_samples:
            types_set.update([anno.label_type for anno in sample.annos])
        print("num types: ", len(types_set))
        assert len(types_set) == 14
        print(util.p_string(list(types_set)))
        with util.open_make_dirs(f'./datasets/legaleval/types.txt', 'w') as types_file:
            for type in types_set:
                print(type, file=types_file)


    def create_anno_file(self, dataset_split: DatasetSplit, section: LegalSection):
        samples = self.get_samples(dataset_split, section)
        if dataset_split == DatasetSplit.test:
            raise Exception("cannot create annos file for test set")
        elif dataset_split == DatasetSplit.train:
            annos_file_path = f'./datasets/legaleval/gold-annos/{dataset_split.name}/{section.name}/annos-{section.name}-{dataset_split.name}.tsv'
        else:
            annos_file_path = f'./datasets/legaleval/gold-annos/{dataset_split.name}/{section.name}/annos-{section.name}-{dataset_split.name}.tsv'
        with util.open_make_dirs(annos_file_path, 'w') as annos_file:
            print("about to write")
            print(annos_file_path)
            writer = csv.writer(annos_file, delimiter='\t')
            header = ['sample_id', 'begin', 'end', 'type', 'extraction']
            writer.writerow(header)
            for sample in samples:
                sample_annos = sample.annos
                for anno in sample_annos:
                    row = [sample.id, anno.begin_offset, anno.end_offset, anno.label_type, anno.extraction]
                    writer.writerow(row)

    def get_samples(self, dataset_split: DatasetSplit, section: LegalSection) -> List[Sample]:
        if dataset_split == DatasetSplit.test:
            raise Exception("cannot create annos file for test set")
        elif dataset_split == DatasetSplit.train:
            raw_file_path = f'./legal_raw/NER_TRAIN/NER_TRAIN_{section.name}.json'
        else:
            raw_file_path = f'./legal_raw/NER_DEV/NER_DEV_{section.name}.json'
        ret = []
        with open(raw_file_path, 'r') as f:
            json_data = json.load(f)
            print("num samples", len(json_data))
            for raw_sample in json_data:
                assert 'id' in raw_sample
                sample_id = raw_sample['id']
                sample_text= raw_sample['data']['text']
                assert 'annotations' in raw_sample
                assert len(raw_sample['annotations']) == 1
                assert 'result' in raw_sample['annotations'][0]
                raw_annos = self.__get_raw_annos(raw_sample)
                parsed_annos = [self.__parse_anno_raw(raw_anno) for raw_anno in raw_annos]
                ret.append(Sample(sample_text, sample_id, parsed_annos))
        return ret

    def create_gate_file(self, dataset_split: DatasetSplit, section: LegalSection):
        gate_file_folder_path = "./datasets/legaleval/gate-input" 
        util.create_directory_structure(gate_file_folder_path)
        samples = self.get_samples(dataset_split, section)
        self.add_token_annotations(samples)
        sample_offset = 0
        document_text = ""
        document_annos = []
        for sample in samples:
            document_text += (sample.text + '\n')
            document_annos.append(Anno(sample_offset, len(document_text), 'Sample', '', {"id": sample.id}))
            for anno in sample.annos:
                new_start_offset = anno.begin_offset + sample_offset
                new_end_offset = anno.end_offset + sample_offset
                anno.features['orig_start_offset'] = anno.begin_offset
                anno.features['orig_end_offset'] = anno.end_offset
                document_annos.append(Anno(new_start_offset, new_end_offset, anno.label_type, anno.extraction, anno.features))
            sample_offset += (len(sample.text) + 1)
        gate_document = Document(document_text)
        default_ann_set = gate_document.annset()
        for corrected_anno in document_annos:
            default_ann_set.add(int(corrected_anno.begin_offset), int(corrected_anno.end_offset), corrected_anno.label_type, corrected_anno.features)
        gate_document.save(gate_file_folder_path + f'/{section.name}-{dataset_split.name}.bdocjs')

    def add_token_annotations(self, samples: List[Sample]): 
        for sample in samples:
            tokenized_doc = self.nlp(sample.text)
            token_annos = []
            for token in tokenized_doc:
                start_offset = token.idx
                end_offset = start_offset + len(token)
                token_annos.append(Anno(start_offset, end_offset, "Token", str(token)))
            sample.annos.extend(token_annos)

    def create_input_file(self, dataset_split: DatasetSplit, section: LegalSection):
        folder_path = f"./datasets/legaleval/input-files/{dataset_split.name}/{section.name}"
        util.create_directory_structure(folder_path)
        samples = self.get_samples(dataset_split, section)
        nlp = spacy.load("en_core_web_sm")
        all_tokens_json = []
        for sample in samples:
            doc = nlp(sample.text)
            for token in doc:
                start_offset = token.idx
                end_offset = start_offset + len(token)
                token_json = {'Token': [{"string": str(token), "startOffset": start_offset,
                                     "endOffset": end_offset, "length": len(token)}],
                              'Sample': [{"id": sample.id, "startOffset": 0}],
                            }
                all_tokens_json.append(token_json)
        with util.open_make_dirs(f'{folder_path}/{dataset_split.name}-{section.name}.json', 'w') as output_file:
            json.dump(all_tokens_json, output_file, indent=4)
                


preproc = PreprocessLegal()
preproc.clean_up()

preproc.create_anno_file(DatasetSplit.train, LegalSection.PREAMBLE)
preproc.create_anno_file(DatasetSplit.valid, LegalSection.PREAMBLE)
preproc.create_gate_file(DatasetSplit.train, LegalSection.PREAMBLE)
preproc.create_gate_file(DatasetSplit.valid, LegalSection.PREAMBLE)

preproc.create_anno_file(DatasetSplit.train, LegalSection.JUDGEMENT)
preproc.create_anno_file(DatasetSplit.valid, LegalSection.JUDGEMENT)
preproc.create_gate_file(DatasetSplit.train, LegalSection.JUDGEMENT)
preproc.create_gate_file(DatasetSplit.valid, LegalSection.JUDGEMENT)

preproc.create_types_file()

preproc.create_input_file(DatasetSplit.valid, LegalSection.PREAMBLE)
preproc.create_input_file(DatasetSplit.valid, LegalSection.JUDGEMENT)
preproc.create_input_file(DatasetSplit.train, LegalSection.PREAMBLE)
preproc.create_input_file(DatasetSplit.train, LegalSection.JUDGEMENT)