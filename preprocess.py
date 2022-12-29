from structs import *
import util
import csv
from abc import ABC, abstractmethod
import spacy
import json


class Preprocessor(ABC):
    """
    An abstraction which standardizes the preprocessing
    of all NER datasets. This standardization makes it **easy** to 
    run new NER models on **all** NER dataset.
    """

    def __init__(
            self,
            raw_data_folder_path: str,
            entity_type_file_path: str,
            annotations_file_path: str,
            visualization_file_path: str,
            tokens_file_path: str,
            sample_text_file_path: str
    ) -> None:
        """
        Creates a preprocessor configured with some file paths
        that represent its output locations.

        Args:
            raw_data_folder_path: str
                the folder in which the raw data
                (provided by the organizers) is located.
            entity_type_file_path: str
                the file(.txt formatted) in which all the entity types of 
                the dataset are going to be listed(one per line).
            annotations_file_path: str
                the file(.tsv formatted) in which all the gold annotations of the
                dataset are going to be stored.
            visualization_file_path: str
                the file(.bdocjs formatted) that is going to be
                used by GATE developer to visualize the annotations.
            tokens_file_path: str
                the file(.json formatted) that will store the tokens
                of each sample of this dataset.
        """
        super().__init__()
        self.raw_data_folder_path = raw_data_folder_path
        self.entity_type_file_path = entity_type_file_path
        assert entity_type_file_path.endswith('.txt')
        self.visualization_file_path = visualization_file_path
        assert visualization_file_path.endswith('.bdocjs')
        self.annotations_file_path = annotations_file_path
        assert annotations_file_path.endswith('.tsv')
        self.tokens_file_path = tokens_file_path
        assert tokens_file_path.endswith('.json')
        self.sample_text_file_path = sample_text_file_path
        assert sample_text_file_path.endswith('.json')
        self.nlp = spacy.load("en_core_web_sm")
        print("Preprocessor Name:", type(self).__name__)
        self.samples = None

    def get_samples_cached(self) -> List[Sample]:
        if self.samples is None:
            print("first time")
            self.samples = self.get_samples()
        else:
            print("using cache")
        return self.samples

    @abstractmethod
    def get_samples(self) -> List[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """

    @abstractmethod
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

    def create_annotations_file(self) -> None:
        """
        Create a TSV(tab seperated values) formatted file that contains all the gold annotations for the dataset.

        The tsv file should have the following columns:
            - sample_id: The ID of the sample.
            - begin: An integer representing the beginning offset of the annotation in the sample.
            - end: An integer representing the end offset of the annotation in the sample.
            - type: The entity type of the annotation.
            - extraction: The text being annotated.
        """
        samples = self.get_samples_cached()
        with util.open_make_dirs(self.annotations_file_path, 'w') as annos_file:
            print("about to write")
            print(self.annotations_file_path)
            writer = csv.writer(annos_file, delimiter='\t')
            header = ['sample_id', 'begin', 'end', 'type', 'extraction']
            writer.writerow(header)
            for sample in samples:
                sample_annos = sample.annos
                for anno in sample_annos:
                    row = [sample.id, anno.begin_offset, anno.end_offset, anno.label_type, anno.extraction]
                    writer.writerow(row)

    def add_token_annotations(self, samples: List[Sample]) -> None:
        """
        Adds token annotations to each given sample by tokenizing
        the text using spacy.

        Args:
            samples: list of samples we want to tokenize
        """
        for sample in samples:
            tokenized_doc = self.nlp(sample.text)
            token_annos = []
            for token in tokenized_doc:
                start_offset = token.idx
                end_offset = start_offset + len(token)
                token_annos.append(Anno(start_offset, end_offset, "Token", str(token)))
            sample.annos.extend(token_annos)

    def create_visualization_file(self) -> None:
        """
        Create a .bdocjs formatted file which can be directly imported 
        into gate developer using the gate bdocjs plugin. 
        """
        samples = self.get_samples_cached()
        self.add_token_annotations(samples)
        sample_to_annos = {}
        sample_to_text = {}
        for sample in samples:
            sample_to_annos[sample.id] = sample.annos
            sample_to_text[sample.id] = sample.text
        util.create_visualization_file(
            self.visualization_file_path,
            sample_to_annos,
            sample_to_text
        )

    def create_tokens_file(self) -> None:
        """
        Create a json file that stores the tokens(and other corresponding information) 
        for each sample.
        """
        samples = self.get_samples_cached()
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
        with util.open_make_dirs(self.tokens_file_path, 'w') as output_file:
            json.dump(all_tokens_json, output_file, indent=4)

    def create_sample_text_file(self) -> None:
        """
        Create a json file that stores the text content of 
        each sample.

        The json file consists of one dictionary(mapping from
        sample ids to sample text content).
        """
        samples = self.get_samples_cached()
        sample_content = {}
        for sample in samples:
            sample_content[sample.id] = sample.text
        with util.open_make_dirs(self.sample_text_file_path, 'w') as output_file:
            json.dump(sample_content, output_file)

    def run(self) -> None:
        """
        Execute the preprocessing steps that generate files which
        can be used to train models.
        """
        print("Preprocessing...")
        print("creating entity file... ")
        self.create_entity_types_file()
        print("done")
        print("creating annos file...")
        self.create_annotations_file()
        print("done")
        print("creating tokens file...")
        self.create_tokens_file()
        print("done")
        print("creating visualization file...")
        self.create_visualization_file()
        print("done")
        print("creating sample text file ")
        self.create_sample_text_file()
        print("DONE Preprocessing!")
