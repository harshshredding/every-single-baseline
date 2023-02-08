from structs import *
import util
from abc import ABC, abstractmethod
from annotators import Annotator, TokenAnnotator
from preamble import *
from pydoc import locate


class Preprocessor(ABC):
    """
    An abstraction which standardizes the preprocessing
    of all NER datasets. This standardization makes it **easy** to 
    run new NER models on **all** NER dataset.
    """

    def __init__(
            self,
            dataset_split: DatasetSplit,
            preprocessor_type: str,
            dataset: Dataset,
            annotators: List[Annotator] = []
    ) -> None:
        """
        Creates a preprocessor configured with some file paths
        that represent its output locations.

        Args:
            preprocessor_type: the type of preprocessor (mention details like annotators)
            dataset: the name of the dataset we are preprocessing
            annotators: the list of annotators we need to run on the dataset
        """
        super().__init__()
        self.preprocessor_name = preprocessor_type
        self.preprocessor_full_name = f"{dataset.name}_{dataset_split.name}_{preprocessor_type}"
        self.data_folder_path = f"./preprocessed_data"
        self.visualization_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_visualization.bdocjs"
        self.samples_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_samples.json"
        self.entity_types_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_types.txt"
        self.samples = None
        self.annotators = annotators
        self.dataset_split = dataset_split
        self.preprocessor_type = preprocessor_type
        self.dataset = dataset

    def run_annotation_pipeline(self, samples: List[Sample]):
        """
        All samples are annotated by the given annotators in a
        defined sequence (some annotator depend on others).

        Args:
            samples: the samples we want to annotate.
        """
        assert self.samples is not None
        for annotator in self.annotators:
            annotator.annotate(samples)

    def get_samples_cached(self) -> List[Sample]:
        """
        We cache `Samples` after extracting them from raw data.
        """
        if self.samples is None:
            print("first time extracting samples")
            self.samples = self.get_samples()
            self.run_annotation_pipeline(self.samples)
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
    def get_entity_types(self) -> List[str]:
        """
        Returns:
            The list of all entity types (represented as unique strings).
        """

    def create_entity_types_file(self):
        """
        Creates a file that lists the entity types.
        """
        all_entity_types = self.get_entity_types()
        all_types_set = set(all_entity_types)
        assert len(all_types_set) == len(all_entity_types)  # no duplicates allowed
        with util.open_make_dirs(self.entity_types_file_path, 'w') as types_file:
            for type_name in all_types_set:
                print(type_name, file=types_file)

    def create_visualization_file(self) -> None:
        """
        Create a .bdocjs formatted file which can be directly imported 
        into gate developer using the gate bdocjs plugin. 
        """
        samples = self.get_samples_cached()
        sample_to_annos = {}
        sample_to_text = {}
        for sample in samples:
            gold_and_external_annos = sample.annos.gold + sample.annos.external
            sample_to_annos[sample.id] = gold_and_external_annos
            sample_to_text[sample.id] = sample.text

        util.create_visualization_file(
            self.visualization_file_path,
            sample_to_annos,
            sample_to_text
        )

    def store_samples(self) -> None:
        """
        Persist the samples on disk.
        """
        samples = self.get_samples_cached()
        util.write_samples(samples, self.samples_file_path)

    def run(self) -> None:
        """
        Execute the preprocessing steps that generate files which
        can be used to train models.
        """
        print_section()
        print_green(f"Preprocessing {self.preprocessor_full_name}")
        print("Creating data folder")
        util.create_directory_structure(self.data_folder_path)
        print("Creating entity file... ")
        self.create_entity_types_file()
        print("Creating visualization file...")
        self.create_visualization_file()
        print("Creating samples json file")
        self.store_samples()
        print("Done Preprocessing!")


def preprocess_train_and_valid_data(preprocessor_module_name: str, preprocessor_name: str):
    preprocessor_class = locate(f"preprocessors.{preprocessor_module_name}.{preprocessor_name}")
    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.valid,
        preprocessor_type="vanilla",
        annotators=[TokenAnnotator()]
    )
    preprocessor.run()

    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.train,
        preprocessor_type="vanilla",
        annotators=[TokenAnnotator()]
    )
    preprocessor.run()
