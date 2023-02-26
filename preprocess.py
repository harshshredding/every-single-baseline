from structs import *
import util
from abc import ABC, abstractmethod
from annotators import Annotator, TokenAnnotator, SlidingWindowAnnotator
from preamble import *
from pydoc import locate
from enum import Enum
from typing import Type


class PreprocessorRunType(Enum):
    production = 0
    dry_run = 1


class Preprocessor(ABC):
    """
    An abstraction for preprocessing a dataset.
    """

    def __init__(
            self,
            dataset_split: DatasetSplit,
            preprocessor_type: str,
            dataset: Dataset,
            annotators: List[Annotator] = [],
            run_mode: PreprocessorRunType = PreprocessorRunType.production
    ) -> None:
        """
        Creates a preprocessor configured with some file paths
        that represent its output locations.

        Args:
            preprocessor_type: the type of preprocessor (mention details like annotators)
            dataset: the name of the dataset we are preprocessing
            annotators: the list of annotators we need to run on the dataset
            run_mode: In production-mode, all samples are preprocessed, and in sample-mode only
                the first 300 samples are preprocessed.
        """
        super().__init__()
        self.preprocessor_name = preprocessor_type
        self.preprocessor_full_name = f"{dataset.name}_{dataset_split.name}_{preprocessor_type}_{run_mode.name}"
        self.data_folder_path = f"./preprocessed_data"
        self.visualization_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_visualization.bdocjs"
        self.samples_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_samples.json"
        self.entity_types_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_types.txt"
        self.samples: List[Sample] | None = None
        self.annotators = annotators
        self.dataset_split = dataset_split
        self.preprocessor_type = preprocessor_type
        self.dataset = dataset
        self.run_mode = run_mode
        self.print_info()

    def print_info(self):
        print("\n\n------ INFO --------")
        print(blue("Preprocessor Name:"), green(self.preprocessor_full_name))
        print(blue("Run Mode:"), green(self.run_mode.name))
        print(blue("Dataset:"), green(self.dataset.name))
        print("--------INFO----------\n\n")

    def run_annotation_pipeline(self):
        """
        All samples are annotated by the given annotators in a
        defined sequence (some annotator depend on others).
        """
        assert self.samples is not None
        for annotator in self.annotators:
            self.samples = annotator.annotate(self.samples)

    def get_samples_cached(self) -> List[Sample]:
        """
        We cache `Samples` after extracting them from raw data.
        """
        if self.samples is None:
            print(red("Creating Cache of Samples"))
            self.samples = self.get_samples()
            if self.run_mode == PreprocessorRunType.dry_run:
                print(blue("Selecting first 300."))
                self.samples = self.samples[:300]
            self.run_annotation_pipeline()
        else:
            print(green("using cache"))
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


def preprocess_train_and_valid_custom_tokens(preprocessor_module_name: str, preprocessor_name: str,
                                             preprocessor_type='vanilla'):
    preprocessor_class = locate(f"preprocessors.{preprocessor_module_name}.{preprocessor_name}")
    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.valid,
        preprocessor_type=preprocessor_type,
        annotators=[TokenAnnotator()]
    )
    preprocessor.run()

    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.train,
        preprocessor_type=preprocessor_type,
        annotators=[TokenAnnotator()]
    )
    preprocessor.run()


def get_preprocessor_class(preprocessor_module_name, preprocessor_name) -> Type[Preprocessor]:
    return locate(f"preprocessors.{preprocessor_module_name}.{preprocessor_name}")


def preprocess_train_and_valid_vanilla(
        preprocessor_module_name: str,
        preprocessor_name: str,
        preprocessor_type: str,
        dataset: Dataset,
        run_mode: PreprocessorRunType = PreprocessorRunType.production,
):
    preprocessor_class = get_preprocessor_class(preprocessor_module_name, preprocessor_name)

    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.valid,
        preprocessor_type=preprocessor_type,
        annotators=[],
        run_mode=run_mode,
        dataset=dataset
    )
    preprocessor.run()

    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.train,
        preprocessor_type=preprocessor_type,
        annotators=[],
        run_mode=run_mode,
        dataset=dataset
    )
    preprocessor.run()


def preprocess_vanilla(
        preprocessor_module_name: str,
        preprocessor_name: str,
        preprocessor_type: str,
        dataset: Dataset,
        run_mode: PreprocessorRunType,
        dataset_splits: List[DatasetSplit],
):
    assert len(dataset_splits)
    preprocessor_class = get_preprocessor_class(preprocessor_module_name, preprocessor_name)
    for dataset_split in dataset_splits:
        preprocessor = preprocessor_class(
            dataset_split=dataset_split,
            preprocessor_type=preprocessor_type,
            annotators=[],
            run_mode=run_mode,
            dataset=dataset
        )
        preprocessor.run()


def preprocess_train_and_valid_with_window(
        preprocessor_module_name: str,
        preprocessor_name: str,
        preprocessor_type: str,
        dataset: Dataset,
        run_mode: PreprocessorRunType = PreprocessorRunType.production,
        window_size: int = 100,
        stride_size: int = 50
):
    preprocessor_class = get_preprocessor_class(preprocessor_module_name, preprocessor_name)

    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.valid,
        preprocessor_type=preprocessor_type,
        annotators=[SlidingWindowAnnotator(window_size=window_size, stride=stride_size)],
        run_mode=run_mode,
        dataset=dataset
    )
    preprocessor.run()

    preprocessor = preprocessor_class(
        dataset_split=DatasetSplit.train,
        preprocessor_type=preprocessor_type,
        annotators=[SlidingWindowAnnotator(window_size=window_size, stride=stride_size)],
        run_mode=run_mode,
        dataset=dataset
    )
    preprocessor.run()
