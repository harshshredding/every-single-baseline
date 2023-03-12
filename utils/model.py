import torch.nn

from structs import Anno, Sample
from typing import List
from abc import ABC, abstractmethod
from utils.config import ModelConfig, DatasetConfig

PredictionsBatch = List[List[Anno]]


class ModelClaC(ABC, torch.nn.Module):
    """
    The model abstraction used by the framework. Every model
    should inherit this abstraction.
    """

    def __init__(
            self,
            model_config: ModelConfig,
            dataset_config: DatasetConfig
    ):
        super(ModelClaC, self).__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config

    @abstractmethod
    def forward(
            self,
            samples: List[Sample]
    ) -> tuple[torch.Tensor, PredictionsBatch]:
        """
        Forward pass.
        :param samples: batch of samples
        :return: a tuple with loss(a tensor) and the batch of predictions made by the model
        """

