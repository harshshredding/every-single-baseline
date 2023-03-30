import torch.nn as nn
import torch
import math
from torch import Tensor

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


class PositionalEncodingBatch(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3
        x = torch.unsqueeze(x, dim=2)
        x = x + self.pe[:x.size(1)]
        x = self.dropout(x)
        x = torch.squeeze(x, dim=2)
        return x
