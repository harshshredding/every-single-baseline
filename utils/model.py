import torch.nn as nn
import torch
import math
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

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


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncodingOriginal(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingOriginal, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_bert_embeddings_for_batch(bert_model, encoding: BatchEncoding):
    bert_embeddings_batch = bert_model(encoding['input_ids'], return_dict=True)
    # SHAPE: (batch, seq, emb_dim)
    bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']
    return bert_embeddings_batch
