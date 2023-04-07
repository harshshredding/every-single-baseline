from models.span_batched_no_custom_tok import SpanDefault

from models.all_models import SeqLabelerNoTokenization
from utils.config import ModelConfig, DatasetConfig
import torch.nn as nn
from structs import Anno, Sample
from transformers.tokenization_utils_base import BatchEncoding
import torch
from preamble import *
from utils.model import PositionalEncodingOriginal, EndpointSpanExtractor
from models.seq_label_default_external import get_bert_embeddings_with_external_knowledge_for_batch,\
        get_bert_embeddings_with_external_knowledge_pos_for_batch


class SpanDefaultExternal(SpanDefault):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.input_dim + 2), nhead=9)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        assert model_config.external_feature_type is not None
        self.external_feature_type = model_config.external_feature_type
        self.classifier = nn.Linear(2*(self.input_dim + 2), self.num_class)
        self.endpoint_span_extractor = EndpointSpanExtractor(self.input_dim + 2)


    def get_bert_embeddings_for_batch(self, encoding: BatchEncoding, samples: list[Sample]):
        return get_bert_embeddings_with_external_knowledge_for_batch(
                bert_model=self.bert_model,
                encoding=encoding,
                samples=samples,
                external_feature_type=self.external_feature_type,
                input_dim=self.input_dim,
                transformer=self.transformer
        )


class SpanDefaultExternalPos(SpanDefaultExternal):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        self.pos_encoder = PositionalEncodingOriginal(d_model=(self.input_dim + 2))


    def get_bert_embeddings_for_batch(self, encoding: BatchEncoding, samples: list[Sample]):
        return get_bert_embeddings_with_external_knowledge_pos_for_batch(
                bert_model=self.bert_model,
                encoding=encoding,
                samples=samples,
                external_feature_type=self.external_feature_type,
                input_dim=self.input_dim,
                transformer=self.transformer,
                pos_encoder=self.pos_encoder
        )



