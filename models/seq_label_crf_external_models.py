from models.seq_label_crf_models import SeqLabelerDefaultCRF
from utils.config import ModelConfig, DatasetConfig
import torch.nn as nn
from utils.model import PositionalEncodingOriginal
from models.seq_label_default_external import get_bert_embeddings_with_external_knowledge_pos_for_batch
from transformers.tokenization_utils_base import BatchEncoding
from structs import Sample

# self.linear = nn.Linear(self.input_dim, len(self.flair_dictionary))

class SeqLabelerCrfPos(SeqLabelerDefaultCRF):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        # OVERRIDE
        self.linear = nn.Linear(self.input_dim + 2, len(self.flair_dictionary))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.input_dim + 2), nhead=9)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        assert model_config.external_feature_type is not None
        self.external_feature_type = model_config.external_feature_type
        
        self.pos_encoder = PositionalEncodingOriginal(d_model=(self.input_dim + 2))


    def get_bert_embeddings_for_batch(self, encoding: BatchEncoding, samples: list[Sample]):
        return get_bert_embeddings_with_external_knowledge_pos_for_batch(
                bert_model=self.bert_model,
                encoding=encoding,
                samples=samples,
                external_feature_type=self.external_feature_type,
                input_dim=self.input_dim,
                transformer=self.transformer,
                pos_encoder=self.pos_encoder)



    







