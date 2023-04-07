from models.all_models import SeqLabelerNoTokenization
from utils.config import ModelConfig, DatasetConfig
import torch.nn as nn
from structs import Anno, Sample
from transformers.tokenization_utils_base import BatchEncoding
import torch
from preamble import *
from utils.model import PositionalEncodingBatch

def get_external_annos_of_type(sample: Sample, anno_type: str):
    return [anno for anno in sample.annos.external if anno.label_type == anno_type]

def get_gazetteer_match_labels(batch_encoding: BatchEncoding, gazetteer_annos: list[Anno], batch_idx: int) -> list[list[int]]:
    labels = [[1, 0] for _ in batch_encoding.tokens(batch_index=batch_idx)]
    char_spans = [
            batch_encoding.token_to_chars(token_idx) 
            for token_idx in range(len(batch_encoding.tokens(batch_index=batch_idx)))
    ]
    for anno in gazetteer_annos:
        start_token_idx = [
                i 
                for i, char_span in enumerate(char_spans)
                if (char_span is not None) and (char_span.start == anno.begin_offset)
        ]  
        end_token_idx = [
                i 
                for i, char_span in enumerate(char_spans)
                if (char_span is not None) and (char_span.end == anno.end_offset)
        ]
        if len(start_token_idx) and len(end_token_idx):
            assert char_spans[start_token_idx[0]].start == anno.begin_offset
            assert char_spans[end_token_idx[0]].end == anno.end_offset
            for i in range(start_token_idx[0], end_token_idx[0] + 1):
                labels[i] = [0, 1]
    assert len(labels) == len(batch_encoding.tokens(batch_index=batch_idx))
    return labels



def get_bert_embeddings_with_external_knowledge_for_batch(
        bert_model,
        encoding: BatchEncoding,
        samples: list[Sample],
        external_feature_type,
        input_dim,
        transformer
    ):
    bert_embeddings_batch = bert_model(encoding['input_ids'], return_dict=True)
    # SHAPE: (batch, seq, emb_dim)
    bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']

    one_hot_labels = []
    for batch_idx, sample in enumerate(samples):
        one_hot_labels.append(
                get_gazetteer_match_labels(
                    batch_encoding=encoding, 
                    gazetteer_annos=get_external_annos_of_type(sample=sample, anno_type=external_feature_type),
                    batch_idx=batch_idx
                )
        )

    one_hot_labels_tensor = torch.tensor(one_hot_labels, device=device)
    assert len(one_hot_labels_tensor.shape) == 3
    assert one_hot_labels_tensor.shape[0] == len(samples)
    assert one_hot_labels_tensor.shape[1] == len(encoding.tokens())
    assert one_hot_labels_tensor.shape[2] == 2

    enriched_embeddings = torch.cat((bert_embeddings_batch, one_hot_labels_tensor), dim=2)
    assert len(enriched_embeddings.shape) == 3
    assert enriched_embeddings.shape[0] == len(samples)
    assert enriched_embeddings.shape[1] == len(encoding.tokens())
    assert enriched_embeddings.shape[2] == input_dim + 2

    enriched_embeddings = transformer(enriched_embeddings)

    return enriched_embeddings







class SeqLabelDefaultExternal(SeqLabelerNoTokenization):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.input_dim + 2), nhead=9)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        assert model_config.external_feature_type is not None
        self.external_feature_type = model_config.external_feature_type
        self.classifier = nn.Linear(self.input_dim + 2, self.num_class)

    
    def get_bert_embeddings_for_batch(self, encoding: BatchEncoding, samples: list[Sample]):
        bert_embeddings_batch = self.bert_model(encoding['input_ids'], return_dict=True)
        # SHAPE: (batch, seq, emb_dim)
        bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']

        one_hot_labels = []
        for batch_idx, sample in enumerate(samples):
            one_hot_labels.append(
                    get_gazetteer_match_labels(
                        batch_encoding=encoding, 
                        gazetteer_annos=get_external_annos_of_type(sample=sample, anno_type=self.external_feature_type),
                        batch_idx=batch_idx
                    )
            )

        one_hot_labels_tensor = torch.tensor(one_hot_labels, device=device)
        assert len(one_hot_labels_tensor.shape) == 3
        assert one_hot_labels_tensor.shape[0] == len(samples)
        assert one_hot_labels_tensor.shape[1] == len(encoding.tokens())
        assert one_hot_labels_tensor.shape[2] == 2

        enriched_embeddings = torch.cat((bert_embeddings_batch, one_hot_labels_tensor), dim=2)
        assert len(enriched_embeddings.shape) == 3
        assert enriched_embeddings.shape[0] == len(samples)
        assert enriched_embeddings.shape[1] == len(encoding.tokens())
        assert enriched_embeddings.shape[2] == self.input_dim + 2

        enriched_embeddings = self.transformer(enriched_embeddings)

        return enriched_embeddings


class SeqLabelDefaultExternalTinyTransformer(SeqLabelDefaultExternal):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.input_dim + 2), nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=1)


class SeqLabelDefaultExternalPos(SeqLabelDefaultExternal):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        self.pos_encoder = PositionalEncodingBatch(d_model=(self.input_dim + 2))


    def get_bert_embeddings_for_batch(self, encoding: BatchEncoding, samples: list[Sample]):
        bert_embeddings_batch = self.bert_model(encoding['input_ids'], return_dict=True)
        # SHAPE: (batch, seq, emb_dim)
        bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']

        one_hot_labels = []
        for batch_idx, sample in enumerate(samples):
            one_hot_labels.append(
                    get_gazetteer_match_labels(
                        batch_encoding=encoding, 
                        gazetteer_annos=get_external_annos_of_type(sample=sample, anno_type=self.external_feature_type),
                        batch_idx=batch_idx
                    )
            )

        one_hot_labels_tensor = torch.tensor(one_hot_labels, device=device)
        assert len(one_hot_labels_tensor.shape) == 3
        assert one_hot_labels_tensor.shape[0] == len(samples)
        assert one_hot_labels_tensor.shape[1] == len(encoding.tokens())
        assert one_hot_labels_tensor.shape[2] == 2

        enriched_embeddings = torch.cat((bert_embeddings_batch, one_hot_labels_tensor), dim=2)
        assert len(enriched_embeddings.shape) == 3
        assert enriched_embeddings.shape[0] == len(samples)
        assert enriched_embeddings.shape[1] == len(encoding.tokens())
        assert enriched_embeddings.shape[2] == self.input_dim + 2

        enriched_embeddings = self.pos_encoder(enriched_embeddings)

        enriched_embeddings = self.transformer(enriched_embeddings)

        return enriched_embeddings



