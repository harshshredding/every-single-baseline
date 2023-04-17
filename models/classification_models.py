from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils.model import ClassificationPredictions, ModelClaC, SeqLabelPredictions, get_bert_encoding_for_batch
from utils.config import ModelConfig, DatasetConfig
import torch
import torch.nn as nn
from structs import Sample
from preamble import *

def check_samples_for_meta_labels(samples: list[Sample]):
    for sample in samples:
        assert len(sample.annos.gold) == 1
        assert sample.annos.gold[0].label_type in ['correct', 'incorrect']

def get_gold_meta_labels(samples: list[Sample], type_to_idx: dict[str, int]):
    return [type_to_idx[sample.annos.gold[0].label_type] 
            for sample in samples]

class MetaDefault(ModelClaC):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(MetaDefault, self).__init__(model_config, dataset_config)
        assert set(all_types) == set(['correct', 'incorrect'])
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = len(all_types)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.loss_function = nn.CrossEntropyLoss()
        self.type_to_idx = {type_name: i for i, type_name in enumerate(all_types)}
        self.idx_to_type = {i: type_name for type_name, i in self.type_to_idx.items()}
        assert len(self.type_to_idx) == 2


    def forward(
        self,
        samples: list[Sample]
    ) -> tuple[torch.Tensor, SeqLabelPredictions|ClassificationPredictions]:
        """
        Forward pass.
        :param samples: batch of samples
        :return: a tuple with loss(a tensor) and the batch of predictions made by the model
        """
        check_samples_for_meta_labels(samples=samples)

        bert_encoding = get_bert_encoding_for_batch(
                samples=samples,
                bert_tokenizer=self.bert_tokenizer,
                model_config=self.model_config
        )

        bert_embeddings = self.bert_model(bert_encoding.input_ids, return_dict=True)

        cls_embeddings: torch.Tensor = bert_embeddings.pooler_output        
        assert tensor_shape(cls_embeddings) == [len(samples), self.input_dim]

        predictions_logits = self.classifier(cls_embeddings)
        assert tensor_shape(predictions_logits) == [len(samples), 2]

        gold_labels = get_gold_meta_labels(
                    samples=samples,
                    type_to_idx=self.type_to_idx
                ) 
        assert len(gold_labels) == len(samples)
        gold_labels = torch.tensor(gold_labels).to(device)
        assert tensor_shape(gold_labels) == [2]
 
        loss = self.loss_function(
            predictions_logits,
            gold_labels
        )

        prediction_indices = torch.argmax(predictions_logits, dim=1).cpu().detach().numpy()
        assert len(prediction_indices) == len(samples)

        predicted_labels = [self.idx_to_type[prediction_idx] for prediction_idx in prediction_indices]
        assert len(predicted_labels) == len(samples)

        return loss, predicted_labels




