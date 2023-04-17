from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils.model import ModelClaC, PredictionsBatch, get_bert_encoding_for_batch
from utils.config import ModelConfig, DatasetConfig
import torch
import torch.nn as nn
from structs import Sample

class ClassifierDefault(ModelClaC):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(ClassifierDefault, self).__init__(model_config, dataset_config)
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = len(all_types)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.loss_function = nn.CrossEntropyLoss()
        self.type_to_idx = {type_name: i for i, type_name in enumerate(all_types)}
        self.idx_to_type = {i: type_name for type_name, i in self.type_to_idx.items()}


    def forward(
        self,
        samples: list[Sample]
    ) -> tuple[torch.Tensor, PredictionsBatch]:
        """
        Forward pass.
        :param samples: batch of samples
        :return: a tuple with loss(a tensor) and the batch of predictions made by the model
        """
        bert_encoding = get_bert_encoding_for_batch(
                samples=samples,
                bert_tokenizer=self.bert_tokenizer,
                model_config=self.model_config
        )
        bert_embeddings = self.bert_model(bert_encoding.input_ids, return_dict=True)
        raise NotImplementedError()
