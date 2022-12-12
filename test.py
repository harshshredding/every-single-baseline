from util import enumerate_spans, log_tensor
from typing import List
import torch
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from args import args, device
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding
from structs import Anno, TokenData
import train_util
import util
import logging


class SpanBert(torch.nn.Module):
    def __init__(self):
        super(SpanBert, self).__init__()
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.input_dim = 768
        self.num_class = (args['num_types'] * 2) + 1
        self.classifier = nn.Linear(self.input_dim * 2, self.num_class)
        self.endpoint_span_extractor = EndpointSpanExtractor(self.input_dim)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self,
                sample_token_data: List[TokenData],
                sample_annos: List[Anno]
                ):
        """Forward pass
        Args:
            sample_token_data (List[TokenData]): Token data of `one` sample.
            sample_annos (List[Anno]): Annotations of one sample.
        Returns:
            Tensor[shape(batch_size, num_spans, num_classes)] classification of each span
        """
        tokens = util.get_token_strings(sample_token_data)
        offsets_list = util.get_token_offsets(sample_token_data)
        bert_encoding = self.bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
        bert_embeddings = self.bert_model(
            bert_encoding['input_ids'], return_dict=True)
        # SHAPE: (seq_len, 768)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        # SHAPE: (batch_size, seq_len, 768)
        bert_embeddings = torch.unsqueeze(bert_embeddings, 0)
        span_list = enumerate_spans(bert_encoding.word_ids())
        # SHAPE: (batch_size, seq_len, 2)
        spans: torch.Tensor = torch.tensor([span_list])
        # SHAPE: (batch_size, num_spans, endpoint_dim)
        span_embeddings = self.endpoint_span_extractor(bert_embeddings, spans)
        # SHAPE: (batch_size, num_spans, num_classes)
        return self.classifier(span_embeddings)


valid_annos = train_util.get_valid_annos_dict()
print(valid_annos['1440492653229993984'])

# tokens = ["this", "is", "a", "sentence"]
# bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = SpanBert()
# batch_encoding: BatchEncoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
#                                                add_special_tokens=False, truncation=True, max_length=512)
# print(model(batch_encoding).shape)
