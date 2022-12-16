from util import enumerate_spans, log_tensor, get_token_level_spans, get_sub_token_level_spans
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
        self.num_class = args['num_types'] + 1
        self.classifier = nn.Linear(self.input_dim * 2, self.num_class)
        self.endpoint_span_extractor = EndpointSpanExtractor(self.input_dim)
        self.loss_function = nn.CrossEntropyLoss()
        all_types = util.get_all_types(args['types_file_path'])
        self.type_to_idx = {type_name: i for i, type_name in enumerate(all_types)}
        # Add NO_TYPE type which represents "no annotation"
        self.type_to_idx['NO_TYPE'] = len(self.type_to_idx)
        assert len(self.type_to_idx) == self.num_class, "Num of classes should be equal to num of types"

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
        token_level_annos = get_token_level_spans(sample_token_data, sample_annos)
        bert_encoding = self.bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
        sub_token_level_annos = get_sub_token_level_spans(token_level_annos, bert_encoding)
        bert_embeddings = self.bert_model(
            bert_encoding['input_ids'], return_dict=True)
        # SHAPE: (seq_len, 768)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        # SHAPE: (batch_size, seq_len, 768)
        bert_embeddings = torch.unsqueeze(bert_embeddings, 0)
        all_possible_spans_list = enumerate_spans(bert_encoding.word_ids())
        all_possible_spans_labels = self.label_all_possible_spans(all_possible_spans_list, sub_token_level_annos)
        # SHAPE: (batch_size, num_spans)
        all_possible_spans_labels = torch.tensor([all_possible_spans_labels])
        # SHAPE: (batch_size, seq_len, 2)
        all_possible_spans_tensor: torch.Tensor = torch.tensor([all_possible_spans_list])
        # SHAPE: (batch_size, num_spans, endpoint_dim)
        span_embeddings = self.endpoint_span_extractor(bert_embeddings, all_possible_spans_tensor)
        # SHAPE: (batch_size, num_spans, num_classes)
        predicted_all_possible_spans = self.classifier(span_embeddings)
        loss = self.loss_function(torch.squeeze(predicted_all_possible_spans, 0),
                                  torch.squeeze(all_possible_spans_labels, 0))
        return loss

    def label_all_possible_spans(self, all_possible_spans_list, sub_token_level_annos):
        all_possible_spans_labels = []
        for span in all_possible_spans_list:
            corresponding_anno_list = [anno for anno in sub_token_level_annos if
                                       (anno[0] == span[0]) and (anno[1] == span[1])]
            if len(corresponding_anno_list):
                assert len(corresponding_anno_list) == 1, "Don't expect multiple annotations to match one span"
                corresponding_anno = corresponding_anno_list[0]
                all_possible_spans_labels.append(self.type_to_idx[corresponding_anno[2]])
            else:
                all_possible_spans_labels.append(self.type_to_idx["NO_TYPE"])
        assert len(all_possible_spans_labels) == len(all_possible_spans_list)
        return all_possible_spans_labels


# valid_tokens = train_util.get_valid_tokens()
# valid_annos = train_util.get_valid_annos_dict()
# sample_annos = valid_annos['1440492653229993984']
# sample_token_data = valid_tokens['1440492653229993984']
# model = SpanBert()
# loss = model(sample_token_data, sample_annos)
# loss.backward()
#
tokens = ["a-complex-token", "is", "a", "sentence"]
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
batch_encoding: BatchEncoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                               add_special_tokens=False, truncation=True, max_length=512)
print(batch_encoding.word_ids())
for word_idx in range(len(tokens)):
    print(batch_encoding.word_to_tokens(word_idx))
print(batch_encoding.word_to_tokens(4))
