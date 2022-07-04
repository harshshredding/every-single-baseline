from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from mi_rim import *
import pytorch_lightning as pl
from args import args


class SeqLabeler(torch.nn.Module):
    def __init__(self):
        super(SeqLabeler, self).__init__()
        self.num_mechanisms = 1
        self.top_k = 1
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, [768])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, encoding):
        x = encoding
        outputs = self.bert_model(x['input_ids'], return_dict=True)
        outputs = outputs['last_hidden_state'][0]
        rim_input = [outputs]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class SeqLabelerUMLS(torch.nn.Module):
    def __init__(self):
        super(SeqLabelerUMLS, self).__init__()
        self.num_mechanisms = 2
        self.top_k = 2
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_embeddings):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        rim_input = [bert_embeddings, umls_embeddings]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class LITSeqLabeler(pl.LightningModule):
    def __init__(self, num_mechanisms, hidden_size, top_k, num_class):
        super(LITSeqLabeler, self).__init__()
        self.num_class = num_class
        self.rim = MI_RIM('lstm', num_mechanisms, top_k, hidden_size, input_sizes=[768])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
        self.classifier = nn.Linear(128, self.num_class)

    def forward(self, encoding):
        x = encoding
        outputs = self.bert_model(x['input_ids'], return_dict=True)
        outputs = outputs['last_hidden_state'][0]
        rim_input = [outputs]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, encoding, labels):
        x = encoding
        outputs = self.bert_model(x['input_ids'], return_dict=True)
        outputs = outputs['last_hidden_state'][0]
        rim_input = [outputs]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        classifier_output = self.classifier(rim_output_hidden_states)
