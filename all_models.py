import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from mi_rim import *
from torch import Tensor
import util
from typing import List
from structs import TokenData, Anno, Sample
from transformers.tokenization_utils_base import BatchEncoding
import train_util
from flair.models.sequence_tagger_utils.crf import CRF
from flair.models.sequence_tagger_utils.viterbi import ViterbiLoss, ViterbiDecoder
from flair.data import Dictionary
from utils.config import DatasetConfig, ModelConfig
from utils.universal import Option, OptionState
from utils.config import get_experiment_config
from pudb import set_trace
from models.span_batched_no_custom_tok import SpanNoTokenizationBatched


class Embedding(nn.Module):
    def __init__(self, emb_dim, vocab_size, initialize_emb, word_to_ix):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim).requires_grad_(False)
        if initialize_emb:
            inv_dic = {v: k for k, v in word_to_ix.items()}
            for key in initialize_emb.keys():
                if key in word_to_ix:
                    ind = word_to_ix[key]
                    self.embedding.weight.data[ind].copy_(torch.from_numpy(initialize_emb[key]))

    def forward(self, input):
        return self.embedding(input)


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#
class PositionalEncoding(nn.Module):
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
        x = torch.unsqueeze(x, dim=1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = torch.squeeze(x, dim=1)
        return x


class SeqLabelerRim(torch.nn.Module):
    def __init__(self, dataset_config):
        super(SeqLabelerRim, self).__init__()
        self.num_mechanisms = 1
        self.top_k = 1
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, [768])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
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
    def __init__(self, dataset_config):
        super(SeqLabelerUMLS, self).__init__()
        self.num_mechanisms = 2
        self.top_k = 2
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_embeddings):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        rim_input = [bert_embeddings, umls_embeddings]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class SeqLabelerAllResources(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(SeqLabelerAllResources, self).__init__()
        self.num_mechanisms = 3
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_indices, pos_indices):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        rim_input = [bert_embeddings, umls_embeddings, pos_embeddings]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class SeqLabelerAllResourcesSmallerTopK(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(SeqLabelerAllResourcesSmallerTopK, self).__init__()
        self.num_mechanisms = 3
        self.top_k = 2
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_indices, pos_indices):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        rim_input = [bert_embeddings, umls_embeddings, pos_embeddings]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class SeqLabelerDisGaz(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(SeqLabelerDisGaz, self).__init__()
        self.num_mechanisms = 4
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20, 2])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_indices, pos_indices, dis_gaz_embedding):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        rim_input = [bert_embeddings, umls_embeddings, pos_embeddings, dis_gaz_embedding]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class SeqLabelerUMLSDisGaz(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(SeqLabelerUMLSDisGaz, self).__init__()
        self.num_mechanisms = 5
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20, 2, 2])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_indices, pos_indices, dis_gaz_embedding, umls_dis_gaz_embedding):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        rim_input = [bert_embeddings, umls_embeddings, pos_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class SeqLabelerUMLSDisGaz3Classes(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(SeqLabelerUMLSDisGaz3Classes, self).__init__()
        self.num_mechanisms = 5
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 3
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k,
                          self.hidden_size, input_sizes=[dataset_config['bert_model_output_dim'], 50, 20, 2, 2])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_indices, pos_indices, dis_gaz_embedding, umls_dis_gaz_embedding):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        rim_input = [bert_embeddings, umls_embeddings, pos_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class Silver3Classes(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(Silver3Classes, self).__init__()
        self.num_mechanisms = 6
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 3
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k,
                          self.hidden_size, input_sizes=[dataset_config['bert_model_output_dim'], 50, 20, 2, 2, 2])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, umls_indices, pos_indices, dis_gaz_embedding, umls_dis_gaz_embedding,
                silver_gaz_embedding):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        rim_input = [bert_embeddings, umls_embeddings, pos_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding,
                     silver_gaz_embedding]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class LightWeightRIM3Classes(torch.nn.Module):
    def __init__(self, dataset_config):
        super(LightWeightRIM3Classes, self).__init__()
        self.num_mechanisms = 4
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 3
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k,
                          self.hidden_size, input_sizes=[dataset_config['bert_model_output_dim'], 2, 2, 2])
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.classifier = nn.Linear(self.num_mechanisms * self.hidden_size, self.num_class)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        x = bert_encoding
        bert_embeddings = self.bert_model(x['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        rim_input = [bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding]
        rim_output = self.rim(rim_input)
        rim_output_hidden_states = rim_output[0]
        return self.classifier(rim_output_hidden_states)


class OneEncoder3Classes(torch.nn.Module):
    def __init__(self, dataset_config):
        super(OneEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1030
        self.num_class = 3
        self.num_heads = 10
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding), 1)
        out = self.encoder(x)
        return self.classifier(out)


class TransformerEncoder3Classes(torch.nn.Module):
    def __init__(self, dataset_config):
        super(TransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1030
        self.num_class = 3
        self.num_heads = 10
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding), 1)
        out = self.encoder(x)
        return self.classifier(out)


class PositionalTransformerEncoder3Classes(torch.nn.Module):
    def __init__(self, dataset_config):
        super(PositionalTransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1030
        self.num_class = 3
        self.num_heads = 10
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding), 1)
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return self.classifier(out)


class SmallPositionalTransformerEncoder3Classes(torch.nn.Module):
    def __init__(self, dataset_config):
        super(SmallPositionalTransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1030
        self.num_class = 3
        self.num_heads = 10
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding), 1)
        x = self.pos_encoder(x)
        out = self.encoder_layer(x)
        return self.classifier(out)


class ComprehensivePositionalTransformerEncoder3Classes(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx, dataset_config):
        super(ComprehensivePositionalTransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1100
        self.num_class = 3
        self.num_heads = 10
        self.umls = Embedding(50, len(umls_pretrained), umls_pretrained, umls_to_idx)
        self.pos = Embedding(20, len(pos_pretrained), pos_pretrained, pos_to_idx)
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding, umls_indices, pos_indices, dis_gaz_embedding, umls_dis_gaz_embedding,
                silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        pos_embeddings = self.pos(pos_indices)
        umls_embeddings = self.umls(umls_indices)
        x = torch.cat((bert_embeddings, pos_embeddings, umls_embeddings,
                       dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding), 1)
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return self.classifier(out)


class PosEncod3ClassesNoSilverNewGaz(torch.nn.Module):
    def __init__(self, dataset_config):
        super(PosEncod3ClassesNoSilverNewGaz, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1030
        self.num_class = 3
        self.num_heads = 10
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        silver_gaz_embedding_zeros = torch.zeros(silver_gaz_embedding.size(), device=device)
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding_zeros), 1)
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return self.classifier(out)


class PosEncod3ClassesNoSilverBig(torch.nn.Module):
    def __init__(self, dataset_config):
        super(PosEncod3ClassesNoSilverBig, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 2568
        self.num_class = 3
        self.num_heads = 8
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        silver_gaz_embedding_zeros = torch.zeros(silver_gaz_embedding.size(), device=device)
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding_zeros,
                       silver_gaz_embedding_zeros), 1)
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return self.classifier(out)


class PosEncod3ClassesNoSilverSpanish(torch.nn.Module):
    def __init__(self, dataset_config):
        super(PosEncod3ClassesNoSilverSpanish, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 776
        self.num_class = 3
        self.num_heads = 8
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        silver_gaz_embedding_zeros = torch.zeros(silver_gaz_embedding.size(), device=device)
        x = torch.cat((bert_embeddings, dis_gaz_embedding, umls_dis_gaz_embedding, silver_gaz_embedding_zeros,
                       silver_gaz_embedding_zeros), 1)
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return self.classifier(out)


class PosEncod3ClassesOnlyRoberta(torch.nn.Module):
    def __init__(self, dataset_config):
        super(PosEncod3ClassesOnlyRoberta, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1024
        self.num_class = 3
        self.num_heads = 8
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(self, bert_encoding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        x = bert_embeddings
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return self.classifier(out)


class OnlyRoberta3Classes(torch.nn.Module):
    def __init__(self, dataset_config):
        super(OnlyRoberta3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.input_dim = 1024
        self.num_class = (dataset_config['num_types'] * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)

    def forward(self, bert_encoding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        return self.classifier(bert_embeddings)


class JustBert3Classes(torch.nn.Module):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(JustBert3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = (dataset_config.num_types * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        label_to_idx, idx_to_label = util.get_bio_label_idx_dicts(all_types, dataset_config)
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.loss_function = nn.CrossEntropyLoss()
        self.dataset_config = dataset_config
        self.model_config = model_config

    def forward(self,
                samples: List[Sample]
                ):
        assert len(samples) == 1
        sample = samples[0]
        tokens = util.get_tokens_from_sample(sample)
        offsets_list = util.get_token_offsets_from_sample(sample)
        bert_encoding = self.bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        predictions_logits = self.classifier(bert_embeddings)
        expanded_labels = train_util.get_bio_labels_from_annos(util.get_token_annos_from_sample(sample),
                                                               bert_encoding,
                                                               sample.annos.gold,
                                                               self.model_config)
        expanded_labels_indices = [self.label_to_idx[label] for label in expanded_labels]
        expanded_labels_tensor = torch.tensor(expanded_labels_indices).to(device)
        loss = self.loss_function(predictions_logits, expanded_labels_tensor)

        predicted_label_indices_expanded = torch.argmax(predictions_logits, dim=1).cpu().detach().numpy()
        predicted_labels = [self.idx_to_label[label_id] for label_id in predicted_label_indices_expanded]
        predicted_spans_token_index = train_util.get_spans_from_bio_seq_labels(predicted_labels, bert_encoding,
                                                                               self.model_config)
        predicted_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1], span[2]) for span in
                                        predicted_spans_token_index]
        predicted_annos = []
        for span_char_offsets, span_token_idx in zip(predicted_spans_char_offsets, predicted_spans_token_index):
            predicted_annos.append(
                Anno(
                    span_char_offsets[0],
                    span_char_offsets[1],
                    span_char_offsets[2],
                    " ".join(tokens[span_token_idx[0]: span_token_idx[1] + 1])
                )
            )
        return loss, [predicted_annos]


class JustBert3ClassesCRF(torch.nn.Module):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(JustBert3ClassesCRF, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = (dataset_config.num_types * 2) + 1
        label_to_idx, idx_to_label = util.get_bio_label_idx_dicts(all_types, dataset_config)
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.flair_dictionary = self._get_flair_label_dictionary()
        self.loss_function = ViterbiLoss(self.flair_dictionary)
        assert len(self.idx_to_label) + 2 == len(self.flair_dictionary)
        self.crf = CRF(self.flair_dictionary, len(self.flair_dictionary), False)
        self.viterbi_decoder = ViterbiDecoder(self.flair_dictionary)
        self.linear = nn.Linear(self.input_dim, len(self.flair_dictionary))
        self.model_config = model_config
        self.dataset_config = dataset_config

    def _get_flair_label_dictionary(self):
        flair_dictionary = Dictionary(add_unk=False)
        for i in range(len(self.idx_to_label)):
            assert flair_dictionary.add_item(str(self.idx_to_label[i])) == i
        flair_dictionary.set_start_stop_tags()
        return flair_dictionary

    def forward(self,
                sample_token_data: List[TokenData],
                sample_annos: List[Anno]
                ):
        tokens = util.get_token_strings(sample_token_data)
        offsets_list = util.get_token_offsets(sample_token_data)
        bert_encoding = self.bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        # SHAPE (seq_len, num_classes)
        features = self.linear(bert_embeddings)
        # SHAPE (1, seq_len, num_classes)
        features = torch.unsqueeze(features, 0)
        features = self.crf(features)
        # TODO: calculate length using tensor

        expanded_labels = train_util.get_bio_labels_from_annos(sample_token_data, bert_encoding, sample_annos,
                                                               self.model_config)
        expanded_labels_indices = [self.label_to_idx[label] for label in expanded_labels]
        expanded_labels_tensor = torch.tensor(expanded_labels_indices).to(device)
        lengths = torch.tensor([len(expanded_labels_tensor)], dtype=torch.long)
        features_tuple = (features, lengths, self.crf.transitions)
        loss = self.loss_function(features_tuple, expanded_labels_tensor)
        predictions, all_tags = self.viterbi_decoder.decode(features_tuple, False, None)
        predicted_label_strings = [label for label, score in predictions[0]]
        predicted_label_indices_expanded = [self.flair_dictionary.get_idx_for_item(label_string) for label_string in
                                            predicted_label_strings]
        predicted_labels = [self.idx_to_label[label_id] for label_id in predicted_label_indices_expanded]
        predicted_spans_token_index = train_util.get_spans_from_bio_seq_labels(predicted_labels, bert_encoding,
                                                                               self.model_config)
        predicted_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1], span[2]) for span in
                                        predicted_spans_token_index]
        predicted_annos = []
        for span_char_offsets, span_token_idx in zip(predicted_spans_char_offsets, predicted_spans_token_index):
            predicted_annos.append(
                Anno(
                    span_char_offsets[0],
                    span_char_offsets[1],
                    span_char_offsets[2],
                    " ".join(tokens[span_token_idx[0]: span_token_idx[1] + 1])
                )
            )
        return loss, predicted_annos


def heuristic_decode(predicted_annos: List[Anno]):
    to_remove = []
    for curr_anno in predicted_annos:
        overlapping_annos = [
            anno for anno in predicted_annos
            if not ((curr_anno.begin_offset >= anno.end_offset) or (anno.begin_offset >= curr_anno.end_offset))
        ]
        max_confidence = max([anno.features['confidence_value'] for anno in overlapping_annos])
        if curr_anno.features['confidence_value'] < max_confidence:
            to_remove.append(curr_anno)
    return [anno for anno in predicted_annos if anno not in to_remove]


class SpanBert(torch.nn.Module):
    def __init__(self, all_types: List[str], model_config: ModelConfig):
        super(SpanBert, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = len(all_types) + 1
        self.classifier = nn.Linear(self.input_dim * 2, self.num_class)
        self.endpoint_span_extractor = EndpointSpanExtractor(self.input_dim)
        self.loss_function = nn.CrossEntropyLoss()
        self.type_to_idx = {type_name: i for i, type_name in enumerate(all_types)}
        # Add NO_TYPE type which represents "no annotation"
        self.type_to_idx['NO_TYPE'] = len(self.type_to_idx)
        self.idx_to_type = {i: type_name for type_name, i in self.type_to_idx.items()}
        assert len(self.type_to_idx) == self.num_class, "Num of classes should be equal to num of types"

    def forward(
            self,
            samples: List[Sample]
    ):
        assert len(samples) == 1, "Can only handle one sample at a time :("
        sample = samples[0]
        tokens = util.get_tokens_from_sample(sample)
        token_annos = util.get_token_annos_from_sample(sample)
        gold_token_level_annos = util.get_token_level_spans(token_annos, sample.annos.gold)
        bert_encoding = self.bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
        gold_sub_token_level_annos = util.get_sub_token_level_spans(gold_token_level_annos, bert_encoding)
        bert_embeddings = self.bert_model(
            bert_encoding['input_ids'], return_dict=True)
        # SHAPE: (seq_len, 768)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        # SHAPE: (batch_size, seq_len, 768)
        bert_embeddings = torch.unsqueeze(bert_embeddings, 0)
        all_possible_spans_list = util.enumerate_spans(bert_encoding.word_ids())
        all_possible_spans_labels = self.label_all_possible_spans(all_possible_spans_list, gold_sub_token_level_annos)
        # SHAPE: (batch_size, num_spans)
        all_possible_spans_labels = torch.tensor([all_possible_spans_labels], device=device)
        # SHAPE: (batch_size, seq_len, 2)
        all_possible_spans_tensor: torch.Tensor = torch.tensor([all_possible_spans_list], device=device)
        # SHAPE: (batch_size, num_spans, endpoint_dim)
        span_embeddings = self.endpoint_span_extractor(bert_embeddings, all_possible_spans_tensor)
        # SHAPE: (batch_size, num_spans, num_classes)
        predicted_all_possible_spans_logits = self.classifier(span_embeddings)
        loss = self.loss_function(torch.squeeze(predicted_all_possible_spans_logits, 0),
                                  torch.squeeze(all_possible_spans_labels, 0))
        predicted_annos = self.get_predicted_annos(
            predicted_all_possible_spans_logits,
            all_possible_spans_list,
            bert_encoding,
            token_annos
        )
        # predicted_annos = self.heuristic_decode(predicted_annos)
        return loss, [predicted_annos]

    def get_predicted_annos(
            self,
            predicted_all_possible_spans_logits,
            all_possible_spans_list,
            bert_encoding: BatchEncoding,
            token_annos: List[Anno]
    ) -> List[Anno]:
        ret = []
        # SHAPE: (num_spans)
        pred_all_possible_spans_type_indices_list = torch \
            .argmax(torch.squeeze(predicted_all_possible_spans_logits, 0), dim=1) \
            .cpu() \
            .detach().numpy()
        # SHAPE: (num_spans)
        pred_all_possible_spans_max_values = torch \
            .max(torch.squeeze(predicted_all_possible_spans_logits, 0), dim=1) \
            .values \
            .cpu() \
            .detach().numpy()

        assert len(pred_all_possible_spans_type_indices_list.shape) == 1
        for i, span_type_idx in enumerate(pred_all_possible_spans_type_indices_list):
            if span_type_idx != self.type_to_idx['NO_TYPE']:
                # get sub-token level spans
                span_start_subtoken_idx = all_possible_spans_list[i][0]
                span_end_subtoken_idx = all_possible_spans_list[i][1]  # inclusive
                # get token level spans
                span_start_token_idx = bert_encoding.token_to_word(span_start_subtoken_idx)
                span_end_token_idx = bert_encoding.token_to_word(span_end_subtoken_idx)
                assert span_start_token_idx is not None
                assert span_end_token_idx is not None
                # get word char offsets
                span_start_char_offset = token_annos[span_start_token_idx].begin_offset
                span_end_char_offset = token_annos[span_end_token_idx].end_offset
                all_token_strings = [token_anno.extraction for token_anno in token_annos]
                span_text = " ".join(all_token_strings[span_start_token_idx: span_end_token_idx + 1])
                ret.append(
                    Anno(
                        span_start_char_offset,
                        span_end_char_offset,
                        self.idx_to_type[span_type_idx],
                        span_text,
                        {"confidence_value": pred_all_possible_spans_max_values[i]}
                    )
                )
        return ret

    def label_all_possible_spans(self, all_possible_spans_list, sub_token_level_annos):
        all_possible_spans_labels = []
        for span in all_possible_spans_list:
            corresponding_anno_list = [anno for anno in sub_token_level_annos if
                                       (anno[0] == span[0]) and (anno[1] == (span[1] + 1))]  # spans are inclusive
            if len(corresponding_anno_list):
                if len(corresponding_anno_list) > 1:
                    print("WARN: Didn't expect multiple annotations to match one span")
                corresponding_anno = corresponding_anno_list[0]
                all_possible_spans_labels.append(self.type_to_idx[corresponding_anno[2]])
            else:
                all_possible_spans_labels.append(self.type_to_idx["NO_TYPE"])
        assert len(all_possible_spans_labels) == len(all_possible_spans_list)
        return all_possible_spans_labels


class SpanBertNounPhrase(SpanBert):
    def __init__(self, all_types: List[str], model_config: ModelConfig):
        super().__init__(all_types, model_config)
        # Cannot use super's classifier because we are concatenating
        # noun-phrase information(bit) to span embeddings.
        self.classifier = nn.Linear(self.input_dim * 2 + 1, self.num_class)

    def forward(self, sample: Sample):
        """Forward pass
        Args:
            sample_token_data (List[TokenData]): Token data of `one` sample.
            sample_annos (List[Anno]): Annotations of one sample.
        Returns:
            Tensor[shape(batch_size, num_spans, num_classes)] classification of each span
        """
        tokens = util.get_tokens_from_sample(sample)
        token_annos = util.get_token_annos_from_sample(sample)
        gold_spans_token_level = util.get_token_level_spans(token_annos=token_annos, annos_to_convert=sample.annos.gold)
        noun_phrase_annos = util.get_annos_of_type_from_collection('NounPhrase', sample.annos)
        noun_phrase_token_level = util.get_token_level_spans(token_annos=token_annos,
                                                             annos_to_convert=noun_phrase_annos)
        bert_encoding = self.bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
        gold_spans_sub_token_level = util.get_sub_token_level_spans(gold_spans_token_level, bert_encoding)
        noun_phrase_sub_token_level = util.get_sub_token_level_spans(noun_phrase_token_level, bert_encoding)
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        # SHAPE: (seq_len, 768)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        # SHAPE: (batch_size, seq_len, 768)
        bert_embeddings = torch.unsqueeze(bert_embeddings, 0)
        # SHAPE: (num_spans)
        all_possible_spans_list = util.enumerate_spans(bert_encoding.word_ids())
        gold_labels_all_spans = self.label_all_possible_spans(all_possible_spans_list, gold_spans_sub_token_level)
        noun_phrase_labels_all_spans = self.get_noun_phrase_labels_for_all_spans(all_possible_spans_list,
                                                                                 noun_phrase_sub_token_level)
        # SHAPE: (batch_size, num_spans)
        gold_labels_all_spans = torch.tensor([gold_labels_all_spans], device=device)
        # SHAPE: (batch_size, num_spans, 1)
        noun_phrase_labels_all_spans = torch.tensor([noun_phrase_labels_all_spans], device=device).unsqueeze(2)
        # SHAPE: (batch_size, seq_len, 2)
        all_possible_spans_tensor: torch.Tensor = torch.tensor([all_possible_spans_list], device=device)
        # SHAPE: (batch_size, num_spans, endpoint_dim)
        span_embeddings = self.endpoint_span_extractor(bert_embeddings, all_possible_spans_tensor)
        # SHAPE: (batch_size, num_spans, endpoint_dim + 1)
        span_embeddings = torch.cat((span_embeddings, noun_phrase_labels_all_spans), dim=2)
        # SHAPE: (batch_size, num_spans, num_classes)
        predicted_all_possible_spans_logits = self.classifier(span_embeddings)
        loss = self.loss_function(torch.squeeze(predicted_all_possible_spans_logits, 0),
                                  torch.squeeze(gold_labels_all_spans, 0))
        predicted_annos = self.get_predicted_annos(
            predicted_all_possible_spans_logits,
            all_possible_spans_list,
            bert_encoding,
            token_annos
        )
        return loss, predicted_annos

    def get_noun_phrase_labels_for_all_spans(self, all_possible_spans_list, noun_phrase_spans_sub_token):
        for noun_phrase_span in noun_phrase_spans_sub_token:
            assert noun_phrase_span[2] == 'NounPhrase'
        noun_phrase_labels = []
        for span in all_possible_spans_list:
            corresponding_anno_list = [anno for anno in noun_phrase_spans_sub_token
                                       if (anno[0] == span[0]) and (anno[1] == (span[1] + 1))]  # spans are inclusive
            if len(corresponding_anno_list):
                if len(corresponding_anno_list) > 1:
                    print("WARN: Didn't expect multiple annotations to match one span")
                noun_phrase_labels.append(1)
            else:
                noun_phrase_labels.append(0)
        assert len(noun_phrase_labels) == len(all_possible_spans_list)
        return noun_phrase_labels


class SpanBertSpanWidthEmbedding(SpanBert):
    def __init__(self, all_types: List[str], model_config: ModelConfig):
        super(SpanBertSpanWidthEmbedding, self).__init__(all_types=all_types, model_config=model_config)
        self.endpoint_span_extractor = EndpointSpanExtractor(
            self.input_dim,
            span_width_embedding_dim=16,
            num_width_embeddings=512
        )
        self.classifier = nn.Linear((self.input_dim * 2) + 16, self.num_class)


class SeqLabelerBatched(torch.nn.Module):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(SeqLabelerBatched, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = (dataset_config.num_types * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        label_to_idx, idx_to_label = util.get_bio_label_idx_dicts(all_types, dataset_config)
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.loss_function = nn.CrossEntropyLoss()
        self.dataset_config = dataset_config
        self.model_config = model_config

    def forward(self,
                samples: List[Sample]
                ):
        assert isinstance(samples, list)
        tokens_for_batch = util.get_tokens_from_batch(samples)
        offsets_list_for_batch = util.get_token_offsets_from_batch(samples)
        bert_encoding_for_batch = self.bert_tokenizer(tokens_for_batch, return_tensors="pt", is_split_into_words=True,
                                                      add_special_tokens=False, truncation=True, padding=True,
                                                      max_length=512) \
            .to(device)
        bert_embeddings_batch = self.bert_model(bert_encoding_for_batch['input_ids'], return_dict=True)
        # SHAPE: (batch, seq, emb_dim)
        bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']
        predictions_logits_batch = self.classifier(bert_embeddings_batch)
        expanded_labels_batch = train_util.get_bio_labels_from_annos_batch(
            util.get_token_annos_from_batch(samples),
            bert_encoding_for_batch,
            [sample.annos.gold for sample in samples]
        )
        expanded_labels_indices_batch = [
            [self.label_to_idx[label] for label in expanded_labels]
            for expanded_labels in expanded_labels_batch
        ]
        expanded_labels_tensor_batch = torch.tensor(expanded_labels_indices_batch).to(device)

        loss = self.loss_function(torch.permute(predictions_logits_batch, (0, 2, 1)), expanded_labels_tensor_batch)

        predicted_label_indices_expanded_batch = torch.argmax(predictions_logits_batch, dim=2).cpu().detach().numpy()
        predicted_labels_batch = [
            [self.idx_to_label[label_id] for label_id in predicted_label_indices_expanded]
            for predicted_label_indices_expanded in predicted_label_indices_expanded_batch
        ]
        predicted_spans_token_index_batch = [
            train_util.get_spans_from_bio_seq_labels(predicted_labels, bert_encoding_for_batch, batch_idx=batch_idx)
            for batch_idx, predicted_labels in enumerate(predicted_labels_batch)
        ]
        predicted_spans_char_offsets_batch = [
            [(offsets_list_for_batch[batch_idx][span[0]][0], offsets_list_for_batch[batch_idx][span[1]][1], span[2])
             for span in predicted_spans_token_index
             ]
            for batch_idx, predicted_spans_token_index in enumerate(predicted_spans_token_index_batch)
        ]
        predicted_annos_batch = []
        assert len(predicted_spans_char_offsets_batch) == len(tokens_for_batch)
        for predicted_spans_char_offsets, predicted_spans_token_index, tokens in zip(predicted_spans_char_offsets_batch,
                                                                                     predicted_spans_token_index_batch,
                                                                                     tokens_for_batch
                                                                                     ):
            predicted_annos = []
            for span_char_offsets, span_token_idx in zip(predicted_spans_char_offsets, predicted_spans_token_index):
                predicted_annos.append(
                    Anno(
                        span_char_offsets[0],
                        span_char_offsets[1],
                        span_char_offsets[2],
                        " ".join(tokens[span_token_idx[0]: span_token_idx[1] + 1])
                    )
                )
            predicted_annos_batch.append(predicted_annos)
        return loss, predicted_annos_batch


class SeqLabelerNoTokenization(torch.nn.Module):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(SeqLabelerNoTokenization, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = (dataset_config.num_types * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        label_to_idx, idx_to_label = util.get_bio_label_idx_dicts(all_types, dataset_config)
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.loss_function = nn.CrossEntropyLoss()
        self.dataset_config = dataset_config
        self.model_config = model_config

    def get_bert_encoding_for_batch(self, samples: List[Sample]) -> transformers.BatchEncoding:
        batch_of_sample_texts = [sample.text for sample in samples]
        bert_encoding_for_batch = self.bert_tokenizer(batch_of_sample_texts, return_tensors="pt",
                                                      is_split_into_words=False,
                                                      add_special_tokens=True, truncation=True, padding=True,
                                                      max_length=512).to(device)
        return bert_encoding_for_batch

    def get_bert_embeddings_for_batch(self, encoding: transformers.BatchEncoding):
        bert_embeddings_batch = self.bert_model(encoding['input_ids'], return_dict=True)
        # SHAPE: (batch, seq, emb_dim)
        bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']
        return bert_embeddings_batch

    def get_token_annos_batch(self, bert_encoding, expected_batch_size) -> List[List[Option[Anno]]]:
        token_ids_matrix = bert_encoding['input_ids']
        batch_size = len(token_ids_matrix)
        num_tokens = len(token_ids_matrix[0])
        assert batch_size == expected_batch_size
        token_annos_batch: List[List[Option[Anno]]] = []
        for batch_idx in range(batch_size):
            char_spans: List[Option[transformers.CharSpan]] = [
                Option(bert_encoding.token_to_chars(batch_or_token_index=batch_idx, token_index=token_idx))
                for token_idx in range(num_tokens)
            ]

            token_annos_batch.append(
                [
                    Option(Anno(begin_offset=span.get_value().start, end_offset=span.get_value().end,
                                label_type='BertTokenAnno', extraction=None))
                    if span.state == OptionState.Something else Option(None)
                    for span in char_spans
                ]
            )
        return token_annos_batch

    def forward(self,
                samples: List[Sample]
                ):
        assert isinstance(samples, list)
        # encoding helps manage tokens created by bert
        bert_encoding_for_batch = self.get_bert_encoding_for_batch(samples)
        # SHAPE (batch_size, seq_len, bert_emb_len)
        bert_embeddings_batch = self.get_bert_embeddings_for_batch(bert_encoding_for_batch)
        predictions_logits_batch = self.classifier(bert_embeddings_batch)

        gold_labels_batch = train_util.get_bio_labels_for_bert_tokens_batch(
            self.get_token_annos_batch(bert_encoding_for_batch, len(samples)),
            [sample.annos.gold for sample in samples]
        )
        assert len(gold_labels_batch) == len(samples)  # labels for each sample in batch
        assert len(gold_labels_batch[0]) == bert_embeddings_batch.shape[1]  # same num labels as tokens

        gold_label_indices = [
            [self.label_to_idx[label] for label in gold_labels]
            for gold_labels in gold_labels_batch
        ]
        gold_label_indices = torch.tensor(gold_label_indices).to(device)

        loss = self.loss_function(
            torch.permute(predictions_logits_batch, (0, 2, 1)),
            gold_label_indices
        )

        predicted_label_indices_batch = torch.argmax(predictions_logits_batch, dim=2).cpu().detach().numpy()
        predicted_labels_batch = [
            [self.idx_to_label[label_id] for label_id in predicted_label_indices]
            for predicted_label_indices in predicted_label_indices_batch
        ]
        predicted_annos_batch: List[List[Anno]] = [
            util.get_annos_from_bio_labels(
                prediction_labels=predicted_labels,
                batch_encoding=bert_encoding_for_batch,
                batch_idx=batch_idx,
                sample_text=samples[batch_idx].text
            )
            for batch_idx, predicted_labels in enumerate(predicted_labels_batch)
        ]
        return loss, predicted_annos_batch


