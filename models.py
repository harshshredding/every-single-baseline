from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from mi_rim import *
from args import args
from torch import Tensor


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


class SeqLabelerAllResources(torch.nn.Module):
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(SeqLabelerAllResources, self).__init__()
        self.num_mechanisms = 3
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(SeqLabelerAllResourcesSmallerTopK, self).__init__()
        self.num_mechanisms = 3
        self.top_k = 2
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(SeqLabelerDisGaz, self).__init__()
        self.num_mechanisms = 4
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20, 2])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(SeqLabelerUMLSDisGaz, self).__init__()
        self.num_mechanisms = 5
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 2
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k, self.hidden_size, input_sizes=[768, 50, 20, 2, 2])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(SeqLabelerUMLSDisGaz3Classes, self).__init__()
        self.num_mechanisms = 5
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 3
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k,
                          self.hidden_size, input_sizes=[args['bert_model_output_dim'], 50, 20, 2, 2])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(Silver3Classes, self).__init__()
        self.num_mechanisms = 6
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 3
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k,
                          self.hidden_size, input_sizes=[args['bert_model_output_dim'], 50, 20, 2, 2, 2])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(LightWeightRIM3Classes, self).__init__()
        self.num_mechanisms = 4
        self.top_k = 3
        self.hidden_size = 128
        self.num_class = 3
        self.rim = MI_RIM('lstm', self.num_mechanisms, self.top_k,
                          self.hidden_size, input_sizes=[args['bert_model_output_dim'], 2, 2, 2])
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(OneEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(TransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(PositionalTransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(SmallPositionalTransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self, umls_pretrained, umls_to_idx, pos_pretrained, pos_to_idx):
        super(ComprehensivePositionalTransformerEncoder3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(PosEncod3ClassesNoSilverNewGaz, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(PosEncod3ClassesNoSilverBig, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(PosEncod3ClassesNoSilverSpanish, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(PosEncod3ClassesOnlyRoberta, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
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
    def __init__(self):
        super(OnlyRoberta3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
        self.input_dim = 1024
        self.num_class = (args['num_types'] * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)

    def forward(self, bert_encoding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        return self.classifier(bert_embeddings)

class JustBert3Classes(torch.nn.Module):
    def __init__(self):
        super(JustBert3Classes, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args['bert_model_name'])
        self.input_dim = args['bert_model_output_dim']
        self.num_class = (args['num_types'] * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)

    def forward(self, bert_encoding):
        bert_embeddings = self.bert_model(bert_encoding['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]
        return self.classifier(bert_embeddings)
