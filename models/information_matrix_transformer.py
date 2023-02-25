import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Model that uses a Transformer Encoder on top of the information matrix.
class TransformerOnInformationMatrix(torch.nn.Module):
    def __init__(self, dataset_config):
        super(TransformerOnInformationMatrix, self).__init__()
        self.bert_model = AutoModel.from_pretrained(dataset_config['bert_model_name'])
        self.bert_base_dim = 768
        # dimensions after concatenation
        self.input_dim = self.bert_base_dim + 8  # 768
        self.num_class = 3  # every token will be classified into 3 categories
        self.num_heads = 8

        # We need to add position information to each token.
        # In other words, we need to indicate the position of each token in the sentence.
        self.pos_encoder = PositionalEncoding(d_model=self.input_dim)
        # I am going to use this classifier to classify each token.
        self.classifier = nn.Linear(self.input_dim, self.num_class)

        # Below is the actual Transformer Encoder which will 'enrich' the token information
        # with the concatenated external knowledge.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

    def forward(
            self,
            bert_encoding_for_one_sentence,
            one_hot_external_knowledge_for_one_sentence
    ):
        bert_embeddings = self.bert_model(bert_encoding_for_one_sentence['input_ids'], return_dict=True)
        bert_embeddings = bert_embeddings['last_hidden_state'][0]

        # concatenate external knowledge one-hot-token-vectors to bert-token-vector
        x = torch.cat((bert_embeddings, one_hot_external_knowledge_for_one_sentence), 1)
        x = self.pos_encoder(x)  # add positional encoding to our concatenated vector
        # pass the concatenated vector through transformer which will return enriched token representations
        # of the same dimensions (bert_embedding_dim + one_hot_dim)
        out = self.encoder(x)

        # do something with the enriched tokens, here I classify each token.
        return self.classifier(out)
