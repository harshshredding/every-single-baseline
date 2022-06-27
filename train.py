from nn_utils import *
from models import SeqLabeler
import numpy as np

from transformers import AutoTokenizer
from read_gate_output import *

sample_to_token_data = get_sample_to_token_data('/home/claclab/harsh/smm4h/smm4h-2022-social-dis-ner/train-2.json')
all_sample_to_token_data = get_all_sample_to_token_data('/home/claclab/harsh/smm4h/smm4h-2022-social-dis-ner')
print(len(sample_to_token_data))
print(len(all_sample_to_token_data))
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
seq_labeler_model = SeqLabeler(1, 128, 1, 2)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq_labeler_model.parameters(), lr=1e-5)
for epoch in range(2):
    epoch_loss = []
    for sample_id in sample_to_token_data:
        optimizer.zero_grad()
        sample_id = '1425026916625666065'
        tokens = get_token_strings(sample_to_token_data[sample_id])
        labels = get_labels(sample_to_token_data[sample_id])
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512)
        expanded_labels = expand_labels(batch_encoding, labels)
        expanded_integer_labels = []
        for label in expanded_labels:
            if label == 'o':
                expanded_integer_labels.append(0)
            else:
                expanded_integer_labels.append(1)
        expanded_integer_labels = torch.tensor(expanded_integer_labels)
        output = seq_labeler_model(batch_encoding)
        loss = loss_function(output, expanded_integer_labels)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.cpu().detach().numpy())
        break
    print(np.array(epoch_loss).mean())