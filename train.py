import torch
from nn_utils import *
from models import *
import numpy as np
import sys
from util import *
from transformers import AutoTokenizer
from read_gate_output import *
from sklearn.metrics import accuracy_score
from train_annos import get_annos_dict
from args import args
from args import device
from args import default_key
import time

print(args)
tweet_to_annos = get_annos_dict(args['annotations_file_path'])
if args['resources']:
    if args['testing_mode']:
        umls_embedding_dict = read_umls_file_small(args['umls_embeddings_path'])
        umls_embedding_dict[default_key] = [0 for _ in range(50)]
        umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
        umls_key_to_index = get_key_to_index(umls_embedding_dict)
    else:
        umls_embedding_dict = read_umls_file(args['umls_embeddings_path'])
        umls_embedding_dict[default_key] = [0 for _ in range(50)]
        umls_embedding_dict = {k: np.array(v) for k, v in umls_embedding_dict.items()}
        umls_key_to_index = get_key_to_index(umls_embedding_dict)
    pos_dict = read_pos_embeddings_file()
    pos_dict[default_key] = [0 for _ in range(20)]
    pos_dict = {k: np.array(v) for k, v in pos_dict.items()}
    pos_to_index = get_key_to_index(pos_dict)
if args['testing_mode']:
    sample_to_token_data_train = get_train_data_small(args['training_data_folder_path'])
    sample_to_token_data_valid = get_valid_data_small(args['validation_data_folder_path'])
else:
    sample_to_token_data_train = get_train_data(args['training_data_folder_path'])
    sample_to_token_data_valid = get_valid_data(args['validation_data_folder_path'])
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
if args['resources']:
    model = SeqLabelerAllResources(umls_pretrained=umls_embedding_dict, umls_to_idx=umls_key_to_index,
                                   pos_pretrained=pos_dict, pos_to_idx=pos_to_index).to(device)
else:
    model = SeqLabeler().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(args['num_epochs']):
    epoch_loss = []
    # Training starts
    print(f"Train epoch {epoch}")
    train_start_time = time.time()
    model.train()
    sample_ids = list(sample_to_token_data_train.keys())
    if args['testing_mode']:
        sample_ids = sample_ids[:10]
    for sample_id in sample_ids:
        optimizer.zero_grad()
        sample_data = sample_to_token_data_train[sample_id]
        tokens = get_token_strings(sample_data)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512).to(device)
        expanded_labels = extract_labels(sample_data, batch_encoding)
        if args['resources']:
            umls_indices = torch.tensor(expand_labels(batch_encoding, get_umls_indices(sample_data, umls_key_to_index)),
                                        device=device)
            pos_indices = torch.tensor(expand_labels(batch_encoding, get_pos_indices(sample_data, pos_to_index)),
                                       device=device)
        if args['resources']:
            model_input = (batch_encoding, umls_indices, pos_indices)
        else:
            model_input = batch_encoding
        output = model(*model_input)
        loss = loss_function(output, expanded_labels)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.cpu().detach().numpy())
    print(
        f"Epoch {epoch} Loss : {np.array(epoch_loss).mean()}, Training time: {str(time.time() - train_start_time)} seconds")
    torch.save(model.state_dict(), args['save_models_dir'] + f"/Epoch_{epoch}_{args['experiment_name']}")
    # Validation starts
    model.eval()
    with torch.no_grad():
        validation_start_time = time.time()
        token_level_accuracy_list = []
        f1_list = []
        sample_ids = list(sample_to_token_data_valid.keys())
        if args['testing_mode']:
            sample_ids = sample_ids[:10]
        for sample_id in sample_ids:
            sample_data = sample_to_token_data_valid[sample_id]
            tokens = get_token_strings(sample_data)
            offsets_list = get_token_offsets(sample_data)
            batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                            add_special_tokens=False, truncation=True, max_length=512).to(device)
            expanded_labels = extract_labels(sample_data, batch_encoding)
            if args['resources']:
                umls_indices = torch.tensor(
                    expand_labels(batch_encoding, get_umls_indices(sample_data, umls_key_to_index)), device=device)
                pos_indices = torch.tensor(expand_labels(batch_encoding, get_pos_indices(sample_data, pos_to_index)),
                                           device=device)
            if args['resources']:
                model_input = (batch_encoding, umls_indices, pos_indices)
            else:
                model_input = batch_encoding
            output = model(*model_input)
            pred_labels_expanded = torch.argmax(output, dim=1).cpu().detach().numpy()
            token_level_accuracy = accuracy_score(list(pred_labels_expanded), list(expanded_labels))
            token_level_accuracy_list.append(token_level_accuracy)
            pred_spans_token_index = get_spans_from_seq_labels(pred_labels_expanded, batch_encoding)
            pred_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                                       pred_spans_token_index]
            label_spans_token_index = get_spans_from_seq_labels(expanded_labels, batch_encoding)
            label_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                                        label_spans_token_index]
            gold_annos = tweet_to_annos.get(sample_id, [])
            gold_spans_char_offsets = [(anno['begin'], anno['end']) for anno in gold_annos]
            label_spans_set = set(label_spans_char_offsets)
            gold_spans_set = set(gold_spans_char_offsets)
            pred_spans_set = set(pred_spans_char_offsets)
            TP = len(gold_spans_set.intersection(pred_spans_set))
            FP = len(pred_spans_set.difference(gold_spans_set))
            FN = len(gold_spans_set.difference(pred_spans_set))
            TN = 0
            F1 = f1(TP, FP, FN)
            f1_list.append(F1)
        print("Token Level Accuracy", np.array(token_level_accuracy_list).mean(),
              f"Validation time : {str(time.time() - validation_start_time)} seconds")
        print("F1", np.array(f1_list).mean())
