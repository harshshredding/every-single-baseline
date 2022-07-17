import torch.optim

from util import *
from transformers import AutoTokenizer
from read_gate_output import *
from sklearn.metrics import accuracy_score
from train_annos import get_annos_dict
from args import args
from args import device
import time
import torch_optimizer

print(args)
tweet_to_annos = get_annos_dict(args['gold_file_path'])
if args['testing_mode']:
    sample_to_token_data_train = get_train_data_small(args['training_data_folder_path'])
    sample_to_token_data_valid = get_valid_data_small(args['validation_data_folder_path'])
else:
    sample_to_token_data_train = get_train_data(args['training_data_folder_path'])
    sample_to_token_data_valid = get_valid_data(args['validation_data_folder_path'])
raw_validation_data = get_raw_validation_data()
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
model = prepare_model()
print("Model Instance", type(model))
loss_function = nn.CrossEntropyLoss()
if args['optimizer'] == 'Ranger':
    optimizer = torch_optimizer.Ranger(model.parameters(), args['learning_rate'])
elif args['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), args['learning_rate'])
elif args['optimizer'] == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), args['learning_rate'])
else:
    raise Exception(f"optimizer not found: {args['optimizer']}")

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
        annos = tweet_to_annos.get(sample_id, [])
        tokens = get_token_strings(sample_data)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512).to(device)
        expanded_labels = extract_labels(sample_data, batch_encoding, annos)
        model_input = prepare_model_input(batch_encoding, sample_data)
        output = model(*model_input)
        expanded_labels_tensor = torch.tensor(expanded_labels).to(device)
        loss = loss_function(output, expanded_labels_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.cpu().detach().numpy())
    print(
        f"Epoch {epoch} Loss : {np.array(epoch_loss).mean()}, Training time: {str(time.time() - train_start_time)} "
        f"seconds")
    torch.save(model.state_dict(), args['save_models_dir'] + f"/Epoch_{epoch}_{args['experiment_name']}")
    # Validation starts
    model.eval()
    with open(args['save_models_dir'] + f"/validation_predictions_{args['experiment_name']}_epoch_{epoch}.tsv", 'w') \
            as predictions_file:
        predictions_file.write('\t'.join(['tweets_id', 'begin', 'end', 'type', 'extraction']))
        with torch.no_grad():
            validation_start_time = time.time()
            token_level_accuracy_list = []
            f1_list = []
            sample_ids = list(sample_to_token_data_valid.keys())
            if args['testing_mode']:
                sample_ids = sample_ids[:10]
            for sample_id in sample_ids:
                raw_text = raw_validation_data[sample_id]
                sample_data = sample_to_token_data_valid[sample_id]
                annos = tweet_to_annos.get(sample_id, [])
                tokens = get_token_strings(sample_data)
                offsets_list = get_token_offsets(sample_data)
                batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                                add_special_tokens=False, truncation=True, max_length=512).to(device)
                expanded_labels = extract_labels(sample_data, batch_encoding, annos)
                model_input = prepare_model_input(batch_encoding, sample_data)
                output = model(*model_input)
                pred_labels_expanded = torch.argmax(output, dim=1).cpu().detach().numpy()
                token_level_accuracy = accuracy_score(list(pred_labels_expanded), list(expanded_labels))
                token_level_accuracy_list.append(token_level_accuracy)
                pred_spans_token_index = get_spans_from_seq_labels(pred_labels_expanded, batch_encoding)
                pred_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1]) for span in
                                           pred_spans_token_index]
                gold_annos = tweet_to_annos.get(sample_id, [])
                gold_spans_char_offsets = [(anno['begin'], anno['end']) for anno in gold_annos]
                gold_spans_set = set(gold_spans_char_offsets)
                pred_spans_set = set(pred_spans_char_offsets)
                TP = gold_spans_set.intersection(pred_spans_set)
                FP = pred_spans_set.difference(gold_spans_set)
                FN = gold_spans_set.difference(pred_spans_set)
                num_TP = len(TP)
                num_FP = len(FP)
                num_FN = len(FN)
                num_TN = 0
                F1 = f1(num_TP, num_FP, num_FN)
                f1_list.append(F1)
                for span in pred_spans_set:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = raw_text[start_offset: end_offset]
                    predictions_file.write(
                        '\n' + '\t'.join([sample_id, str(start_offset), str(end_offset), 'ENFERMEDAD', extraction]))
        print("Token Level Accuracy", np.array(token_level_accuracy_list).mean(),
              f"Validation time : {str(time.time() - validation_start_time)} seconds")
        print("F1", np.array(f1_list).mean())
