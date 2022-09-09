from util import *
from transformers import AutoTokenizer
from read_gate_output import *
from sklearn.metrics import accuracy_score
from train_annos import get_train_annos_dict, get_valid_annos_dict
from args import *
import time

print_args()
sample_to_annos_train = get_train_annos_dict()
sample_to_annos_valid = get_valid_annos_dict()
sample_to_token_data_train = get_train_data()
sample_to_token_data_valid = get_valid_data()
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
model = prepare_model()
loss_function = nn.CrossEntropyLoss()
optimizer = get_optimizer(model)
label_to_idx_dict, idx_to_label_dict = get_label_idx_dicts()
assert len(label_to_idx_dict) == args['num_types'] * 2 + 1

for epoch in range(args['num_epochs']):
    epoch_loss = []
    # Training starts
    print(f"Train epoch {epoch}")
    train_start_time = time.time()
    model.train()
    sample_ids = list(sample_to_token_data_train.keys())
    if testing_mode:
        sample_ids = sample_ids[:10]
    for sample_id in sample_ids:
        optimizer.zero_grad()
        sample_data = sample_to_token_data_train[sample_id]
        annos = sample_to_annos_train.get(sample_id, [])
        tokens = get_token_strings(sample_data)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512).to(device)
        expanded_labels = extract_expanded_labels(sample_data, batch_encoding, annos)
        expanded_labels_indices = [label_to_idx_dict[label] for label in expanded_labels]
        model_input = prepare_model_input(batch_encoding, sample_data)
        output = model(*model_input)
        expanded_labels_tensor = torch.tensor(expanded_labels_indices).to(device)
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
    with open(args['save_models_dir'] + f"/predictions_{args['experiment_name']}_epoch_{epoch}.tsv", 'w') \
            as predictions_file, \
            open(args['save_models_dir'] + f"/errors_{args['experiment_name']}_epoch_{epoch}.tsv", 'w') as errors_file:
        predictions_file.write('\t'.join(['sample_id', 'begin', 'end', 'type', 'extraction']))
        errors_file.write('\t'.join(['sample_id', 'begin', 'end', 'type', 'extraction', 'mistake_type']))
        with torch.no_grad():
            validation_start_time = time.time()
            token_level_accuracy_list = []
            f1_list = []
            sample_ids = list(sample_to_token_data_valid.keys())
            if testing_mode:
                sample_ids = sample_ids[:10]
            for sample_id in sample_ids:
                sample_data = sample_to_token_data_valid[sample_id]
                annos = sample_to_annos_valid.get(sample_id, [])
                tokens = get_token_strings(sample_data)
                offsets_list = get_token_offsets(sample_data)
                batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                                add_special_tokens=False, truncation=True, max_length=512).to(device)
                expanded_labels = extract_expanded_labels(sample_data, batch_encoding, annos)
                expanded_labels_indices = [label_to_idx_dict[label] for label in expanded_labels]
                model_input = prepare_model_input(batch_encoding, sample_data)
                output = model(*model_input)
                pred_label_indices_expanded = torch.argmax(output, dim=1).cpu().detach().numpy()
                token_level_accuracy = accuracy_score(list(pred_label_indices_expanded), list(expanded_labels_indices))
                token_level_accuracy_list.append(token_level_accuracy)
                pred_labels = [idx_to_label_dict[label_id] for label_id in pred_label_indices_expanded]
                pred_spans_token_index = get_spans_from_seq_labels(pred_labels, batch_encoding)
                pred_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1], span[2]) for span in
                                           pred_spans_token_index]
                gold_annos = sample_to_annos_valid.get(sample_id, [])
                gold_spans_char_offsets = [(anno.begin_offset, anno.end_offset, anno.label_type) for anno in gold_annos]
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
                # write predictions
                for span in pred_spans_set:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = get_extraction(tokens=tokens, offsets=offsets_list, start=start_offset, end=end_offset)
                    predictions_file.write(
                        '\n' + '\t'.join([sample_id, str(start_offset), str(end_offset), span[2], extraction]))
                # write false negative errors
                for span in FP:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = get_extraction(tokens=tokens, offsets=offsets_list, start=start_offset, end=end_offset)
                    errors_file.write(
                        '\n' + '\t'.join([sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FP']))
                # write false positive errors
                for span in FN:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = get_extraction(tokens=tokens, offsets=offsets_list, start=start_offset, end=end_offset)
                    errors_file.write(
                        '\n' + '\t'.join([sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FN']))
        print("Token Level Accuracy", np.array(token_level_accuracy_list).mean(),
              f"Validation time : {str(time.time() - validation_start_time)} seconds")
        print("F1", np.array(f1_list).mean())
