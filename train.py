import util
import train_util
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from args import *
import time
import numpy as np
import csv
import logging
logging.basicConfig(filename='train.log', encoding='utf-8', level=logging.INFO)

train_util.print_args()

# -------- READ DATA ---------
# TODO: read Samples instead of reading annos, text, tokens separately.
logging.info("starting to read data.")
sample_to_annos_train = train_util.get_train_annos_dict()
sample_to_annos_valid = train_util.get_valid_annos_dict()
sample_to_token_data_train = train_util.get_train_tokens()
sample_to_token_data_valid = train_util.get_valid_tokens()
sample_to_text_train = train_util.get_train_texts()
sample_to_text_valid = train_util.get_valid_texts()
logging.info(f"num train samples {len(sample_to_text_train)}")
logging.info(f"num valid samples {len(sample_to_text_valid)}")
logging.info("finished reading data.")

# ------ MODEL INITIALISATION --------
logging.info("Starting model initialization.")
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
model = train_util.prepare_model()
loss_function = train_util.get_loss_function()
optimizer = train_util.get_optimizer(model)
label_to_idx_dict, idx_to_label_dict = train_util.get_label_idx_dicts()
all_types = set([label.label_type for label in label_to_idx_dict])
print("all types\n", util.p_string(list(all_types)))
logging.info("Finished model initialization.")

# verify that all label types in annotations are valid types
for _, annos in sample_to_annos_train.items():
    for anno in annos:
        assert anno.label_type in all_types, f"{anno.label_type}"
for _, annos in sample_to_annos_valid.items():
    for anno in annos:
        assert anno.label_type in all_types, f"{anno.label_type}"




for epoch in range(args['num_epochs']):
    epoch_loss = [] 
    # --------- BEGIN TRAINING ----------------
    print(f"Train epoch {epoch}")
    train_start_time = time.time()
    model.train()
    sample_ids = list(sample_to_token_data_train.keys())
    if TESTING_MODE:
        sample_ids = sample_ids[:10]
    for sample_id in sample_ids:
        optimizer.zero_grad()
        sample_data = sample_to_token_data_train[sample_id]
        annos = sample_to_annos_train.get(sample_id, [])
        tokens = util.get_token_strings(sample_data)
        batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512).to(device)
        if not len(batch_encoding.word_ids()): # If we don't have any tokens, no point training
            continue
        if len(batch_encoding.word_ids()) > 512:
            logging.warn(f"sample_id: {sample_id} is too long, num subtokens {len(batch_encoding.word_ids())}")
        expanded_labels = train_util.extract_expanded_labels(sample_data, batch_encoding, annos)
        expanded_labels_indices = [label_to_idx_dict[label] for label in expanded_labels]
        model_input = train_util.prepare_model_input(batch_encoding, sample_data)
        output = model(*model_input)
        expanded_labels_tensor = torch.tensor(expanded_labels_indices).to(device)
        loss = loss_function(output, expanded_labels_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.cpu().detach().numpy())
    print(
        f"Epoch {epoch} Loss : {np.array(epoch_loss).mean()}, Training time: {str(time.time() - train_start_time)} "
        f"seconds")
    torch.save(model.state_dict(), args['save_models_dir'] + f"/Epoch_{epoch}_{EXPERIMENT}")
    # ------------------ BEGIN VALIDATION -------------------
    model.eval()
    errors_file_path = args['save_models_dir'] + f"/errors_{EXPERIMENT}_epoch_{epoch}.tsv"
    predictions_file_path = args['save_models_dir'] + f"/predictions_{EXPERIMENT}_epoch_{epoch}.tsv"
    with open(predictions_file_path, 'w') as predictions_file, open(errors_file_path, 'w') as errors_file:
        mistakes_file_writer = csv.writer(errors_file, delimiter='\t')
        mistakes_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction', 'mistake_type']
        mistakes_file_writer.writerow(mistakes_file_header)
        predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
        predictions_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction', 'mistake_type']
        predictions_file_writer.writerow(predictions_file_header)
        with torch.no_grad():
            validation_start_time = time.time()
            token_level_accuracy_list = []
            f1_list = []
            sample_ids = list(sample_to_token_data_valid.keys())
            if TESTING_MODE:
                sample_ids = sample_ids[:10]
            num_TP_total = 0
            num_FP_total = 0
            num_FN_total = 0
            for sample_id in sample_ids:
                sample_data = sample_to_token_data_valid[sample_id]
                annos = sample_to_annos_valid.get(sample_id, [])
                tokens = util.get_token_strings(sample_data)
                offsets_list = util.get_token_offsets(sample_data)
                batch_encoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                                add_special_tokens=False, truncation=True, max_length=512).to(device)
                if not len(batch_encoding.word_ids()): # If we don't have any tokens, no point training
                    continue
                expanded_labels = train_util.extract_expanded_labels(sample_data, batch_encoding, annos)
                expanded_labels_indices = [label_to_idx_dict[label] for label in expanded_labels]
                model_input = train_util.prepare_model_input(batch_encoding, sample_data)
                output = model(*model_input)
                pred_label_indices_expanded = torch.argmax(output, dim=1).cpu().detach().numpy()
                token_level_accuracy = accuracy_score(list(pred_label_indices_expanded), list(expanded_labels_indices))
                token_level_accuracy_list.append(token_level_accuracy)
                pred_labels = [idx_to_label_dict[label_id] for label_id in pred_label_indices_expanded]
                pred_spans_token_index = train_util.get_spans_from_seq_labels(pred_labels, batch_encoding)
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
                num_TP_total += num_TP
                num_FP = len(FP)
                num_FP_total += num_FP
                num_FN = len(FN)
                num_FN_total += num_FN
                num_TN = 0
                F1 = util.f1(num_TP, num_FP, num_FN)
                f1_list.append(F1[0])
                # write predictions
                for span in pred_spans_set:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = util.get_extraction(tokens=tokens, token_offsets=offsets_list, start=start_offset, end=end_offset)
                    predictions_file_writer.writerow([sample_id, str(start_offset), str(end_offset), span[2], extraction])
                # write false negative errors
                for span in FP:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = util.get_extraction(tokens=tokens, token_offsets=offsets_list, start=start_offset, end=end_offset)
                    mistakes_file_writer.writerow([sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FP'])
                # write false positive errors
                for span in FN:
                    start_offset = span[0]
                    end_offset = span[1]
                    extraction = util.get_extraction(tokens=tokens, token_offsets=offsets_list, start=start_offset, end=end_offset)
                    mistakes_file_writer.writerow([sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FN'])
        print("Token Level Accuracy", np.array(token_level_accuracy_list).mean(),
              f"Validation time : {str(time.time() - validation_start_time)} seconds")
        print("Macro f1: ", np.array(f1_list).mean())
        micro_f1, micro_precision, micro_recall = util.f1(num_TP_total, num_FP_total, num_FN_total)
        print(f"Micro f1 {micro_f1}, prec {micro_precision}, recall {micro_recall}") 
    visualize_errors_file_path = args['save_models_dir'] + f"/visualize_errors_{EXPERIMENT}_epoch_{epoch}.bdocjs"
    util.create_mistakes_visualization(errors_file_path, visualize_errors_file_path, sample_to_annos_valid, sample_to_text_valid)