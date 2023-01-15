import util
import utils.dropbox as dropbox_util
import train_util
from transformers import AutoTokenizer
from args import TESTING_MODE, EXPERIMENT_NAME
import time
import numpy as np
import logging  # configured in args.py
import csv
from dataset_configs import span_bert_configurations
import torch

# Setup logging
root_logger = logging.getLogger()
roots_handler = root_logger.handlers[0]
roots_handler.setFormatter(logging.Formatter('%(name)s: %(message)s')) # change formatting
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
logging.getLogger('dropbox').setLevel(logging.WARN)

# -------- CREATE IMPORTANT DIRECTORIES -------
logger.info("Create folders for training")

training_results_folder_path = './training_results'

mistakes_folder_path = f'{training_results_folder_path}/mistakes'
error_visualization_folder_path = f'{training_results_folder_path}/error_visualizations'
predictions_folder_path = f'{training_results_folder_path}/predictions'
models_folder_path = f'{training_results_folder_path}/models'
performance_folder_path = f'{training_results_folder_path}/performance'

# Create training-results directories
util.create_directory_structure(mistakes_folder_path)
util.create_directory_structure(error_visualization_folder_path)
util.create_directory_structure(predictions_folder_path)
util.create_directory_structure(models_folder_path)
util.create_directory_structure(performance_folder_path)

performance_file_path = f"{performance_folder_path}/performance_{EXPERIMENT_NAME}.csv"
train_util.create_performance_file_header(performance_file_path)

dropbox_util.verify_connection()

for dataset_config in span_bert_configurations:
    train_util.print_args(dataset_config)
    # -------- READ DATA ---------
    # TODO: read Samples instead of reading annos, text, tokens separately.
    logger.info("Starting to read data.")
    sample_to_annos_train = train_util.get_train_annos_dict(dataset_config)
    sample_to_annos_valid = train_util.get_valid_annos_dict(dataset_config)
    sample_to_token_data_train = train_util.get_train_tokens(dataset_config)
    sample_to_token_data_valid = train_util.get_valid_tokens(dataset_config)
    sample_to_text_train = train_util.get_train_texts(dataset_config)
    sample_to_text_valid = train_util.get_valid_texts(dataset_config)
    logger.info(f"num train samples {len(sample_to_text_train)}")
    logger.info(f"num valid samples {len(sample_to_text_valid)}")
    logger.info("finished reading data.")

    # ------ MODEL INITIALISATION --------
    logger.info("Starting model initialization.")
    bert_tokenizer = AutoTokenizer.from_pretrained(dataset_config['bert_model_name'])
    model = train_util.prepare_model(dataset_config)
    optimizer = train_util.get_optimizer(model, dataset_config)
    all_types = util.get_all_types(dataset_config['types_file_path'])
    logger.debug(f"all types\n {util.p_string(list(all_types))}")
    logger.info("Finished model initialization.")

    # verify that all label types in annotations are valid types
    for _, sample_annos in sample_to_annos_train.items():
        for anno in sample_annos:
            assert anno.label_type in all_types, f"{anno.label_type}"
    for _, sample_annos in sample_to_annos_valid.items():
        for anno in sample_annos:
            assert anno.label_type in all_types, f"{anno.label_type}"

    for epoch in range(dataset_config['num_epochs']):
        # Don't train for more than 2 epochs while testing
        if TESTING_MODE and epoch > 1:
            break

        epoch_loss = []
        # --------- BEGIN TRAINING ----------------
        logger.info(f"Train epoch {epoch}")
        train_start_time = time.time()
        model.train()
        train_sample_ids = list(sample_to_token_data_train.keys())
        if TESTING_MODE:
            train_sample_ids = train_sample_ids[:10]
        for sample_id in train_sample_ids:
            optimizer.zero_grad()
            sample_token_data = sample_to_token_data_train[sample_id]
            sample_annos = sample_to_annos_train.get(sample_id, [])
            loss, predicted_annos = model(sample_token_data, sample_annos)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().detach().numpy())
        logger.info(f"Done training epoch {epoch}")
        logger.info(
            f"Epoch {epoch} Loss : {np.array(epoch_loss).mean()}, Training Time: {str(time.time() - train_start_time)} "
            f"seconds")
        # torch.save(model.state_dict(), f"{models_folder_path}/Epoch_{epoch}_{EXPERIMENT}")
        # logger.info("done saving model")
        # ------------------ BEGIN VALIDATION -------------------
        logger.info("Starting validation")
        model.eval()
        mistakes_file_path = f"{mistakes_folder_path}/mistakes_{EXPERIMENT_NAME}_epoch_{epoch}.tsv"
        predictions_file_path = f"{predictions_folder_path}/predictions_{EXPERIMENT_NAME}_epoch_{epoch}.tsv"
        with open(predictions_file_path, 'w') as predictions_file, \
                open(mistakes_file_path, 'w') as mistakes_file:
            #  --- GET FILES READY FOR WRITING ---
            predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
            mistakes_file_writer = csv.writer(mistakes_file, delimiter='\t')
            train_util.prepare_file_headers(mistakes_file_writer, predictions_file_writer)
            with torch.no_grad():
                validation_start_time = time.time()
                f1_list = []
                valid_sample_ids = list(sample_to_token_data_valid.keys())
                if TESTING_MODE:
                    valid_sample_ids = valid_sample_ids[:10]
                num_TP_total = 0
                num_FP_total = 0
                num_FN_total = 0
                for sample_id in valid_sample_ids:
                    token_data_valid = sample_to_token_data_valid[sample_id]
                    gold_annos_valid = sample_to_annos_valid.get(sample_id, [])
                    loss, predicted_annos_valid = model(token_data_valid, gold_annos_valid)
                    gold_annos_set_valid = set(
                        [
                            (gold_anno.begin_offset, gold_anno.end_offset, gold_anno.label_type)
                            for gold_anno in gold_annos_valid
                        ]
                    )
                    predicted_annos_set_valid = set(
                        [
                            (predicted_anno.begin_offset, predicted_anno.end_offset, predicted_anno.label_type)
                            for predicted_anno in predicted_annos_valid
                        ]
                    )

                    # calculate true positives, false positives, and false negatives
                    true_positives_sample = gold_annos_set_valid.intersection(predicted_annos_set_valid)
                    false_positives_sample = predicted_annos_set_valid.difference(gold_annos_set_valid)
                    false_negatives_sample = gold_annos_set_valid.difference(predicted_annos_set_valid)
                    num_TP = len(true_positives_sample)
                    num_TP_total += num_TP
                    num_FP = len(false_positives_sample)
                    num_FP_total += num_FP
                    num_FN = len(false_negatives_sample)
                    num_FN_total += num_FN

                    # write sample predictions
                    train_util.store_predictions(sample_id, token_data_valid, predicted_annos_valid,
                                                 predictions_file_writer)
                    # write sample mistakes
                    train_util.store_mistakes(sample_id, false_positives_sample, false_negatives_sample,
                                              mistakes_file_writer, token_data_valid)
        micro_f1, micro_precision, micro_recall = util.f1(num_TP_total, num_FP_total, num_FN_total)
        logger.info(f"Micro f1 {micro_f1}, prec {micro_precision}, recall {micro_recall}")
        visualize_errors_file_path = f"{error_visualization_folder_path}/" \
                                     f"visualize_errors_{EXPERIMENT_NAME}_epoch_{epoch}.bdocjs"
        util.create_mistakes_visualization(mistakes_file_path, visualize_errors_file_path, sample_to_annos_valid,
                                           sample_to_text_valid)
        train_util.store_performance_result(performance_file_path, micro_f1, epoch, EXPERIMENT_NAME,
                                            dataset_config['dataset'])

        # upload files to dropbox
        dropbox_util.upload_file(visualize_errors_file_path)
        # dropbox_util.upload_file(predictions_file_path)
        # dropbox_util.upload_file(mistakes_file_path)
        dropbox_util.upload_file(performance_file_path)

        logger.info(f"Epoch {epoch} DONE!\n\n\n")

    #             pred_label_indices_expanded = torch.argmax(output, dim=1).cpu().detach().numpy()
    #             token_level_accuracy = accuracy_score(list(pred_label_indices_expanded), list(expanded_labels_indices))
    #             token_level_accuracy_list.append(token_level_accuracy)
    #             pred_labels = [idx_to_label_dict[label_id] for label_id in pred_label_indices_expanded]
    #             pred_spans_token_index = train_util.get_spans_from_seq_labels(pred_labels, batch_encoding)
    #             pred_spans_char_offsets = [(offsets_list[span[0]][0], offsets_list[span[1]][1], span[2]) for span in
    #                                        pred_spans_token_index]
    #             gold_annos = sample_to_annos_valid.get(sample_id, [])
    #             gold_spans_char_offsets = [(anno.begin_offset, anno.end_offset, anno.label_type) for anno in gold_annos]
    #             gold_spans_set = set(gold_spans_char_offsets)
    #             pred_spans_set = set(pred_spans_char_offsets)
    #             TP = gold_spans_set.intersection(pred_spans_set)
    #             FP = pred_spans_set.difference(gold_spans_set)
    #             FN = gold_spans_set.difference(pred_spans_set)
    #             num_TP = len(TP)
    #             num_TP_total += num_TP
    #             num_FP = len(FP)
    #             num_FP_total += num_FP
    #             num_FN = len(FN)
    #             num_FN_total += num_FN
    #             num_TN = 0
    #             F1 = util.f1(num_TP, num_FP, num_FN)
    #             f1_list.append(F1[0])
    #             # write predictions
    #             for span in pred_spans_set:
    #                 start_offset = span[0]
    #                 end_offset = span[1]
    #                 extraction = util.get_extraction(tokens=tokens, token_offsets=offsets_list, start=start_offset,
    #                                                  end=end_offset)
    #                 predictions_file_writer.writerow(
    #                     [sample_id, str(start_offset), str(end_offset), span[2], extraction])
    #             # write false negative errors
    #             for span in FP:
    #                 start_offset = span[0]
    #                 end_offset = span[1]
    #                 extraction = util.get_extraction(tokens=tokens, token_offsets=offsets_list, start=start_offset,
    #                                                  end=end_offset)
    #                 mistakes_file_writer.writerow(
    #                     [sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FP'])
    #             # write false positive errors
    #             for span in FN:
    #                 start_offset = span[0]
    #                 end_offset = span[1]
    #                 extraction = util.get_extraction(tokens=tokens, token_offsets=offsets_list, start=start_offset,
    #                                                  end=end_offset)
    #                 mistakes_file_writer.writerow(
    #                     [sample_id, str(start_offset), str(end_offset), span[2], extraction, 'FN'])
    #     print("Token Level Accuracy", np.array(token_level_accuracy_list).mean(),
    #           f"Validation time : {str(time.time() - validation_start_time)} seconds")
    #     print("Macro f1: ", np.array(f1_list).mean())
    #     micro_f1, micro_precision, micro_recall = util.f1(num_TP_total, num_FP_total, num_FN_total)
    #     print(f"Micro f1 {micro_f1}, prec {micro_precision}, recall {micro_recall}")
    # visualize_errors_file_path = args['save_models_dir'] + f"/visualize_errors_{EXPERIMENT}_epoch_{epoch}.bdocjs"
    # util.create_mistakes_visualization(errors_file_path, visualize_errors_file_path, sample_to_annos_valid,
    #                                    sample_to_text_valid)


print("Training Finished!!")