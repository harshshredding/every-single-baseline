import json
import pandas as pd
import csv
import argparse
from pathlib import Path
import subprocess

chatgpt_predictions_file_path = './chatgpt_social_dis_ner_test.json'
with open(chatgpt_predictions_file_path, 'r') as chat_gpt_predictions_file:
    chat_gpt_predictions = json.load(chat_gpt_predictions_file)

chat_gpt_response_dict = {
    sample_id: diseases
    for sample_id, diseases in chat_gpt_predictions
}

    
predictions_tsv_file = '/Users/harshverma/every-single-baseline/useful_scripts/combo_defaults/harshv_research_nlp/experiment_social_dis_ner_all_defaults_combo_social_dis_ner_vanilla_combo_model_seq_large_default_test_epoch_11_predictions.tsv'
output_file = './useful_scripts/vanilla_combo_seq_early.tsv'

predictions_df = pd.read_csv(predictions_tsv_file, sep='\t')

def chatgpt_submission():
    with open(output_file, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['tweets_id', 'begin', 'end', 'type', 'extraction'])
        for _, row in predictions_df.iterrows():
            sample_id = str(row['sample_id'])
            assert sample_id in chat_gpt_response_dict, f"{row['sample_id']}"
            chat_gpt_response = chat_gpt_response_dict[sample_id]
            gpt_predictions_prefix = chat_gpt_response + ' [SEP] '
            begin = int(row['begin'])
            end = int(row['end'])
            begin -= len(gpt_predictions_prefix)
            end -= len(gpt_predictions_prefix)
            writer.writerow([sample_id, str(begin), str(end), row['type'], row['extraction']])


def vanilla_submission():
    with open(output_file, 'w') as output_tsv: 
        writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
        writer.writerow(['tweets_id', 'begin', 'end', 'type', 'extraction'])
        for _, row in predictions_df.iterrows():
            sample_id = str(row['sample_id'])
            begin = int(row['begin'])
            end = int(row['end'])
            writer.writerow([sample_id, str(begin), str(end), row['type'], row['extraction']]) 

def zip_file(file_path):
    file_path = Path(file_path)
    file_stem = file_path.stem
    file_folder = str(file_path.parent)

    cmd_str = f"cd {file_folder} && zip {file_stem}.zip {file_stem}.tsv"
    subprocess.run(cmd_str, shell=True)



argsparser = argparse.ArgumentParser("social dis ner submission script")
argsparser.add_argument("submission_type", help="the type of submission you want to make. Specify 'chatgpt' or 'vanilla':", type=str)
args = argsparser.parse_args()
if args.submission_type == 'chatgpt':
    chatgpt_submission()
elif args.submission_type == 'vanilla':
    vanilla_submission()
else:
    raise RuntimeError(f"Submission type '{args.submission_type}' is not supported")
zip_file(output_file)
