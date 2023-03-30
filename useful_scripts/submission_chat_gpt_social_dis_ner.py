import json
import pandas as pd
import csv

chatgpt_predictions_file_path = './chatgpt_social_dis_ner_test.json'
with open(chatgpt_predictions_file_path, 'r') as chat_gpt_predictions_file:
    chat_gpt_predictions = json.load(chat_gpt_predictions_file)

chat_gpt_response_dict = {
    sample_id: diseases
    for sample_id, diseases in chat_gpt_predictions
}

    
chat_gpt_tsv_file = './useful_scripts/experiment_chatgpt_span_social_dis_ner_transformer_big_social_dis_ner_chatgpt_span_large_default_test_epoch_19_predictions.tsv'
df = pd.read_csv(chat_gpt_tsv_file, sep='\t')

with open('./useful_scripts/final_span_transformer_large.tsv', 'w') as output_tsv: 
    writer = csv.writer(output_tsv, delimiter='\t', lineterminator='\n')
    writer.writerow(['tweets_id', 'begin', 'end', 'type', 'extraction'])
    for _, row in df.iterrows():
        sample_id = str(row['sample_id'])
        assert sample_id in chat_gpt_response_dict, f"{row['sample_id']}"
        chat_gpt_response = chat_gpt_response_dict[sample_id]
        gpt_predictions_prefix = chat_gpt_response + ' [SEP] '
        begin = int(row['begin'])
        end = int(row['end'])
        begin -= len(gpt_predictions_prefix)
        end -= len(gpt_predictions_prefix)
        writer.writerow([sample_id, str(begin), str(end), row['type'], row['extraction']]) 
