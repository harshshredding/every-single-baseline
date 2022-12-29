import os
import csv
from train_annos import get_annos_dict
from args import args
annos_file_path = args['gold_file_path']
input_folder_path = './test_data/test-data/test-data-txt-files'
output_folder_path = './gate_input_test'
tweet_to_annos = get_annos_dict(annos_file_path)
for file_index in range(25):
    with open(output_folder_path + f'/gate_input_{file_index+1}.csv', 'w') as csv_output:
        writer = csv.writer(csv_output)
        header = ['twitter_id', 'tweet_text', 'spans']
        writer.writerow(header)
        data_files_list = os.listdir(input_folder_path)[file_index * 1000:(file_index + 1) * 1000]
        for filename in data_files_list:
            data_file_path = os.path.join(input_folder_path, filename)
            with open(data_file_path, 'r') as f:
                data = f.read()
                new_str = str()
                for char in data:
                    if ord(char) < 2047:
                        new_str = new_str + char
                    else:
                        new_str = new_str + ' '
                data = new_str
            twitter_id = filename[:-4]
            tweet_annos = tweet_to_annos.get(twitter_id, [])
            spans = "@".join([f"{anno['begin']}:{anno['end']}" for anno in tweet_annos])
            row_to_write = [twitter_id, data, spans]
            writer.writerow(row_to_write)