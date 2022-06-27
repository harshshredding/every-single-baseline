import os
import csv
from train_annos import get_annos_dict
tweet_to_annos = get_annos_dict()
task_name = 'task_10_disease'
print(len(tweet_to_annos))
train_folder_path = '/home/claclab/harsh/smm4h-2022-social-dis-ner/socialdisner-data/train-valid-txt-files/training'
for file_index in range(5):
    with open(f'gate_input_{task_name}_{file_index+1}.csv', 'w') as csv_output:
        writer = csv.writer(csv_output)
        header = ['twitter_id', 'tweet_text', 'spans']
        writer.writerow(header)
        data_files_list = os.listdir(train_folder_path)[file_index*1000:(file_index+1)*1000]
        for filename in data_files_list:
            data_file_path = os.path.join(train_folder_path, filename)
            with open(data_file_path, 'r') as f:
                data = f.read()
                new_str = str()
                for char in data:
                    if ord(char) < 2047:
                        new_str = new_str + char
                    else:
                        new_str = new_str + ' '
                data=new_str
            twitter_id = filename[:-4]
            tweet_annos = tweet_to_annos.get(twitter_id, [])
            spans = "@".join([f"{anno['begin']}:{anno['end']}" for anno in tweet_annos])
            row_to_write = [twitter_id, data, spans]
            writer.writerow(row_to_write)