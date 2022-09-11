import csv
import json


def create_annos_file(split):
    with open(f'../../datasets/few-nerd-dataset/supervised/{split}.txt', 'r') as f:
        labels = []
        curr_sample_labels = []
        prev_label = None
        label_start = None
        offset = 0
        curr_extraction = None
        for line in f.readlines():
            line = line.strip()
            if len(line):
                assert len(line.split('\t')) == 2
                token, label = line.split('\t')
                if label != 'O':
                    if label == prev_label:
                        assert label_start is not None
                        assert len(curr_extraction) > 0
                        curr_extraction.append(token)
                    else:
                        if label_start is not None:
                            assert prev_label is not None and prev_label != 'O' and len(curr_extraction)
                            curr_sample_labels.append((label_start, offset, prev_label, ' '.join(curr_extraction)))
                        label_start = offset
                        curr_extraction = [token]
                else:
                    if label_start is not None:
                        assert prev_label is not None and prev_label != 'O' and len(curr_extraction)
                        curr_sample_labels.append((label_start, offset, prev_label, ' '.join(curr_extraction)))
                    label_start = None
                    curr_extraction = None
                prev_label = label
                offset += len(token)
            else:
                if label_start is not None:
                    assert prev_label is not None and prev_label != 'O' and len(curr_extraction)
                    curr_sample_labels.append((label_start, offset, prev_label, ' '.join(curr_extraction)))
                labels.append(curr_sample_labels)
                curr_sample_labels = []
                prev_label = None
                label_start = None
                offset = 0
                curr_extraction = None
        with open(f'../../datasets/few-nerd-dataset/gold-annos/few_nerd_{split}_annos.tsv', 'w') as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            header = ['sample_id', 'begin', 'end', 'type', 'extraction']
            writer.writerow(header)
            for i, sample_labels in enumerate(labels):
                for label in sample_labels:
                    row = [str(i), str(label[0]), str(label[1]), str(label[2]), str(label[3])]
                    writer.writerow(row)


def create_model_input_files(split, small=False):
    with open(f'../../datasets/few-nerd-dataset/supervised/{split}.txt', 'r') as f:
        sample_id = 0
        all_tokens = []
        offset = 0
        for line in f.readlines():
            line = line.strip()
            if len(line):
                assert len(line.split('\t')) == 2
                token, label = line.split('\t')
                token_data = {'Token': [{"string": token, "startOffset": offset,
                                         "endOffset": offset + len(token), "length": len(token)}],
                              'Sample': [{"id": sample_id, "startOffset": 0}]}
                if label != 'O':
                    token_data['Span'] = [{"type": label}]
                all_tokens.append(token_data)
                offset += len(token)
            else:
                sample_id += 1
                if small and sample_id > 100:
                    break
                offset = 0
        output_file_name = f"{split}.json"
        output_folder_name = f'../../datasets/few-nerd-dataset/input_files_{split}'
        if small:
            output_file_name = f"{split}_small.json"
            output_folder_name = f'../../datasets/few-nerd-dataset/input_files_{split}_small'
        with open(f'{output_folder_name}/{output_file_name}', 'w') as output_file:
            json.dump(all_tokens, output_file)


# create_annos_file('dev')
# create_annos_file('test')
# create_annos_file('train')
#
create_model_input_files('dev')
create_model_input_files('test')
create_model_input_files('train')

create_model_input_files('dev', small=True)
create_model_input_files('test', small=True)
create_model_input_files('train', small=True)