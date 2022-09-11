from nn_utils import *

with open(f'datasets/few-nerd-dataset/supervised/train.txt', 'r') as f:
    labels = set()
    for line in f.readlines():
        line = line.strip()
        if len(line):
            assert len(line.split('\t')) == 2
            token, label = line.split('\t')
            if label != 'O':
                labels.add(label)
    with open('datasets/few-nerd-dataset/types.txt', 'w') as output_file:
        for label in sorted(list(labels)):
            print(label, file=output_file)
