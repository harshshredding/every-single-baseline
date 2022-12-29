import pandas as pd
from args import *

df = pd.read_csv(args['silver_file_path'], sep='\t')
term_set = set()
with open('gazetteers/silver_gazetteer.lst', 'w') as train_gaz_file:
    for _, row in df.iterrows():
        term = row['extraction']
        term = term.strip()
        term = term.replace("#", "")
        term = term.replace("@", "")
        if term not in term_set:
            print(term, file=train_gaz_file)
            term_set.add(term)