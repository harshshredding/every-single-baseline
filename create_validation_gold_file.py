from util import *
import pandas as pd
validation_ids = get_validation_ids()
validation_ids = set(validation_ids)
df = pd.read_csv(args['gold_file_path'], sep='\t')
with open('validation_entities.tsv', 'w') as validation_gold:
    print("tweets_id\tbegin\tend\ttype\textraction", file=validation_gold)
    for _, row in df.iterrows():
        tweet_id = row['tweets_id']
        begin = row['begin']
        end = row['end']
        type_string = row['type']
        extraction = row['extraction']
        if str(tweet_id) in validation_ids:
            print(f"{tweet_id}\t{begin}\t{end}\t{type_string}\t{extraction}", file=validation_gold)