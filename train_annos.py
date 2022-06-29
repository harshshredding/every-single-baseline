import pandas as pd


def get_annos_dict(annos_file_path):
    df = pd.read_csv(annos_file_path, sep='\t')
    tweet_to_annos = {}
    for i, row in df.iterrows():
        annos_list = tweet_to_annos.get(str(row['tweets_id']), [])
        annos_list.append({"begin": row['begin'], "end": row['end'], "extraction": row['extraction']})
        tweet_to_annos[str(row['tweets_id'])] = annos_list
    return tweet_to_annos
