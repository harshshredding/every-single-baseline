import json


def get_sample_to_token_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for token_data in data:
        assert 'tweet_text' in token_data
        assert len(token_data['tweet_text']) == 1
    sample_to_tokens = {}
    for token_data in data:
        sample_id = token_data['tweet_text'][0]['twitter_id']
        sample_tokens = sample_to_tokens.get(sample_id, [])
        sample_tokens.append(token_data)
        sample_to_tokens[sample_id] = sample_tokens
    return sample_to_tokens


def get_token_strings(sample_data):
    only_token_strings = []
    for token_data in sample_data:
        only_token_strings.append(token_data['Token'][0]['string'])
    return only_token_strings


def get_labels(sample_data):
    disease_tags = []
    for token_data in sample_data:
        if 'Disease' in token_data:
            disease_tags.append('Disease')
        else:
            disease_tags.append('o')
    return disease_tags
