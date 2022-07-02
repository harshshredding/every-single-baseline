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


def get_train_data(directory_path):
    sample_to_tokens = {}
    for file_index in range(5):
        file_index += 1
        file_path = directory_path + f'/train-{file_index}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        for token_data in data:
            assert 'tweet_text' in token_data
            assert len(token_data['tweet_text']) == 1
        for token_data in data:
            sample_id = token_data['tweet_text'][0]['twitter_id']
            sample_tokens = sample_to_tokens.get(sample_id, [])
            sample_tokens.append(token_data)
            sample_to_tokens[sample_id] = sample_tokens
    return sample_to_tokens


def get_valid_data(directory_path):
    sample_to_tokens = {}
    for file_index in range(3):
        file_index += 1
        file_path = directory_path + f'/valid-{file_index}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        for token_data in data:
            assert 'tweet_text' in token_data
            assert len(token_data['tweet_text']) == 1
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
        if 'Span' in token_data:
            disease_tags.append('Disease')
        else:
            disease_tags.append('o')
    return disease_tags


def get_token_offsets(sample_data):
    offsets_list = []
    for token_data in sample_data:
        tweet_start = token_data['tweet_text'][0]['startOffset']
        offsets_list.append((token_data['Token'][0]['startOffset'] - tweet_start,
                             token_data['Token'][0]['endOffset'] - tweet_start))
    return offsets_list


def get_umls_data(sample_data):
    umls_tags = []
    for token_data in sample_data:
        if 'UMLS' in token_data:
            umls_tags.append(token_data['UMLS'])
        else:
            umls_tags.append('o')
    return umls_tags
