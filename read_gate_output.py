import json
from args import default_key


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


def get_train_data_small(directory_path):
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
        break
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


def get_valid_data_small(directory_path):
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
        break
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


def get_labels_rich(sample_data, annos):
    labels = get_labels(sample_data)
    offsets = get_token_offsets(sample_data)
    new_labels = []
    for (label, offset) in zip(labels, offsets):
        if label == 'Disease':
            anno_same_start = [anno for anno in annos if anno['begin'] == offset[0]]
            if len(anno_same_start) > 0:
                new_labels.append('DiseaseStart')
            else:
                # avoid DiseaseMid without a DiseaseStart
                if (len(new_labels) > 0) and (new_labels[-1] != 'o'):
                    new_labels.append('DiseaseMid')
                else:
                    new_labels.append('o')
        else:
            new_labels.append('o')
    return new_labels


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


def get_dis_gaz_labels(sample_data):
    output = []
    for token_data in sample_data:
        if 'DisGaz' in token_data:
            output.append('DisGaz')
        else:
            output.append('o')
    return output


def get_dis_gaz_one_hot(sample_data):
    dis_labels = get_dis_gaz_labels(sample_data)
    return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]


def get_umls_diz_gaz_labels(sample_data):
    output = []
    for token_data in sample_data:
        if 'UMLS_Disease' in token_data:
            output.append('UmlsDisGaz')
        else:
            output.append('o')
    return output


def get_umls_dis_gaz_one_hot(sample_data):
    dis_labels = get_umls_diz_gaz_labels(sample_data)
    return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]


def get_silver_dis_one_hot(sample_data):
    dis_labels = get_silver_dis_labels(sample_data)
    return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]


def get_silver_dis_labels(sample_data):
    output = []
    for token_data in sample_data:
        if 'SilverDisGaz' in token_data:
            output.append('SilverDisGaz')
        else:
            output.append('o')
    return output


def get_pos_data(sample_data):
    pos_tags = []
    for token_data in sample_data:
        pos_tags.append(token_data['Token'][0]['category'])
    return pos_tags


def get_umls_indices(sample_data, umls_key_to_index):
    umls_data = get_umls_data(sample_data)
    umls_keys = [default_key if umls == 'o' else umls[0]['CUI'] for umls in umls_data]
    default_index = umls_key_to_index[default_key]
    umls_indices = [umls_key_to_index.get(key, default_index) for key in umls_keys]
    return umls_indices


def get_pos_indices(sample_data, pos_key_to_index):
    pos_tags = get_pos_data(sample_data)
    default_index = pos_key_to_index[default_key]
    pos_indices = [pos_key_to_index.get(tag, default_index) for tag in pos_tags]
    return pos_indices
