import csv
from pathlib import Path

import torch
from gatenlp import Document
from structs import *
from typing import Dict, List, Tuple
import json
import os
import pandas as pd
from collections import deque
from spacy.tokens.span import Span
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer
from utils.config import DatasetConfig
import logging
from pudb import set_trace
import shutil


def get_bert_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def get_user_input(input_message: str, possible_values: List[str]):
    options_string = ''
    for option in possible_values:
        options_string += f"- {option}\n"
    user_input = input(f"{input_message}\n Choose from: \n{options_string}")
    if len(possible_values):
        while user_input not in possible_values:
            user_input = input(f"incorrect input '{user_input}', please choose from the given possible values: \n")
    return user_input


def delete_preprocessed_data_folder():
    """
    Use with CAUTION.
    Deletes the preprocessed data folder in the root directory.
    """
    shutil.rmtree('./preprocessed_data')


def ensure_no_sample_gets_truncated_by_bert(samples: List[Sample], dataset_config: DatasetConfig):
    bert_tokenizer = get_bert_tokenizer()
    for sample in samples:
        tokens = get_tokens_from_sample(sample)
        batch_encoding = bert_tokenizer(
            tokens, is_split_into_words=True, truncation=True, return_tensors='pt', add_special_tokens=False
        )
        if batch_encoding['input_ids'].shape[1] == bert_tokenizer.model_max_length:
            print(f"WARN: In dataset {dataset_config.dataset_name}, the sample {sample.id} is being truncated")


def write_samples(samples: List[Sample], output_json_file_path: str):
    with open(output_json_file_path, 'w') as output_file:
        json.dump(samples, output_file, default=vars)


def create_json_file(output_file_path: str, data):
    assert output_file_path.endswith('.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file)


def read_samples(input_json_file_path: str) -> List[Sample]:
    ret = []
    with open(input_json_file_path, 'r') as f:
        sample_list_raw = json.load(f)
        assert type(sample_list_raw) == list
        for sample_raw in sample_list_raw:
            sample = Sample(
                text=sample_raw['text'],
                id=sample_raw['id'],
                annos=AnnotationCollection(
                    gold=get_annotations_from_raw_list(
                        sample_raw["annos"]["gold"]),
                    external=get_annotations_from_raw_list(
                        sample_raw["annos"]["external"])
                )
            )
            ret.append(sample)
    return ret


def get_annos_of_type_from_collection(
        label_type: str,
        collection: AnnotationCollection
) -> List[Anno]:
    return [anno for anno in collection.external if anno.label_type == label_type]


def get_annotations_from_raw_list(annotation_raw_list) -> List[Anno]:
    return [
        Anno(
            begin_offset=annotation_raw['begin_offset'],
            end_offset=annotation_raw['end_offset'],
            label_type=annotation_raw['label_type'],
            extraction=annotation_raw['extraction'],
            features={}
        )
        for annotation_raw in annotation_raw_list
    ]


def visualize_constituency_tree_bfs(spacy_sentence_span):
    """
    Pretty print the constituency tree with helpful indentations.

    Args:
        spacy_sentence_span: (a spacy span)
            The span of one sentence generated after running 
            the benepar constituency parse pipeline.
    """
    queue = deque()
    queue.append((spacy_sentence_span, 0))
    while len(queue):
        curr_span, depth = queue.popleft()
        print(depth * "\t", curr_span._.labels, curr_span)
        for const in curr_span._.children:
            queue.append((const, depth + 1))


def visualize_constituency_tree_dfs(spacy_sentence_span: Span):
    """
    Pretty print the constituency tree with helpful indentations.

    Args:
        spacy_sentence_span: (a spacy span)
            The span of one sentence generated after running 
            the benepar constituency parse pipeline.
    """

    def dfs(curr_span: Span, depth):
        print(depth * "\t", curr_span._.labels,
              (curr_span.start_char, curr_span.end_char), curr_span)
        for const in curr_span._.children:
            dfs(const, depth + 1)

    dfs(spacy_sentence_span, 0)


def get_noun_phrase_annotations(spacy_sentence_span: Span) -> List[Anno]:
    """
    From the given spacy sentence span, get all the noun-phrase annos.

    Args:
        spacy_sentence_span: (a spacy span)
            The span of one sentence generated after running 
            the benepar constituency parse pipeline.
    """
    ret = []
    for constituent in spacy_sentence_span._.constituents:
        if len(constituent._.labels) and ('NP' in constituent._.labels):
            ret.append(Anno(constituent.start_char, constituent.end_char,
                            "NounPhrase", str(constituent)))
    return ret


def raise_why():
    raise Exception("why are we using this ?")


def read_umls_file(umls_file_path):
    umls_embedding_dict = {}
    with open(umls_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line_split = line.split(',')
            assert len(line_split) == 51
            umls_id = line_split[0]
            embedding_vector = [float(val)
                                for val in line_split[1:len(line_split)]]
            umls_embedding_dict[umls_id] = embedding_vector
    return umls_embedding_dict


def get_indices_for_dict_keys(some_dict):
    key_to_index = {}
    for index, key in enumerate(some_dict.keys()):
        key_to_index[key] = index
    return key_to_index


def log_tensor(name, tensor: torch.Tensor):
    logging.debug(f"{name}, {tensor.shape}, {type(tensor)}")


def read_umls_file_small(umls_file_path):
    umls_embedding_dict = {}
    with open(umls_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line_split = line.split(',')
            assert len(line_split) == 51
            umls_id = line_split[0]
            embedding_vector = [float(val)
                                for val in line_split[1:len(line_split)]]
            umls_embedding_dict[umls_id] = embedding_vector
            break
    return umls_embedding_dict


def get_extraction(tokens_data: List[TokenData], start_offset: int, end_offset: int):
    extraction = []
    for token in tokens_data:
        if (start_offset <= token.token_start_offset) and (token.token_end_offset <= end_offset):
            extraction.append(token.token_string)
    return ' '.join(extraction)


def print_list(some_list):
    print(p_string(some_list))


def create_gate_input_file(output_file_path, sample_to_token_data: Dict[str, List[TokenData]],
                           annos_dict: Dict[str, List[Anno]], num_samples=None):
    raise_why()
    with open(output_file_path, 'w') as output_file:
        writer = csv.writer(output_file)
        header = ['sample_id', 'text', 'spans']
        writer.writerow(header)
        sample_list = list(sample_to_token_data.keys())
        if num_samples is not None:
            sample_list = sample_list[:num_samples]
        for sample_id in sample_list:
            gold_annos = annos_dict.get(sample_id, [])
            sample_data = sample_to_token_data[sample_id]
            sample_text = ''.join(get_token_strings(sample_data))
            spans = "@".join([f"{anno.begin_offset}:{anno.end_offset}" for anno in gold_annos])
            row_to_write = [sample_id, sample_text, spans]
            writer.writerow(row_to_write)


def create_gate_file(output_file_path: str, sample_to_token_data: Dict[str, List[TokenData]],
                     annos_dict: Dict[str, List[Anno]], num_samples=None) -> None:
    raise_why()
    assert output_file_path[-7:] == '.bdocjs'
    sample_list = list(sample_to_token_data.keys())
    if num_samples is not None:
        sample_list = sample_list[:num_samples]
    curr_sample_offset = 0
    document_text = ''
    all_gate_annos = []
    for sample_id in sample_list:
        sample_start_offset = curr_sample_offset
        gold_annos = annos_dict.get(sample_id, [])
        sample_data = sample_to_token_data[sample_id]
        sample_text = ' '.join(get_token_strings(sample_data)) + '\n'
        all_gate_annos.extend([(curr_sample_offset + anno.begin_offset, curr_sample_offset + anno.end_offset,
                                anno.label_type, anno.features) for anno in gold_annos])
        all_gate_annos.extend([(curr_sample_offset + anno.begin_offset, curr_sample_offset +
                                anno.end_offset, 'Span', anno.features) for anno in gold_annos])
        document_text += sample_text
        curr_sample_offset += len(sample_text)
        sample_end_offset = curr_sample_offset
        all_gate_annos.append(
            (sample_start_offset, sample_end_offset, 'Sample', {'sample_id': sample_id}))
    gate_document = Document(document_text)
    default_ann_set = gate_document.annset()
    for gate_anno in all_gate_annos:
        default_ann_set.add(int(gate_anno[0]), int(
            gate_anno[1]), gate_anno[2], gate_anno[3])
    gate_document.save(output_file_path)


def p_string(obj) -> str:
    return json.dumps(obj=obj, indent=4)


def get_spans_from_seq_labels_2_classes(predictions_sub, batch_encoding):
    span_list = []
    start = None
    for i, label in enumerate(predictions_sub):
        if label == 0:
            if start is not None:
                span_list.append((start, i - 1))
                start = None
        else:
            assert label == 1
            if start is None:
                start = i
    if start is not None:
        span_list.append((start, len(predictions_sub) - 1))
    span_list_word = [(batch_encoding.token_to_word(span[0]), batch_encoding.token_to_word(span[1])) for span in
                      span_list]
    return span_list_word


def enumerate_spans(sentence: List,
                    offset: int = 0,
                    max_span_width: int | None = None,
                    min_span_width: int = 1,
                    ) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. 
    Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width,
    which will be used to exclude spans outside of this range.

    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. 
    This allows filtering by length, regex matches, pos tags or any Spacy 
    `Token` attributes, for example.

    # Parameters

    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this 
        function can be used with strings, or Spacy `Tokens` or other 
        sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is 
        helpful if the sentence is part of a larger structure, such as a 
        document, which the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. 
        Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            spans.append((start, end))
    return spans


def get_spans_from_bio_labels(predictions_sub: List[Label], batch_encoding, batch_idx: int) \
        -> List[tuple[int, int, str]]:
    span_list = []
    start = None
    start_label = None
    for i, label in enumerate(predictions_sub):
        if label.bio_tag == BioTag.out:
            if start is not None:
                span_list.append((start, i - 1, start_label))
                start = None
                start_label = None
        elif label.bio_tag == BioTag.begin:
            if start is not None:
                span_list.append((start, i - 1, start_label))
            start = i
            start_label = label.label_type
        elif label.bio_tag == BioTag.inside:
            if (start is not None) and (start_label != label.label_type):
                span_list.append((start, i - 1, start_label))
                start = None
                start_label = None
        else:
            raise Exception(f'Illegal label {label}')
    if start is not None:
        span_list.append((start, len(predictions_sub) - 1, start_label))
    span_list_word_idx = [
        (batch_encoding.token_to_word(batch_or_token_index=batch_idx, token_index=span[0]),
         batch_encoding.token_to_word(batch_or_token_index=batch_idx, token_index=span[1]),
         span[2])
        for span in span_list
    ]
    # filter the invalid spans due to padding
    span_list_word_idx = [span for span in span_list_word_idx if (span[0] is not None) and (span[1] is not None)]
    return span_list_word_idx


def f1(TP, FP, FN) -> tuple[float, float, float]:
    if (TP + FP) == 0:
        precision = None
    else:
        precision = TP / (TP + FP)
    if (FN + TP) == 0:
        recall = None
    else:
        recall = TP / (FN + TP)
    if (precision is None) or (recall is None) or ((precision + recall) == 0):
        return 0, 0, 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall


def get_tweet_data(folder_path):
    id_to_data = {}
    data_files_list = os.listdir(folder_path)
    for filename in data_files_list:
        data_file_path = os.path.join(folder_path, filename)
        with open(data_file_path, 'r') as f:
            data = f.read()
        twitter_id = filename[:-4]
        id_to_data[twitter_id] = data
    return id_to_data


def get_mistakes_annos(mistakes_file_path) -> SampleAnnotations:
    """
    Get the annotations that correspond to mistakes for each sample using
    the given mistakes file. 

    Args:
        mistakes_file_path: the file-path representing the file(a .tsv file)
        that contains the mistakes made by a model.
    """
    df = pd.read_csv(mistakes_file_path, sep='\t')
    sample_to_annos = {}
    for _, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        annos_list.append(Anno(int(row['begin']), int(
            row['end']), row['mistake_type'], row['extraction'], {"type": row['type']}))
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos


def remove_if_exists(file_path: str):
    """
    If file exists, then remove it.

    Args:
        file_path: str
            the file path of the file we wnat to remove
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def make_sentence_samples(sample: Sample, nlp) -> List[Sample]:
    """
    Takes a sample and creates mini-samples, where each
    mini-sample represents a sentence.

    sample: (Sample)
        The sample object we want to split.
    nlp: 
        The spacy pipeline we want to use to perform sentence splitting.
    """
    ret_sent_samples = []
    spacy_doc = nlp(sample.text)
    assert spacy_doc.has_annotation("SENT_START")
    for i, sent in enumerate(spacy_doc.sents):
        annos_contained_in_sent = [anno for anno in sample.annos.gold if (
                sent.start_char <= anno.begin_offset and anno.end_offset <= sent.end_char)]
        sent_annos = []
        for contained_anno in annos_contained_in_sent:
            new_start = contained_anno.begin_offset - sent.start_char
            new_end = contained_anno.end_offset - sent.start_char
            new_extraction = sent.text[new_start:new_end]
            sent_annos.append(
                Anno(new_start, new_end, contained_anno.label_type, new_extraction))
        ret_sent_samples.append(
            Sample(sent.text, f"{sample.id}_sent_{i}", AnnotationCollection(sent_annos, []))
        )
    return ret_sent_samples


def create_mistakes_visualization(
        mistakes_file_path: str,
        mistakes_visualization_file_path: str,
        validation_samples: List[Sample]
) -> None:
    """
    Create a gate-visualization-file(.bdocjs format) that contains the mistakes
    made by a trained model.

    Args:
        - mistakes_file_path: the file path containing the mistakes of the model
        - gate_visualization_file_path: the gate visualization file path to create 
    """
    mistake_annos_dict = get_mistakes_annos(mistakes_file_path)
    combined_annos_dict = {}
    for sample in validation_samples:
        gold_annos_list = sample.annos.gold
        mistake_annos_list = mistake_annos_dict.get(sample.id, [])
        combined_list = gold_annos_list + mistake_annos_list
        for anno in combined_list:
            anno.begin_offset = int(anno.begin_offset)
            anno.end_offset = int(anno.end_offset)
        combined_annos_dict[sample.id] = combined_list
    sample_to_text_valid = {sample.id: sample.text for sample in validation_samples}
    create_visualization_file(
        mistakes_visualization_file_path,
        combined_annos_dict,
        sample_to_text_valid
    )


# TODO: Make method use the Sample data-structure.
def create_visualization_file(
        visualization_file_path: str,
        sample_to_annos: Dict[SampleId, List[Anno]],
        sample_to_text: Dict[SampleId, str]
) -> None:
    """
    Create a .bdocjs formatted file which can me directly imported into gate developer.
    We create the file using the given text and annotations.

    Args:
        visualization_file_path: str
            the path of the visualization file we want to create
        annos_dict: Dict[str, List[Anno]]
            mapping from sample ids to annotations
        sample_to_text:
            mapping from sample ids to text
    """
    assert visualization_file_path.endswith(".bdocjs")
    sample_offset = 0
    document_text = ""
    ofsetted_annos = []
    for sample_id in sample_to_annos:
        document_text += (sample_to_text[sample_id] + '\n')
        ofsetted_annos.append(Anno(sample_offset, len(
            document_text), 'Sample', '', {"id": sample_id}))
        for anno in sample_to_annos[sample_id]:
            new_start_offset = anno.begin_offset + sample_offset
            new_end_offset = anno.end_offset + sample_offset
            anno.features['orig_start_offset'] = anno.begin_offset
            anno.features['orig_end_offset'] = anno.end_offset
            ofsetted_annos.append(Anno(
                new_start_offset, new_end_offset, anno.label_type, anno.extraction, anno.features))
        sample_offset += (len(sample_to_text[sample_id]) + 1)
    gate_document = Document(document_text)
    default_ann_set = gate_document.annset()
    for ofsetted_annotation in ofsetted_annos:
        default_ann_set.add(
            int(ofsetted_annotation.begin_offset),
            int(ofsetted_annotation.end_offset),
            ofsetted_annotation.label_type,
            ofsetted_annotation.features)
    gate_document.save(visualization_file_path)


def get_tokens_from_sample(sample: Sample) -> List[str]:
    token_annos = get_token_annos_from_sample(sample)
    return [token_anno.extraction for token_anno in token_annos]


def get_tokens_from_batch(samples: List[Sample]) -> List[List[str]]:
    ret = []
    for sample in samples:
        ret.append(get_tokens_from_sample(sample))
    return ret


def get_token_annos_from_sample(sample: Sample) -> List[Anno]:
    external_annos = sample.annos.external
    token_annos = [
        anno for anno in external_annos if anno.label_type == 'Token']
    assert len(
        token_annos), f"No token annotation exists in sample {sample.id}!"
    return token_annos


def get_token_annos_from_batch(batch: List[Sample]) -> List[List[Anno]]:
    return [
        get_token_annos_from_sample(sample)
        for sample in batch
    ]


def get_annos_dict(annos_file_path: str) -> Dict[SampleId, List[Anno]]:
    """
    Read annotations for each sample from the given file and return 
    a dict from sample_ids to corresponding annotations.
    """
    assert annos_file_path.endswith(".tsv")
    df = pd.read_csv(annos_file_path, sep='\t')
    sample_to_annos = {}
    for i, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        annos_list.append(
            Anno(row['begin'], row['end'], row['type'], row['extraction']))
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos


def get_all_types(types_file_path: str, num_expected_types: int) -> List[str]:
    ret = []
    with open(types_file_path, 'r') as types_file:
        for line in types_file:
            type_name = line.strip()
            if len(type_name):
                ret.append(type_name)
    assert len(ret) == num_expected_types, f"Expected {num_expected_types} num types, " \
                                           f"but found {len(ret)} in types file."
    return ret


def get_bio_label_idx_dicts(all_types: List[str], dataset_config: DatasetConfig) -> tuple[
    Dict[Label, int], Dict[int, Label]]:
    """
    get dictionaries mapping from BIO labels to their corresponding indices.
    """
    label_to_idx_dict = {}
    for type_string in all_types:
        assert len(type_string), "Type cannot be an empty string"
        label_to_idx_dict[Label(type_string, BioTag.begin)] = len(
            label_to_idx_dict)
        label_to_idx_dict[Label(type_string, BioTag.inside)] = len(
            label_to_idx_dict)
    label_to_idx_dict[Label.get_outside_label()] = len(label_to_idx_dict)
    idx_to_label_dict = {}
    for label in label_to_idx_dict:
        idx_to_label_dict[label_to_idx_dict[label]] = label
    assert len(label_to_idx_dict) == len(idx_to_label_dict)
    assert len(label_to_idx_dict) == dataset_config.num_types * 2 + 1
    return label_to_idx_dict, idx_to_label_dict


def open_make_dirs(file_path, mode):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return open(file_path, mode)


def create_directory_structure(folder_path: str):
    """
    Creates all the directories on the given `folder_path`.
    Doesn't throw an error if directories already exist.
    Args:
        folder_path: the directory path to create.
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)


# TODO: deprecate because every dataset should have the same representation.
def parse_token_data(token_data_raw) -> TokenData:
    """
    Parse out a token data object out of the raw json.

    token_data_raw(dict): raw JSON representing a token.
    dataset(Dataset): the dataset the token belongs to.
    """
    return TokenData(
        str(token_data_raw['Sample'][0]['id']),
        token_data_raw['Token'][0]['string'],
        token_data_raw['Token'][0]['length'],
        token_data_raw['Token'][0]['startOffset'],
        token_data_raw['Token'][0]['endOffset'],
    )


# TODO:  deprecate
def read_data_from_folder(data_folder) -> Dict[str, List[TokenData]]:
    sample_to_tokens = {}
    data_files_list = os.listdir(data_folder)
    for filename in data_files_list:
        data_file_path = os.path.join(data_folder, filename)
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        for token_data in data:
            parsed_token_data = parse_token_data(token_data)
            sample_id = str(parsed_token_data.sample_id)
            sample_tokens = sample_to_tokens.get(sample_id, [])
            sample_tokens.append(parsed_token_data)
            sample_to_tokens[sample_id] = sample_tokens
    return sample_to_tokens


def get_tokens_from_file(file_path) -> Dict[SampleId, List[TokenData]]:
    """
    Read tokens for each sample from the given file
    and make them accessible in the returned dictionary.

    file_path (str): The path to the tokens file (.json formatted).
    """
    ret = {}
    with open(file_path, 'r') as f:
        data = json.load(f)
    for token_data_json in data:
        parsed_token_data = parse_token_data(token_data_json)
        sample_id = str(parsed_token_data.sample_id)
        sample_tokens = ret.get(sample_id, [])
        sample_tokens.append(parsed_token_data)
        ret[sample_id] = sample_tokens
    return ret


def get_texts(sample_text_file_path: str) -> Dict[SampleId, str]:
    with open(sample_text_file_path, 'r') as sample_text_file:
        return json.load(sample_text_file)


def get_token_strings(sample_data: List[TokenData]):
    only_token_strings = []
    for token_data in sample_data:
        only_token_strings.append(token_data.token_string)
    return only_token_strings


def get_biggest_anno_surrounding_token(annos: List[Anno], token_anno: Anno) -> List[Anno]:
    all_annos_surrounding_token = [
        anno for anno in annos
        if (anno.begin_offset <= token_anno.begin_offset) and (token_anno.end_offset <= anno.end_offset)
    ]
    if len(all_annos_surrounding_token) > 1:
        return [max(all_annos_surrounding_token, key=lambda x: (x.end_offset - x.begin_offset))]
    else:
        return all_annos_surrounding_token


def get_label_strings(sample_data: List[TokenData], annos: List[Anno]):
    ret_labels = []
    for token_data in sample_data:
        surrounding_annos = get_biggest_anno_surrounding_token(
            annos, token_data)
        if not len(surrounding_annos):
            ret_labels.append(OUTSIDE_LABEL_STRING)
        else:
            assert len(surrounding_annos) == 1
            ret_labels.append(surrounding_annos[0].label_type)
    return ret_labels


def assert_tokens_contain(token_data: List[TokenData], strings_to_check: List[str]):
    token_strings_set = set(get_token_strings(token_data))
    strings_to_check_set = set(strings_to_check)
    assert strings_to_check_set.issubset(token_strings_set)


# def get_labels_bio(sample_data: List[TokenData], annos: List[Anno], types_dict) -> List[Label]:
#     labels = get_label_strings(sample_data, types_dict)
#     offsets = get_token_offsets(sample_data)
#     new_labels = []
#     for (label_string, curr_offset) in zip(labels, offsets):
#         if label_string != OUTSIDE_LABEL_STRING:
#             anno_same_start = [anno for anno in annos if anno.begin_offset == curr_offset[0]]
#             in_anno = [anno for anno in annos if
#                        (curr_offset[0] >= anno.begin_offset) and (curr_offset[1] <= anno.end_offset)]
#             if len(anno_same_start) > 0:
#                 new_labels.append(Label(label_string, BioTag.begin))
#             else:
#                 # avoid DiseaseMid without a DiseaseStart
#                 if (len(new_labels) > 0) and (new_labels[-1].bio_tag != BioTag.out) \
#                         and (label_string == new_labels[-1].label_type) and len(in_anno):
#                     new_labels.append(Label(label_string, BioTag.inside))
#                 else:
#                     new_labels.append(Label.get_outside_label())
#         else:
#             new_labels.append(Label.get_outside_label())
#     return new_labels


def get_labels_bio(token_anno_list: List[Anno], gold_annos: List[Anno]) -> List[Label]:
    """
    Takes all tokens and gold annotations for a sample
    and outputs a labels(one for each token) representing 
    whether a token is at the beginning(B), inside(I), or outside(O) of an entity.
    """
    new_labels = []
    for token_anno in token_anno_list:
        annos_that_surround = get_biggest_anno_surrounding_token(gold_annos, token_anno)
        if not len(annos_that_surround):
            new_labels.append(Label.get_outside_label())
        else:
            assert len(annos_that_surround) == 1
            annos_with_same_start = [
                anno for anno in annos_that_surround if anno.begin_offset == token_anno.begin_offset]
            if len(annos_with_same_start):
                new_labels.append(
                    Label(annos_with_same_start[0].label_type, BioTag.begin))
            else:
                new_labels.append(
                    Label(annos_that_surround[0].label_type, BioTag.inside))
    return new_labels


def get_token_offsets(sample_data: List[TokenData]) -> List[tuple]:
    offsets_list = []
    for token_data in sample_data:
        offsets_list.append((token_data.token_start_offset,
                             token_data.token_end_offset))
    return offsets_list


def get_token_offsets_from_sample(sample: Sample) -> List[tuple]:
    token_annos = get_token_annos_from_sample(sample)
    return [(token_anno.begin_offset, token_anno.end_offset) for token_anno in token_annos]


def get_token_offsets_from_batch(batch: List[Sample]) -> List[List[tuple]]:
    ret = []
    for sample in batch:
        ret.append(get_token_offsets_from_sample(sample))
    return ret


def get_umls_data(sample_data):
    raise NotImplementedError()
    # umls_tags = []
    # for token_data in sample_data:
    #     if 'UMLS' in token_data:
    #         umls_tags.append(token_data['UMLS'])
    #     else:
    #         umls_tags.append('o')
    # return umls_tags


def get_dis_gaz_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'DisGaz' in token_data:
    #         output.append('DisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_dis_gaz_one_hot(sample_data):
    # dis_labels = get_dis_gaz_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_umls_diz_gaz_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'UMLS_Disease' in token_data:
    #         output.append('UmlsDisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_umls_dis_gaz_one_hot(sample_data):
    # dis_labels = get_umls_diz_gaz_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_silver_dis_one_hot(sample_data):
    # dis_labels = get_silver_dis_labels(sample_data)
    # return [[1, 0] if label == 'o' else [0, 1] for label in dis_labels]
    raise NotImplementedError()


def get_silver_dis_labels(sample_data):
    # output = []
    # for token_data in sample_data:
    #     if 'SilverDisGaz' in token_data:
    #         output.append('SilverDisGaz')
    #     else:
    #         output.append('o')
    # return output
    raise NotImplementedError()


def get_pos_data(sample_data):
    # pos_tags = []
    # for token_data in sample_data:
    #     pos_tags.append(token_data['Token'][0]['category'])
    # return pos_tags
    raise NotImplementedError()


def get_umls_indices(sample_data, umls_key_to_index):
    # umls_data = get_umls_data(sample_data)
    # umls_keys = [default_key if umls == 'o' else umls[0]['CUI'] for umls in umls_data]
    # default_index = umls_key_to_index[default_key]
    # umls_indices = [umls_key_to_index.get(key, default_index) for key in umls_keys]
    # return umls_indices
    raise NotImplementedError()


def get_pos_indices(sample_data, pos_key_to_index):
    # pos_tags = get_pos_data(sample_data)
    # default_index = pos_key_to_index[default_key]
    # pos_indices = [pos_key_to_index.get(tag, default_index) for tag in pos_tags]
    # return pos_indices
    raise NotImplementedError()


# class Embedding(nn.Module):
#     def __init__(self, emb_dim, vocab_size, initialize_emb, word_to_ix):
#         super(Embedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_dim).requires_grad_(False)
#         if initialize_emb:
#             inv_dic = {v: k for k, v in word_to_ix.items()}
#             for key in initialize_emb.keys():
#                 if key in word_to_ix:
#                     ind = word_to_ix[key]
#                     self.embedding.weight.data[ind].copy_(torch.from_numpy(initialize_emb[key]))

#     def forward(self, input):
#         return self.embedding(input)


# ######################################################################
# # ``PositionalEncoding`` module injects some information about the
# # relative or absolute position of the tokens in the sequence. The
# # positional encodings have the same dimension as the embeddings so that
# # the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# # different frequencies.
# #
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.unsqueeze(x, dim=1)
#         x = x + self.pe[:x.size(0)]
#         x = self.dropout(x)
#         x = torch.squeeze(x, dim=1)
#         return x


def expand_labels(batch_encoding, labels):
    """
    return a list of labels with each label in the list
    corresponding to each token in batch_encoding
    """
    new_labels = []
    for token_idx in range(len(batch_encoding.tokens())):
        word_idx = batch_encoding.token_to_word(token_idx)
        new_labels.append(labels[word_idx])
    return new_labels


def expand_labels_rich_batch(batch_encoding, labels: List[Label], batch_idx: int) -> List[Label]:
    """
    return a list of labels with each label in the list
    corresponding to each token in batch_encoding
    """
    new_labels = []
    prev_word_idx = None
    prev_label = None
    encountered_padding = False
    for token_idx, token in enumerate(batch_encoding.tokens(batch_index=batch_idx)):
        word_idx = batch_encoding.token_to_word(batch_or_token_index=batch_idx, token_index=token_idx)

        # Padding related logic
        if encountered_padding:
            assert word_idx is None
        if word_idx is None:
            assert token == '[PAD]'
            encountered_padding = True
            new_labels.append(Label.get_outside_label())
            continue  # move on if we encountered padding

        label = labels[word_idx]
        if (label.bio_tag == BioTag.begin) and (prev_word_idx == word_idx):
            assert prev_label is not None
            new_labels.append(
                Label(label_type=prev_label.label_type, bio_tag=BioTag.inside))
        else:
            new_labels.append(labels[word_idx])
        prev_word_idx = word_idx
        prev_label = label
    return new_labels


def get_token_level_spans(
        token_annos: List[Anno],
        annos_to_convert: List[Anno]) -> List[tuple]:
    """
    Convert char_offset spans to token-level spans
    """
    ret = []
    for curr_gold_anno in annos_to_convert:
        begin_token = [i for i, token_anno in enumerate(token_annos) if
                       curr_gold_anno.begin_offset == token_anno.begin_offset]

        end_token = [i for i, token_anno in enumerate(token_annos) if
                     curr_gold_anno.end_offset == token_anno.end_offset]

        if len(begin_token) and len(end_token):
            assert len(begin_token) == 1
            assert len(end_token) == 1
            ret.append((begin_token[0], end_token[0] + 1, curr_gold_anno.label_type))
    return ret


def get_sub_token_level_spans(
        token_level_spans: List[tuple],
        batch_encoding: BatchEncoding
) -> List[tuple]:
    """
    Convert token-level spans to sub-token-level spans.
    """
    ret = []
    for start_token_idx, end_token_idx, type_name in token_level_spans:
        start_sub_token_span = batch_encoding.word_to_tokens(start_token_idx)
        end_sub_token_span = batch_encoding.word_to_tokens(end_token_idx - 1)
        # TODO: Some span annotations aren't valid because bert doesn't have embeddings for them.
        if (start_sub_token_span is not None) and (end_sub_token_span is not None):
            start_sub_token_idx = start_sub_token_span.start
            end_sub_token_idx = end_sub_token_span.end
            ret.append((start_sub_token_idx, end_sub_token_idx, type_name))
    return ret
