import util
from structs import TokenData, Anno
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from structs import Sample, Anno, AnnotationCollection


def test_token_level_spans():
    assert len(util.get_token_level_spans([], [])) == 0
    # text  = "This is Harsh"
    token_data = [TokenData("sample_id", "This", 4, 0, 4),
                  TokenData("sample_id", "is", 2, 5, 7),
                  TokenData("sample_id", "Harsh", 5, 8, 13)
                  ]
    sample_annos = [Anno(8, 13, 'name', 'Harsh')]
    assert util.get_token_level_spans(
        token_data, sample_annos) == [(2, 3, 'name')]

    # text = "This is Harsh Verma"
    token_data = [TokenData("sample_id", "This", 4, 0, 4),
                  TokenData("sample_id", "is", 2, 5, 7),
                  TokenData("sample_id", "Harsh", 5, 8, 13),
                  TokenData("sample_id", "Verma", 5, 14, 19)
                  ]
    sample_annos = [Anno(8, 13, 'name', 'Harsh'), Anno(
        14, 19, 'name', 'Verma'), Anno(8, 19, 'name', 'Harsh Verma')]
    assert util.get_token_level_spans(token_data, sample_annos) == [
        (2, 3, 'name'), (3, 4, 'name'), (2, 4, 'name')]


def test_sub_token_level_spans():
    # [0, 0, 0, 0, 0, 1, 2, 3]
    tokens = ["a-complex-token", "is", "a", "sentence"]
    token_level_spans = [(0, 1, 'type-1'), (3, 4, 'type-2')]
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch_encoding: BatchEncoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                                                   add_special_tokens=False, truncation=True, max_length=512)
    assert util.get_sub_token_level_spans(token_level_spans, batch_encoding) == [
        (0, 5, 'type-1'), (7, 8, 'type-2')]


def test_get_all_types():
    all_types = util.get_all_types('util_tests/test_types.txt', 3)
    assert len(all_types) == 3
    assert all_types == ['test_type_1', 'test_type_2', 'test_type_3']


def test_read_and_write_samples():
    test_sample = Sample("sample_text", "some_id",
                         AnnotationCollection(
                             gold=[
                                 Anno(1, 10, 'DISEASE', 'HIV')
                             ],
                             external=[]
                         )
                         )
    test_sample_list = [test_sample]
    output_json_file_path = "./util_tests/test_samples.json"
    util.write_samples(test_sample_list, output_json_file_path)
    samples = util.read_samples(output_json_file_path)
    assert len(samples) == 1
    assert samples[0].id == 'some_id'
    assert samples[0].text == 'sample_text'
    assert len(samples[0].annos.external) == 0
    assert len(samples[0].annos.gold) == 1
    assert samples[0].annos.gold[0].begin_offset == 1
    assert samples[0].annos.gold[0].end_offset == 10 
    assert samples[0].annos.gold[0].label_type == 'DISEASE'
    assert samples[0].annos.gold[0].extraction == 'HIV'
    assert samples[0].annos.gold[0].features == {}

    
    another_test_sample = Sample("sample_text", "some_id",
                         AnnotationCollection(
                             gold=[
                                 Anno(1, 10, 'DISEASE', 'HIV')
                             ],
                             external=[
                                Anno(0, 2, 'Name', 'John')
                             ]
                         )
                         )
    test_sample_list = [test_sample, another_test_sample]
    util.write_samples(test_sample_list, output_json_file_path)
    samples = util.read_samples(output_json_file_path)
    assert len(samples) == 2
    assert samples[1].annos.external[0].label_type == 'Name'