from transformers import AutoModel
from transformers import AutoTokenizer
from typing import List


def get_bert_model():
    return AutoModel.from_pretrained('bert-base-uncased')


def get_bert_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def get_bert_encoding(bert_tokenizer, tokens=List[str]):
    return bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True,
                          add_special_tokens=False, truncation=True, max_length=512)
