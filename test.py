import train_util
import util
import spacy
from structs import Sample, Anno
from typing import List
from transformers import AutoTokenizer
from args import args

train_texts = train_util.get_train_texts()
train_tokens = train_util.get_train_tokens()
bert_tokenizer = AutoTokenizer.from_pretrained(args['bert_model_name'])
text = train_texts['caso_clinico_neurologia185_sent_59']
tokens = train_tokens['caso_clinico_neurologia185_sent_59']
print(f"text: {text}, tokens: {tokens}")
batch_encoding = bert_tokenizer(util.get_token_strings(tokens), return_tensors="pt", is_split_into_words=True,
                                        add_special_tokens=False, truncation=True, max_length=512)
print(len(batch_encoding.words()))
# for token in valid_tokens['cc_odontologia25']:
#     print(token.token_start_offset, token.token_end_offset, token.token_string)
# for anno in valid_annos['cc_odontologia25']:
#     print(anno)