from typing import List

import stanza
from transformers import AutoTokenizer

from structs import Anno, Sample

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency')
doc = nlp("he rode a scott motorcycle winning several trophies")
for sentence in doc.sentences:
    print(sentence.constituency)

