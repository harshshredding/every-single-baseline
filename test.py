import benepar
import spacy
from spacy.tokens.span import Span
from typing import List
from structs import Anno, AnnotationCollection, Sample
import json
from transformers import AutoTokenizer

# def get_all_noun_phrases(span) -> List[Span]:
#     ret = []
#     child: Span
#     for child in span._.children:
#         child_labels = child._.labels
#         if 'NP' in child_labels:
#             ret.append(child)
#         ret.extend(get_all_noun_phrases(child))
#     return ret

# def get_noun_phrase_annos(span) -> List[Anno]:
#     noun_phrases = get_all_noun_phrases(span)
#     noun_phrase: Span
#     return [Anno(noun_phrase.start_char, noun_phrase.end_char,
#                  'NounPhrase', str(noun_phrase), {})
#             for noun_phrase in noun_phrases]

# def print_sentence_info(sentence):
#     print(sentence._.parse_string)
#     print()
#     all_noun_phrases = get_all_noun_phrases(sentence)
#     for noun_phrase in all_noun_phrases:
#         print(noun_phrase)
#     print("-"*10)

benepar.download('benepar_en3')
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

doc = nlp("This is a sentence."\
          "And this is another sentence."\
          "I love apples and bananas."\
          "John loves Alice"\
          )

for sent in doc.sents:
    print_sentence_info(sent)
    annos = get_noun_phrase_annos(sent)
    print(annos)

