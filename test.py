import benepar
benepar.download('benepar_en3')
import spacy
from spacy.tokens.span import Span
nlp = spacy.load('en_core_web_md')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
doc = nlp("he rode a scott motorcycle winning several trophies")
sent = list(doc.sents)[0]
print("\n"*3)
print(sent._.parse_string)
print("\n"*3)
# print(type(sent._))
# second_part : Span = list(sent._.children)[1]
# print(second_part.start_char, second_part.end_char)
# print(len("he rode a scott motorcycle winning several trophies"))
