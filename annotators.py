from abc import ABC, abstractmethod
from typing import List
import benepar
from structs import Sample
import spacy
import util

class Annotator(ABC):
    """
    Represents a piece of computation that annotates text in some way.
    """
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def annotate(self, samples: List[Sample]):
        """
        Annotate the given samples.
        """
    

class NounPhraseAnnotator(Annotator):
    """
    Annotate all Noun Phrases in all samples.
    """
    def annotate(self, samples: List[Sample]):
        benepar.download('benepar_en3')
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        for sample in samples:
            spacy_doc = nlp(sample.text)
            for sent in spacy_doc.sents:
                noun_phrase_annos = util.get_noun_phrase_annotations(sent)
                sample.annos.external.extend(noun_phrase_annos)