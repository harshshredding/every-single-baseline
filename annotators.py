from abc import ABC, abstractmethod
from typing import List
import benepar
from structs import Sample
import spacy
import util
from preamble import show_progress

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
    def __init__(self) -> None:
        super().__init__()
        benepar.download('benepar_en3')
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    """
    Annotate all Noun Phrases in all samples.
    """
    def annotate(self, samples: List[Sample]):
        print("Annotator: Noun Phrases")
        for sample in show_progress(samples):
            try:
                spacy_doc = self.nlp(sample.text)
            except StopIteration:
                print(f"StopIteration: Cannot annotate sample {sample.id} with noun-phrases")
            except ValueError as e:
                print(e)
                print(f"ValueError: Cannot annotate sample {sample.id} with noun-phrases.")
            else:
                for sent in spacy_doc.sents:
                    noun_phrase_annos = util.get_noun_phrase_annotations(sent)
                    sample.annos.external.extend(noun_phrase_annos)
