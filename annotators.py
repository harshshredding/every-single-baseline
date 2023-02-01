from abc import ABC, abstractmethod
from typing import List
import benepar
from structs import Sample, Anno
import spacy
import util
from preamble import show_progress
import requests
import bs4


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


class TokenAnnotator(Annotator):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')

    def annotate(self, samples: List[Sample]):
        """
        Annotate all tokens.
        """
        print("Annotator: Tokenizer")
        for sample in show_progress(samples):
            tokenized_doc = self.nlp(sample.text, disable=['parser', 'tagger', 'ner'])
            token_annos = []
            for token in tokenized_doc:
                start_offset = token.idx
                end_offset = start_offset + len(token)
                token_annos.append(
                    Anno(start_offset, end_offset, "Token", str(token))
                )
            sample.annos.external.extend(token_annos)


def google_get(query_plain):
    with requests.session() as c:
        url = 'https://www.google.com/search'
        query = {'q': query_plain}
        return requests.get(url, params=query)


def get_google_search_headings(query_string_plain: str) -> List[str]:
    request_result = google_get(query_string_plain)
    html_text = request_result.text

    # H3 with classes LC20lb MBeuO DKV0Md
    soup = bs4.BeautifulSoup(html_text, "html.parser")
    heading_objects = soup.find_all("h3")

    return [heading_object.getText() for heading_object in heading_objects]


class GoogleSearch(Annotator):
    def __init__(self) -> None:
        super().__init__()

    def annotate(self, samples: List[Sample]):
        print("Annotator: Google Search")
        for sample in show_progress(samples):
            sample.annos.external.append(Anno(0, len(sample.text), 'OriginalSample', 'N/A'))
            google_search_results = get_google_search_headings(sample.text)
            search_result_string = ".".join(google_search_results)
            sample.text = ".".join([sample.text, search_result_string])
