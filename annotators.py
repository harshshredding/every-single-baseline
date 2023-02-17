from abc import ABC, abstractmethod
from typing import List
import benepar
from structs import Sample, Anno, Span, AnnotationCollection
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
    def annotate(self, samples: List[Sample]) -> List[Sample]:
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

    def annotate(self, samples: List[Sample]) -> List[Sample]:
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
        return samples


class TokenAnnotator(Annotator):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')

    def annotate(self, samples: List[Sample]) -> List[Sample]:
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
        return samples


class SlidingWindowAnnotator(Annotator):
    def __init__(self, window_size=100) -> None:
        super().__init__()
        self.window_size = window_size

    def get_subsamples(self, sample: Sample) -> List[Sample]:
        assert not len(sample.annos.external)
        ret = []
        sample_text = sample.text
        for i in range(0, len(sample_text), self.window_size):
            head_span = Span(begin=max(i - self.window_size, 0),
                             end=i)
            focus_span = Span(begin=i,
                              end=min(i + self.window_size, len(sample_text)))
            tail_span = Span(begin=min(i + self.window_size, len(sample_text)),
                             end=min(i + (self.window_size * 2), len(sample_text)))
            annos_in_focus = [anno for anno in sample.annos.gold
                              if (focus_span.begin <= anno.begin_offset) and (anno.end_offset <= focus_span.end)]
            head_text = sample_text[head_span.begin:head_span.end] + ' [SEP] '
            focus_text = sample_text[focus_span.begin:focus_span.end]
            tail_text = ' [SEP] ' + sample_text[tail_span.begin:tail_span.end]
            adjusted_annos_in_focus = [
                Anno(
                    begin_offset=(anno.begin_offset - focus_span.begin + len(head_text)),
                    end_offset=(anno.end_offset - focus_span.begin + len(head_text)),
                    label_type=anno.label_type,
                    extraction=anno.extraction,
                    features=anno.features
                )
                for anno in annos_in_focus
            ]
            ret.append(
                Sample(
                    text=(head_text + focus_text + tail_text),
                    id=(sample.id + '_subsample_' + str(i)),
                    annos=AnnotationCollection(gold=adjusted_annos_in_focus, external=[])
                )
            )
        return ret

    def annotate(self, samples: List[Sample]) -> List[Sample]:
        """
        For a given sample, generate sub-samples while sliding the
        window over it.
        """
        print("Annotator Sliding Window")
        ret = []
        for sample in show_progress(samples):
            ret.extend(self.get_subsamples(sample))
        return ret


def google_get(query_plain):
    with requests.session() as c:
        url = 'https://www.google.com/search'
        query = {'q': query_plain}
        return requests.get(url, params=query, headers={'User-agent': 'your bot 0.1'})


def get_google_search_headings(query_string_plain: str) -> List[str]:
    request_result = google_get(query_string_plain)
    print(request_result)

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
