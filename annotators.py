from abc import ABC, abstractmethod
from typing import List
import benepar
from structs import Sample, Anno, Span, AnnotationCollection
import spacy
import util
from preamble import show_progress
import requests
import bs4
import json


class Annotator(ABC):
    """
    Represents a piece of computation that annotates text in some way.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def annotate(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate the given samples.
        """
        print(f"Annotator : {self.name}")
        return self.annotate_helper(samples)

    @abstractmethod
    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate the given samples.
        """



class NounPhraseAnnotator(Annotator):
    def __init__(self) -> None:
        super().__init__("NounPhraseAnnotator")
        benepar.download('benepar_en3')
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    """
    Annotate all Noun Phrases in all samples.
    """
    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
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
        super().__init__("Token Annotator")
        self.nlp = spacy.load('en_core_web_sm')

    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate all tokens.
        """
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
    def __init__(self, window_size: int, stride: int) -> None:
        super().__init__("SlidingWindowAnnotator")
        self.window_size = window_size
        self.stride = stride
        assert stride <= window_size

    def get_subsamples(self, sample: Sample) -> List[Sample]:
        assert not len(sample.annos.external)
        ret = []
        sample_text = sample.text
        for i in range(0, len(sample_text), self.stride):
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

    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
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
        super().__init__("Google Search")

    def annotate_helper(self, samples: List[Sample]):
        print("Annotator: Google Search")
        for sample in show_progress(samples):
            sample.annos.external.append(Anno(0, len(sample.text), 'OriginalSample', 'N/A'))
            google_search_results = get_google_search_headings(sample.text)
            search_result_string = ".".join(google_search_results)
            sample.text = ".".join([sample.text, search_result_string])



def remove_period(string: str):
    if string.endswith('.'):
        return string[:-1]
    else:
        return string

def get_chatgpt_dictionary() -> set[str]:
    all_preds = []
    with open('./chatgpt_social_dis_ner_test.json', 'r') as preds_train, \
         open('./chatgpt_social_dis_ner_train.json', 'r') as preds_test, \
         open('./chatgpt_social_dis_ner_valid.json', 'r') as preds_valid:
        all_preds = json.load(preds_train) + json.load(preds_test) + json.load(preds_valid)
    print("num total preds", len(all_preds))

    all_diseases: list[str] = []
    for _, disease_list_string in all_preds:
        all_diseases.extend(disease_list_string.split(','))
    all_diseases = [disease.strip() for disease in all_diseases]
    all_diseases = [remove_period(disease) for disease in all_diseases]
    return set(all_diseases)

def get_matches(dictionary: set, sentence: str, knowlege_type: str) -> list[Anno]:
    matches = []
    for entry in dictionary:
        for i in range(len(sentence)):
            if sentence[i:].startswith(entry):
                matches.append(
                    Anno(
                        begin_offset=i, 
                        end_offset=(i + len(entry)),
                        label_type=knowlege_type,
                        extraction=entry,
                    )
                )
    return matches

def get_matches_faster(dictionary: set, sentence: str, knowlege_type: str) -> list[Anno]:
    matches = []
    for entry in dictionary:
        for i in range(len(sentence)):
            if sentence.find(entry, i) == i:
                matches.append(
                    Anno(
                        begin_offset=i, 
                        end_offset=(i + len(entry)),
                        label_type=knowlege_type,
                        extraction=entry,
                    )
                )
    return matches


def get_matches_faster_2(dictionary: set, sentence: str, knowlege_type: str) -> list[Anno]:
    matches = []
    for i in range(len(sentence)):
        for j in range(i+1, len(sentence)):
            substring = sentence[i:j]
            if substring in dictionary:
                matches.append(
                    Anno(
                        begin_offset=i, 
                        end_offset=j,
                        label_type=knowlege_type,
                        extraction=substring,
                    )
                )
    return matches


class ExternalKnowledgeAnnotator(Annotator):
    def __init__(self, dictionary: set, knowlege_type: str) -> None:
        super().__init__("ExternalKnowledgeAnnotator")
        self.dictionary = dictionary
        self.knowlege_type = knowlege_type

    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate all tokens.
        """
        for sample in show_progress(samples):
            external_knowledge_annos = get_matches_faster_2(
                                            self.dictionary,
                                            sample.text,
                                            self.knowlege_type
                                            )
            sample.annos.external.extend(external_knowledge_annos)
        return samples

def get_chatgpt_disease_annotator() -> ExternalKnowledgeAnnotator:
    chatgpt_disease_dictionary = get_chatgpt_dictionary()
    knowlege_type = 'ChatGptDisease'
    return ExternalKnowledgeAnnotator(dictionary=chatgpt_disease_dictionary, knowlege_type=knowlege_type)

