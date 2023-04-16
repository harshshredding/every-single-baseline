from abc import ABC, abstractmethod
from typing import List
import benepar
from structs import Dataset, Sample, Anno, Span, AnnotationCollection, DatasetSplit
from spacy.tokens.span import Span as SpacySpan
import spacy
import util
from preamble import show_progress
import requests
import bs4
import json
from collections import Counter
import nltk
from utils.spans import contained_in
from preamble import *

from utils.easy_testing import\
  get_test_samples_by_dataset_name,\
  get_train_samples_by_dataset_name,\
  get_valid_samples_by_dataset_name


class Annotator(ABC):
    """
    Represents a piece of computation that annotates text in some way.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def annotate(self, samples: List[Sample], dataset_split: DatasetSplit) -> List[Sample]:
        """
        Annotate the given samples.
        """
        print(f"Annotator : {self.name}")
        return self.annotate_helper(samples, dataset_split=dataset_split)

    @abstractmethod
    def annotate_helper(self, samples: List[Sample], dataset_split: DatasetSplit) -> List[Sample]:
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



def get_sentence_sample(sentence: SpacySpan, sentence_idx: int, sample: Sample) -> Sample:
    gold_annos = sample.annos.gold
    gold_annos_accross_boundary = [
        anno 
        for anno in gold_annos
        if (not contained_in(
                    outside=(sentence.start_char, sentence.end_char), 
                    inside=(anno.begin_offset, anno.end_offset))
            )
            and
            (
                (sentence.start_char <= anno.begin_offset <= sentence.end_char)
                or
                (sentence.start_char <= anno.end_offset <= sentence.end_char)
            )  
    ]
    if len(gold_annos_accross_boundary):
        print(red(f"WARN : Gold Annos accross sentence boundary \n annos: {gold_annos_accross_boundary} \n sample: {sample}"))

    gold_annos_in_sentence = [
        anno 
        for anno in gold_annos
        if contained_in(
            outside=(sentence.start_char, sentence.end_char), 
            inside=(anno.begin_offset, anno.end_offset)
        )
    ]

    annos_with_corrected_offsets = [
        Anno(
            begin_offset=(anno.begin_offset - sentence.start_char),
            end_offset=(anno.end_offset - sentence.start_char),
            label_type=anno.label_type,
            extraction=anno.extraction
            )
        for anno in gold_annos_in_sentence
    ]

    for anno in annos_with_corrected_offsets:
        if sentence.text[anno.begin_offset:anno.end_offset] != anno.extraction:
            print(f"WARN: anno sentence text does not match extraction :\n"
                  f"anno text: {sentence.text[anno.begin_offset: anno.end_offset]}\n"
                  f"extraction: {anno.extraction}\n"
                  f"sample: {sample}")

    return Sample(
        text=sentence.text,
        id=sample.id + str(sentence_idx),
        annos=AnnotationCollection(gold=annos_with_corrected_offsets, external=[])
    ) 


class SentenceAnnotator(Annotator):
    def __init__(self) -> None:
        super().__init__("SentenceAnnotator")
        self.nlp = spacy.load('en_core_web_md')

    def annotate_helper(self, samples: List[Sample], dataset_split: DatasetSplit) -> List[Sample]:
        sentence_samples = []
        for sample in show_progress(samples):
            spacy_doc = self.nlp(sample.text)
            for sent_idx, spacy_sentence in enumerate(spacy_doc.sents):
                sentence_samples.append(
                    get_sentence_sample(
                        sentence=spacy_sentence,
                        sentence_idx=sent_idx,
                        sample=sample
                    )
                )
        return sentence_samples


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



def get_focus_subsample(
        sample: Sample,
        head_span: Span,
        focus_span: Span,
        tail_span: Span,
        new_sample_id: str
    ) -> Sample:
    assert not len(sample.annos.external)
    sample_text = sample.text
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
    return Sample(
        text=(head_text + focus_text + tail_text),
        id=new_sample_id,
        annos=AnnotationCollection(gold=adjusted_annos_in_focus, external=[])
    )


class SlidingSentenceAnnotator(Annotator):
    def __init__(self, window_size=100) -> None:
        super().__init__("SlidingSentenceAnnotator")
        self.nlp = spacy.load('en_core_web_md')
        self.window_size = window_size

    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        sentence_samples = []
        for sample in show_progress(samples):
            spacy_doc = self.nlp(sample.text)
            for sent_idx, spacy_sentence in enumerate(spacy_doc.sents):
                sentence_start = spacy_sentence.start_char
                sentence_end = spacy_sentence.end_char
                head_span = Span(begin=max(sentence_start - self.window_size, 0),
                                 end=sentence_start)
                focus_span = Span(begin=sentence_start,
                                  end=sentence_end)
                tail_span = Span(begin=min(sentence_end, len(sample.text)),
                                 end=min(sentence_end + self.window_size, len(sample.text)))
                sentence_samples.append(
                    get_focus_subsample(
                        sample=sample,
                        head_span=head_span,
                        focus_span=focus_span,
                        tail_span=tail_span,
                        new_sample_id=f"{sample.id}_sentence_{sent_idx}"
                    )
                )
        return sentence_samples





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

def get_disease_dictionary_for_sample(disease_string: str):
    all_diseases = disease_string.split(',')
    all_diseases = [disease.strip() for disease in all_diseases]
    all_diseases = [remove_period(disease) for disease in all_diseases]
    return set(all_diseases)


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


class ExternalKnowledgeAnnotatorExact(Annotator):
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


class ExternalKnowledgeAnnotatorLoweredExact(Annotator):
    def __init__(self, dictionary: set, knowlege_type: str) -> None:
        super().__init__("ExternalKnowledgeAnnotator")
        self.lowered_dictionary = set([el.lower() for el in dictionary])
        self.knowlege_type = knowlege_type

    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate all tokens.
        """
        for sample in show_progress(samples):
            external_knowledge_annos = get_matches_faster_2(
                                            self.lowered_dictionary,
                                            sample.text.lower(),
                                            self.knowlege_type
                                            )
            sample.annos.external.extend(external_knowledge_annos)
        return samples


class ExternalKnowledgeAnnotatorLoweredExactWordBoundary(Annotator):
    def __init__(self, dictionary: set, knowlege_type: str) -> None:
        super().__init__("ExternalKnowledgeAnnotator")
        self.lowered_dictionary = set([el.lower() for el in dictionary])
        self.knowlege_type = knowlege_type

    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate all tokens.
        """
        for sample in show_progress(samples):
            external_knowledge_annos = get_matches_word_boundary(
                                            self.lowered_dictionary,
                                            sample.text.lower(),
                                            self.knowlege_type
                                            )
            sample.annos.external.extend(external_knowledge_annos)
        return samples


def lower_dict(dict_to_lower: set):
    return set([el.lower() for el in dict_to_lower])


def get_annos_set(
        vanilla_dataset_config_name: str,
        dataset_split: DatasetSplit):
    annos_set: set[str] = set()
    match dataset_split:
        case DatasetSplit.train:
            samples = get_train_samples_by_dataset_name(dataset_config_name=vanilla_dataset_config_name)
        case DatasetSplit.valid:
            samples = get_valid_samples_by_dataset_name(dataset_config_name=vanilla_dataset_config_name)
        case DatasetSplit.test:
            samples = get_test_samples_by_dataset_name(dataset_config_name=vanilla_dataset_config_name)
        case _:
            raise unsupported_type_error(dataset_split)
    for sample in samples:
        gold_annos = sample.annos.gold
        for gold_anno in gold_annos:
            annos_set.add(gold_anno.extraction)
    return annos_set


class UmlsDiseaseExternalKnowledgeWithGold(Annotator):
    def __init__(self, vanilla_dataset_config_name: str) -> None:
        super().__init__("ExternalKnowledgeAnnotator")
        self.umls_dict = read_umls_disease_gazetteer_dict()
        self.umls_dict = lower_dict(self.umls_dict)


        self.train_annos_dict = get_annos_set(
                                    vanilla_dataset_config_name,
                                    dataset_split=DatasetSplit.train)
        self.train_annos_dict = lower_dict(self.train_annos_dict)
        print("len train annos set", len(self.train_annos_dict))


        self.valid_annos_dict = get_annos_set(
                                    vanilla_dataset_config_name,
                                    dataset_split=DatasetSplit.valid)
        self.valid_annos_dict = lower_dict(self.valid_annos_dict)
        print("len valid annos set", len(self.valid_annos_dict))

        
        self.umls_with_train = self.umls_dict.union(self.train_annos_dict)
        self.umls_with_train_and_valid = self.umls_dict.union(self.train_annos_dict).union(self.valid_annos_dict)

        assert len(self.valid_annos_dict) < len(self.train_annos_dict)
        assert len(self.umls_dict) < len(self.umls_with_train) < len(self.umls_with_train_and_valid)

        self.knowlege_type = 'UmlsDiseaseGold'


    def annotate_helper(self, samples: List[Sample], dataset_split: DatasetSplit) -> List[Sample]:
        """
        Annotate all tokens.
        """
        match dataset_split:
            case DatasetSplit.train:
                lowered_dictionary = self.umls_dict
            case DatasetSplit.valid:
                lowered_dictionary = self.umls_with_train
            case DatasetSplit.test:
                lowered_dictionary = self.umls_with_train_and_valid
            case _:
                raise unsupported_type_error(dataset_split) 

        for term in lowered_dictionary:
            assert term.islower() or term.isnumeric()
                
        for sample in show_progress(samples):
            external_knowledge_annos = get_matches_word_boundary(
                                            lowered_dictionary,
                                            sample.text.lower(),
                                            self.knowlege_type
                                            )
            sample.annos.external.extend(external_knowledge_annos)
        return samples


def get_chatgpt_disease_list_from_string(disease_string: str) -> set[str]:
    all_diseases = disease_string.split(',')
    all_diseases = [disease.strip() for disease in all_diseases]
    all_diseases = [remove_period(disease) for disease in all_diseases]
    return set(all_diseases)


def get_chatgpt_preds_dict() -> dict[str, set[str]]:
    all_preds = []
    with open('./chatgpt_social_dis_ner_test.json', 'r') as preds_train, \
         open('./chatgpt_social_dis_ner_train.json', 'r') as preds_test, \
         open('./chatgpt_social_dis_ner_valid.json', 'r') as preds_valid:
        all_preds = json.load(preds_train) + json.load(preds_test) + json.load(preds_valid)
    
    preds_dict = {}
    for sample_id, pred in all_preds:
        if sample_id in preds_dict:
            preds_dict[sample_id] = ','.join((preds_dict[sample_id], pred))
        else:
            preds_dict[sample_id] = pred

    all_samples = get_test_samples_by_dataset_name('social_dis_ner_vanilla') + \
                  get_valid_samples_by_dataset_name('social_dis_ner_vanilla') + \
                  get_train_samples_by_dataset_name('social_dis_ner_vanilla')

    for sample in all_samples:
        assert sample.id in preds_dict
    
    preds_dict = {sample_id: get_chatgpt_disease_list_from_string(disease_string) 
                  for sample_id, disease_string in preds_dict.items()}

    return preds_dict

class ExternalKnowledgePerSampleAnnotator(Annotator):
    def __init__(
            self,
            sample_predictions_dict: dict[str, set[str]],
            knowlege_type: str):
        super().__init__("ExternalKnowledgePerSampleAnnotator")
        self.sample_predictions_dict = sample_predictions_dict
        self.knowlege_type = knowlege_type


    def annotate_helper(self, samples: List[Sample]) -> List[Sample]:
        """
        Annotate all tokens.
        """
        for sample in show_progress(samples):
            external_knowledge_annos = get_matches_faster_2(
                                            self.sample_predictions_dict[sample.id],
                                            sample.text,
                                            self.knowlege_type
                                            )
            sample.annos.external.extend(external_knowledge_annos)
        return samples



def get_stop_words():
    english_stop_words_string = "0,1,2,3,4,5,6,7,8,9,a,A,about,above,across,after,again,against,all,almost,alone,along,already,also,although,always,am,among,an,and,another,any,anyone,anything,anywhere,are,aren't,around,as,at,b,B,back,be,became,because,become,becomes,been,before,behind,being,below,between,both,but,by,c,C,can,cannot,can't,could,couldn't,d,D,did,didn't,do,does,doesn't,doing,done,don't,down,during,e,E,each,either,enough,even,ever,every,everyone,everything,everywhere,f,F,few,find,first,for,four,from,full,further,g,G,get,give,go,h,H,had,hadn't,has,hasn't,have,haven't,having,he,he'd,he'll,her,here,here's,hers,herself,he's,him,himself,his,how,however,how's,i,I,i'd,if,i'll,i'm,in,interest,into,is,isn't,it,it's,its,itself,i've,j,J,k,K,keep,l,L,last,least,less,let's,m,M,made,many,may,me,might,more,most,mostly,much,must,mustn't,my,myself,n,N,never,next,no,nobody,noone,nor,not,nothing,now,nowhere,o,O,of,off,often,on,once,one,only,or,other,others,ought,our,ours,ourselves,out,over,own,p,P,part,per,perhaps,put,q,Q,r,R,rather,s,S,same,see,seem,seemed,seeming,seems,several,shan't,she,she'd,she'll,she's,should,shouldn't,show,side,since,so,some,someone,something,somewhere,still,such,t,T,take,than,that,that's,the,their,theirs,them,themselves,then,there,therefore,there's,these,they,they'd,they'll,they're,they've,this,those,though,three,through,thus,to,together,too,toward,two,u,U,under,until,up,upon,us,v,V,very,w,W,was,wasn't,we,we'd,we'll,well,we're,were,weren't,we've,what,what's,when,when's,where,where's,whether,which,while,who,whole,whom,who's,whose,why,why's,will,with,within,without,won't,would,wouldn't,x,X,y,Y,yet,you,you'd,you'll,your,you're,yours,yourself,yourselves,you've,z,Z"
    english_stop_words_set = set([stop_word.lower() for stop_word in english_stop_words_string.split(',')])
    nltk_english = set(nltk.corpus.stopwords.words('english'))
    nltk_spanish = set(nltk.corpus.stopwords.words('spanish'))
    return set(list(english_stop_words_set) + list(nltk_english) + list(nltk_spanish))


def split_umls_entry_into_tokens(umls_string: str):
    # remove some punctuations
    umls_string = umls_string.replace(';', ' ')
    umls_string = umls_string.replace(',', ' ')
    umls_string = umls_string.replace('(', ' ')
    umls_string = umls_string.replace(')', ' ')
    umls_string = umls_string.lower()
    return umls_string.split()


def get_umls_disease_set_smart() -> set[str]:
    diseases_list = []
    stop_words = get_stop_words()
    with open('./umls_disease_gazetteer_new.lst', 'r') as umls_file:
        for line in umls_file:
            disease_string = line.strip()
            diseases_list.extend(split_umls_entry_into_tokens(disease_string))
    diseases_counter = Counter(diseases_list)
    diseases_counter = {disease: count
                        for disease, count in diseases_counter.items()
                        if len(disease) > 1}
    diseases_counter = {disease: count
                        for disease, count in diseases_counter.items()
                        if disease not in stop_words}
    counts_list = [(count, disease) 
                           for disease, count in diseases_counter.items()]
    sorted_counts_list =  sorted(counts_list, reverse=True)
    disease_set =  set([disease for _, disease in sorted_counts_list])
    assert len(disease_set) == 88215
    return disease_set


def remove_extra_info(disease_string: str):
    if ';' in disease_string:
        return disease_string[:disease_string.find(';')]
    elif ',' in disease_string:
        return disease_string[:disease_string.find(',')]
    else:
        return disease_string

def read_umls_disease_gazetteer_dict():
    disease_list = []
    with open('./umls_disease_gazetteer_new.lst', 'r') as umls_file:
        for line in umls_file:
            disease_list.append(line.strip())
    list_without_extra_info = [remove_extra_info(disease_string) for disease_string in disease_list]
    return set(list_without_extra_info)

def has_word_boundaries(anno: Anno, sentence: str) -> bool:
    assert anno.begin_offset < anno.end_offset
    if anno.begin_offset > 0:
        if sentence[anno.begin_offset - 1].isalpha():
            return False
    if anno.end_offset < len(sentence):
        if sentence[anno.end_offset].isalpha():
            return False
    return True

def get_matches_word_boundary(dictionary: set, sentence: str, knowlege_type: str):
    matches = get_matches_faster_2(dictionary, sentence, knowlege_type)
    matches_with_word_boundaries = [match for match in matches if has_word_boundaries(match, sentence)]
    return matches_with_word_boundaries

def get_chatgpt_disease_annotator() -> ExternalKnowledgeAnnotatorExact:
    chatgpt_disease_dictionary = get_chatgpt_dictionary()
    knowlege_type = 'ChatGptDisease'
    return ExternalKnowledgeAnnotatorExact(dictionary=chatgpt_disease_dictionary, knowlege_type=knowlege_type)


def get_chatgpt_per_sample_disease_annotator() -> ExternalKnowledgePerSampleAnnotator:
    chatgpt_disease_dictionary = get_chatgpt_preds_dict()
    knowlege_type = 'ChatGptDiseasePerSample'
    return ExternalKnowledgePerSampleAnnotator(
            sample_predictions_dict=chatgpt_disease_dictionary,
            knowlege_type=knowlege_type
            )


def get_umls_disease_annotator_exact():
    umls_disease_dictionary = read_umls_disease_gazetteer_dict()
    knowlege_type = 'UmlsExact'
    return ExternalKnowledgeAnnotatorExact(
        dictionary=umls_disease_dictionary,
        knowlege_type=knowlege_type
    )

def get_umls_disease_annotator_lowered_exact():
    umls_disease_dictionary = read_umls_disease_gazetteer_dict()
    knowlege_type = 'UmlsExactLowered'
    return ExternalKnowledgeAnnotatorLoweredExact(
        dictionary=umls_disease_dictionary,
        knowlege_type=knowlege_type
    )

def get_umls_disease_annotator_lowered_exact_word_boundaries():
    umls_disease_dictionary = read_umls_disease_gazetteer_dict()
    knowlege_type = 'UmlsExactLoweredWordBoundary'
    return ExternalKnowledgeAnnotatorLoweredExactWordBoundary(
        dictionary=umls_disease_dictionary,
        knowlege_type=knowlege_type
    )

def get_umls_disease_smart_exact_word_boundaries_annotator():
    umls_disease_dictionary = get_umls_disease_set_smart()
    knowlege_type = 'UmlsDiseaseSmart'
    return ExternalKnowledgeAnnotatorLoweredExactWordBoundary(
        dictionary=umls_disease_dictionary,
        knowlege_type=knowlege_type
    )

def get_bigger_sliding_window_annotator():
    return SlidingWindowAnnotator(window_size=200, stride=100)

def get_sentence_annotator():
    return SentenceAnnotator()

def get_sliding_sentence_annotator():
    return SlidingSentenceAnnotator()

def get_umls_with_gold_annotator(vanilla_dataset_config_name):
    return UmlsDiseaseExternalKnowledgeWithGold(vanilla_dataset_config_name=vanilla_dataset_config_name)
