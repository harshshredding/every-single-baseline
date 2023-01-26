from models import SpanBertNounPhrase
from utils.config import get_model_config_by_name, get_dataset_config_by_name
import util

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


dataset_config = get_dataset_config_by_name('multiconer_coarse')
model_config = get_model_config_by_name("SpanBertNounPhrase")
all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
model = SpanBertNounPhrase(all_types, model_config)
all_samples = util.read_samples(dataset_config.valid_samples_file_path)
one_sample = all_samples[0]
model(one_sample)