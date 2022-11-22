import train_util
import util
import spacy
from structs import Sample, Anno
from typing import List

nlp = spacy.load("en_core_web_sm")
print(type(nlp))
sample_text = "This is a disease. This is another disease. This disease is awesome."
sample_id = "some_id"
sample_annos = [Anno(10,17,'label','disease'), Anno(35, 42, 'label', 'disease'), Anno(49, 56, 'label','disease')]
test_sample = Sample(sample_text, sample_id, sample_annos)
# spacy_doc = nlp(some_text)
# assert spacy_doc.has_annotation("SENT_START")
# start_annos = [(10,17), (35, 42), (49, 56)]
# end_annos = [(10,17), (16, 23), (5, 12)]
# for i, sent in enumerate(spacy_doc.sents):
#     print(some_text[sent.start_char: sent.end_char])
#     sent_start_offset = sent.start_char
#     print((start_annos[i][0] - sent_start_offset,start_annos[i][1] - sent_start_offset))

def make_sentence_samples(sample: Sample) -> List[Sample]:
    """
    Takes a sample and creates mini-samples, where each
    mini-sample represents a sentence.
    """
    ret_sent_samples = []
    spacy_doc = nlp(sample.text)
    assert spacy_doc.has_annotation("SENT_START")
    for i, sent in enumerate(spacy_doc.sents):
        annos_contained_in_sent = [anno for anno in sample.annos if (sent.start_char <= anno.begin_offset and anno.end_offset <= sent.end_char)]
        sent_annos = []
        for contained_anno in annos_contained_in_sent:
            new_start = contained_anno.begin_offset - sent.start_char
            new_end = contained_anno.end_offset - sent.start_char
            new_extraction = sent.text[new_start:new_end]
            sent_annos.append(Anno(new_start, new_end, contained_anno.label_type, new_extraction))
        ret_sent_samples.append(Sample(sent.text, f"{sample.id}_sent_{i}", sent_annos)) 
    return ret_sent_samples


print(make_sentence_samples(test_sample))


# for token in valid_tokens['cc_odontologia25']:
#     print(token.token_start_offset, token.token_end_offset, token.token_string)
# for anno in valid_annos['cc_odontologia25']:
#     print(anno)