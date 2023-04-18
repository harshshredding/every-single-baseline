from utils.preprocess import Preprocessor, DatasetSplit, PreprocessorRunType
from annotators import Annotator
from structs import Dataset, Sample, AnnotationCollection, Anno
from preamble import *
from random import shuffle
from utils.easy_testing import get_valid_samples_by_dataset_name, get_test_samples_by_dataset_name
from util import read_predictions_file
from collections import defaultdict

def get_ncbi_sample(sample_raw: list) -> Sample:
    title_sample_id, title_tag, title_text = sample_raw[0].split('|')
    assert title_tag == 't'
    assert len(title_text)
    #print(f"sample id: {title_sample_id}")
    #print(f"title text: {title_text}")

    abstract_sample_id, abstract_tag, abstract_text = sample_raw[1].split('|')
    assert abstract_tag == 'a'
    assert len(abstract_text)
    assert abstract_sample_id == title_sample_id 
    #print(f"abstract text: {abstract_text}")

    disease_spans = []
    for anno_row in sample_raw[2:]:
        anno_columns = anno_row.split('\t')
        assert len(anno_columns) == 6
        span_start = int(anno_columns[1])
        span_end = int(anno_columns[2])
        extraction = anno_columns[3]
        disease_spans.append((span_start, span_end, extraction))
    #print(f"disease spans \n: {disease_spans}")

    full_text = title_text + ' ' + abstract_text
    for start, end, gold_extraction in disease_spans:
        if full_text[start:end] != gold_extraction:
            print("Annotation mismatch")
            print("text", full_text[start:end])
            print("gold", gold_extraction)
            print()


    disease_annos = [
        Anno(
            begin_offset=start,
            end_offset=end,
            extraction=gold_extraction,
            label_type='Disease'
        )
        for start, end, gold_extraction in disease_spans
    ]

    return Sample(
        id=title_sample_id,
        text=full_text,
        annos=AnnotationCollection(gold=disease_annos, external=[])
    )

def get_ncbi_raw_samples(corpus_file_path) -> list[list]:
    samples = []
    curr_sample = []
    with open(corpus_file_path, 'r') as ncbi_file:
        for line in ncbi_file:
            line = line.strip()
            if not len(line):
                samples.append(curr_sample)
                curr_sample = []
            else:
                curr_sample.append(line)
    assert len(curr_sample)
    samples.append(curr_sample)
    print(f"found {len(samples)} samples")
    non_empty_samples = [sample for sample in samples if len(sample)]
    print(red(f"empty samples: {len(samples) - len(non_empty_samples)}"))
    return non_empty_samples


class PreprocessNcbiDisease(Preprocessor):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset=Dataset.ncbi_disease,
            annotators=annotators,
            dataset_split=dataset_split,
            run_mode=run_mode,
        )

    def get_samples(self) -> list[Sample]:
        match self.dataset_split:
            case DatasetSplit.train:
                corpus_file_path = './NCBItrainset_corpus.txt'
            case DatasetSplit.valid:
                corpus_file_path = './NCBIdevelopset_corpus.txt'
            case DatasetSplit.test:
                corpus_file_path = './NCBItestset_corpus.txt'
        raw_samples = get_ncbi_raw_samples(corpus_file_path=corpus_file_path)
        samples = [get_ncbi_sample(raw_sample) for raw_sample in raw_samples]
        return samples

    def get_entity_types(self) -> list[str]:
        return ['Disease']


def create_meta_sample(sample: Sample, span: tuple[int, int], label_type: str):
    extraction = sample.text[span[0]: span[1]]
    return Sample(
            text= extraction + ' [SEP] ' + sample.text,
            id=f"{sample.id}@@@{span[0]}@@@{span[1]}",
            annos=AnnotationCollection(
                gold=[Anno(
                    begin_offset=0,
                    end_offset=len(sample.text),
                    label_type=label_type,
                    extraction=extraction
                )],
                external=[]
            )
        )



def get_training_and_valid_set_for_ncbi_disease_meta():
    seq_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_valid_epoch_12_predictions.tsv'
    span_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_valid_epoch_17_predictions.tsv'
    seq_predictions = read_predictions_file(seq_predictions_file_path)
    span_predictions = read_predictions_file(span_predictions_file_path)
    gold_samples = get_valid_samples_by_dataset_name('ncbi_disease_sentence')
    print(len(gold_samples))
    gold_samples = {sample.id: sample for sample in gold_samples}
    for sample_id in seq_predictions:
        assert sample_id in gold_samples
    for sample_id in span_predictions:
        assert sample_id in gold_samples

    samples: list[Sample] = []
    for sample_id in gold_samples:
        sample = gold_samples[sample_id]
        gold_spans = set([(anno.begin_offset, anno.end_offset) for anno in sample.annos.gold])
        seq_prediction_spans = set()
        span_prediction_spans = set()
        if sample_id in seq_predictions:
            seq_prediction_spans = set([(anno.begin_offset, anno.end_offset) for anno in seq_predictions[sample_id]])
        if sample_id in span_predictions:
            span_prediction_spans = set([(anno.begin_offset, anno.end_offset) for anno in span_predictions[sample_id]])
        all_prediction_spans = seq_prediction_spans.union(span_prediction_spans)
        incorrect_prediction_spans = all_prediction_spans.difference(gold_spans)
        correct_prediction_spans = gold_spans
        assert len(incorrect_prediction_spans.intersection(correct_prediction_spans)) ==  0
        for correct_span in correct_prediction_spans:
            samples.append(
                    create_meta_sample(sample=sample, span=correct_span, label_type='correct')
            )
        for incorrect_span in incorrect_prediction_spans:
            samples.append(
                    create_meta_sample(sample=sample, span=incorrect_span, label_type='incorrect')
            )

    shuffle(samples)
    percent_85 = int(len(samples)*0.85)
    train_samples = samples[:percent_85]
    valid_samples = samples[percent_85:]
    assert len(train_samples) + len(valid_samples) == len(samples)
    return train_samples, valid_samples


def get_test_set_for_ncbi_disease_meta():
    seq_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_test_epoch_10_predictions.tsv'
    span_predictions_file_path = '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/test/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_test_epoch_15_predictions.tsv'
    seq_predictions = read_predictions_file(seq_predictions_file_path)
    span_predictions = read_predictions_file(span_predictions_file_path)
    samples = get_test_samples_by_dataset_name('ncbi_disease_sentence')

    gold_samples = {sample.id: sample for sample in samples}
    for sample_id in seq_predictions:
        assert sample_id in gold_samples
    for sample_id in span_predictions:
        assert sample_id in gold_samples

    samples: list[Sample] = []
    for sample_id in gold_samples:
        sample = gold_samples[sample_id]
        seq_prediction_spans = set()
        span_prediction_spans = set()
        if sample_id in seq_predictions:
            seq_prediction_spans = set([(anno.begin_offset, anno.end_offset) for anno in seq_predictions[sample_id]])
        if sample_id in span_predictions:
            span_prediction_spans = set([(anno.begin_offset, anno.end_offset) for anno in span_predictions[sample_id]])
        all_prediction_spans = seq_prediction_spans.union(span_prediction_spans)
        for prediction_span in all_prediction_spans:
            samples.append(
                create_meta_sample(sample=sample, span=prediction_span, label_type='correct')
            )
    shuffle(samples)
    return samples



def get_training_and_valid_set_for_ncbi_disease_meta_large():
    prediction_file_paths = [
        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_valid_epoch_8_predictions.tsv',

        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_valid_epoch_9_predictions.tsv',

        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_valid_epoch_10_predictions.tsv',

        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_valid_epoch_11_predictions.tsv',
        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_seq_large_bio_valid_epoch_12_predictions.tsv',


        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_valid_epoch_15_predictions.tsv',
        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_valid_epoch_16_predictions.tsv',
        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_valid_epoch_17_predictions.tsv',
        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_valid_epoch_18_predictions.tsv',
        '/Users/harshverma/every-single-baseline/meta/ncbi/predictions/valid/experiment_ncbi_sentence_ncbi_disease_sentence_model_span_large_bio_default_valid_epoch_19_predictions.tsv',
    ]
    all_predictions_dict = defaultdict(list)
    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            all_predictions_dict[sample_id].extend(annos)
    gold_samples = get_valid_samples_by_dataset_name('ncbi_disease_sentence')
    print(len(gold_samples))
    gold_samples = {sample.id: sample for sample in gold_samples}
    for sample_id in all_predictions_dict:
        assert sample_id in gold_samples

    meta_samples: list[Sample] = []
    for sample_id in gold_samples:
        sample = gold_samples[sample_id]
        gold_spans = set([(anno.begin_offset, anno.end_offset) for anno in sample.annos.gold])
        prediction_spans = set()
        if sample_id in all_predictions_dict:
            prediction_spans = set([(anno.begin_offset, anno.end_offset) for anno in all_predictions_dict[sample_id]])
        incorrect_prediction_spans = prediction_spans.difference(gold_spans)
        correct_prediction_spans = gold_spans
        assert len(incorrect_prediction_spans.intersection(correct_prediction_spans)) ==  0
        for correct_span in correct_prediction_spans:
            meta_samples.append(
                    create_meta_sample(sample=sample, span=correct_span, label_type='correct')
            )
        for incorrect_span in incorrect_prediction_spans:
            meta_samples.append(
                    create_meta_sample(sample=sample, span=incorrect_span, label_type='incorrect')
            )

    shuffle(meta_samples)
    percent_85 = int(len(meta_samples)*0.85)
    train_samples = meta_samples[:percent_85]
    valid_samples = meta_samples[percent_85:]
    assert len(train_samples) + len(valid_samples) == len(meta_samples)
    return train_samples, valid_samples



class PreprocessNcbiDiseaseMeta(Preprocessor):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset=Dataset.ncbi_disease,
            annotators=annotators,
            dataset_split=dataset_split,
            run_mode=run_mode,
        )

    def get_samples(self) -> list[Sample]:
        test = get_test_set_for_ncbi_disease_meta()
        train, valid = get_training_and_valid_set_for_ncbi_disease_meta()
        match self.dataset_split:
            case DatasetSplit.train:
                samples = train
            case DatasetSplit.valid:
                samples = valid
            case DatasetSplit.test:
                samples = test
        return samples

    def get_entity_types(self) -> list[str]:
        return ['correct', 'incorrect']

class PreprocessNcbiDiseaseMetaBiggerValid(Preprocessor):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset=Dataset.ncbi_disease,
            annotators=annotators,
            dataset_split=dataset_split,
            run_mode=run_mode,
        )

    def get_samples(self) -> list[Sample]:
        test = get_test_set_for_ncbi_disease_meta()
        train, valid = get_training_and_valid_set_for_ncbi_disease_meta_large()
        match self.dataset_split:
            case DatasetSplit.train:
                samples = train
            case DatasetSplit.valid:
                samples = valid
            case DatasetSplit.test:
                samples = test
        return samples

    def get_entity_types(self) -> list[str]:
        return ['correct', 'incorrect']
