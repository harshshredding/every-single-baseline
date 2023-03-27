import train_util
from utils.easy_testing import get_dataset_config_by_name, get_roberta_tokenizer
from train_util import prepare_model
from utils.config import get_experiment_config
from preamble import *
from structs import Anno, Sample
import transformers


def check_if_tokens_overlap(token_annos: List[Option[Anno]], sample_id: str):
    for idx_curr, curr_anno in enumerate(token_annos):
        for idx_other, other_anno in enumerate(token_annos):
            if (idx_curr != idx_other) and (curr_anno.is_something() and other_anno.is_something()):
                no_overlap =  (curr_anno.get_value().end_offset <= other_anno.get_value().begin_offset) \
                    or (other_anno.get_value().end_offset <= curr_anno.get_value().begin_offset)
                if not no_overlap:
                    assert ((curr_anno.get_value().end_offset - curr_anno.get_value().begin_offset) == 1) \
                            or ((other_anno.get_value().end_offset - other_anno.get_value().begin_offset) == 1) \
                            , f"one of the annos needs to be the roberta space character {curr_anno}, {other_anno}"
                    raise RuntimeError(f"token annos should never overlap"
                                       f"\n annos: {(curr_anno.get_value(), other_anno.get_value())}"
                                       f"\n sampleId: {sample_id}"
                                       )

def remove_roberta_overlaps(tokens_batch: List[List[Option[Anno]]], pretrained_model_name) \
    -> List[List[Option[Anno]]]:
    if 'roberta' in pretrained_model_name:
        tokens_batch_without_overlaps = []
        for tokens in tokens_batch:
            tokens_without_overlap = []
            for curr_token_idx in range(len(tokens) - 1):
                curr_token = tokens[curr_token_idx]
                next_token = tokens[curr_token_idx + 1]
                if (curr_token.is_something() and next_token.is_something()) and \
                   (curr_token.get_value().begin_offset == next_token.get_value().begin_offset):
                    assert (curr_token.get_value().end_offset - curr_token.get_value().begin_offset) == 1
                    tokens_without_overlap.append(Option(None))
                else:
                    tokens_without_overlap.append(curr_token)
            tokens_without_overlap.append(tokens[len(tokens) - 1])
            assert len(tokens_without_overlap) == len(tokens)
            tokens_batch_without_overlaps.append(tokens_without_overlap)
        return tokens_batch_without_overlaps
    else:
        return tokens_batch


def get_token_annos_batch(bert_encoding, samples: List[Sample]) -> List[List[Option[Anno]]]:
    expected_batch_size = len(samples)
    token_ids_matrix = bert_encoding['input_ids']
    batch_size = len(token_ids_matrix)
    num_tokens = len(token_ids_matrix[0])
    for batch_idx in range(batch_size):
        assert len(token_ids_matrix[batch_idx]) == num_tokens, "every sample should have the same number of tokens"
    assert batch_size == expected_batch_size
    token_annos_batch: List[List[Option[Anno]]] = []
    for batch_idx in range(batch_size):
        char_spans: List[Option[transformers.CharSpan]] = [
            Option(bert_encoding.token_to_chars(batch_or_token_index=batch_idx, token_index=token_idx))
            for token_idx in range(num_tokens)
        ]

        token_annos_batch.append(
            [
                Option(Anno(begin_offset=span.get_value().start, end_offset=span.get_value().end,
                            label_type='BertTokenAnno', extraction=None))
                if span.state == OptionState.Something else Option(None)
                for span in char_spans
            ]
        )

    # remove overlapping roberta tokens
    token_annos_batch = remove_roberta_overlaps(tokens_batch=token_annos_batch,
                                                pretrained_model_name='xlm-roberta-base')

    # check no token overlaps
    for token_annos, sample in zip(token_annos_batch, samples):
        check_if_tokens_overlap(token_annos, sample.id)

    return token_annos_batch

test_samples_new = train_util.get_test_samples(get_dataset_config_by_name('multiconer_fine_vanilla'))
tokenizer = get_roberta_tokenizer()
offending_sample = [sample for sample in test_samples_new if sample.id == '92f1b98c-a522-49d7-aa4f-8ed6ce22726d']
offending_sample = offending_sample[0]
be = tokenizer([offending_sample.text], return_tensors='pt')
print(offending_sample)
print([be.token_to_chars(token_idx) for token_idx in range(len(be.tokens()))])
print(be.tokens())
get_token_annos_batch(bert_encoding=be, samples=[offending_sample])
#
#seq_new_experiment = get_experiment_config(
#    model_config_module_name='model_seq_large_default',
#    dataset_config_name='multiconer_fine_vanilla'
#)
#
#new_seq_model = prepare_model(seq_new_experiment.model_config, seq_new_experiment.dataset_config)
#new_seq_model.eval()

#print(type(new_seq_model))

#for sample in test_samples_new:
#    loss_new, predictions_new = new_seq_model([sample])
