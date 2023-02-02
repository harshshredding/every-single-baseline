import submission.multiconer_submission as multiconer_submission
from structs import Anno


def test_read_multiconer_predictions():
    predicted_annotations = multiconer_submission.read_multiconer_predictions()
    assert len(predicted_annotations) > 240000
    assert len(predicted_annotations['64dff538-f8c0-49c0-adfa-4cc485c4a1d9']) == 2
    assert len(predicted_annotations['f03ef299-90d3-4a15-b139-73b952b26313']) == 2
    assert len(predicted_annotations['954c29b0-3c88-4156-bad7-59cd38220643']) == 1
    annos = predicted_annotations['954c29b0-3c88-4156-bad7-59cd38220643']
    assert annos[0].label_type == 'OtherPER' and \
           annos[0].extraction == 'reitter' and \
           annos[0].begin_offset == 22 and \
           annos[0].end_offset == 29


def test_remove_nesting_1():
    annos = [
        Anno(0, 2, '', ''),
        Anno(0, 1, '', ''),
        Anno(0, 3, '', ''),
        Anno(0, 5, '', ''),
        Anno(0, 6, '', ''),
    ]
    remaining_annos = multiconer_submission.remove_nesting(annos)
    assert len(remaining_annos) == 1
    remaining_anno = remaining_annos[0]
    assert remaining_anno.begin_offset == 0 and remaining_anno.end_offset == 1


def test_remove_nesting_2():
    annos = [
        Anno(0, 2, '', ''),
        Anno(2, 4, '', ''),
        Anno(0, 3, '', ''),
        Anno(2, 5, '', ''),
    ]
    remaining_annos = multiconer_submission.remove_nesting(annos)
    assert len(remaining_annos) == 2
    assert remaining_annos[0].begin_offset == 0 and remaining_annos[0].end_offset == 2
    assert remaining_annos[1].begin_offset == 2 and remaining_annos[1].end_offset == 4


def test_remove_nesting_3():
    annos = [
        Anno(0, 2, '', ''),
        Anno(1, 3, '', ''),
        Anno(2, 4, '', ''),
        Anno(3, 5, '', ''),
    ]
    remaining_annos = multiconer_submission.remove_nesting(annos)
    assert len(remaining_annos) == 4
    assert remaining_annos[0].begin_offset == 0 and remaining_annos[0].end_offset == 2
    assert remaining_annos[1].begin_offset == 1 and remaining_annos[1].end_offset == 3


def test_get_predictions_without_nesting():
    original_predictions = multiconer_submission.read_multiconer_predictions()
    predictions_without_nesting = multiconer_submission.get_predictions_without_nesting()
    assert len(original_predictions) == len(predictions_without_nesting)
    for sample_id in original_predictions:
        assert sample_id in predictions_without_nesting
        assert len(original_predictions[sample_id]) >= len(predictions_without_nesting[sample_id])
    assert len(original_predictions['37e6d979-b4ec-4bd4-a8b1-ffac861306a3']) \
           > len(predictions_without_nesting['37e6d979-b4ec-4bd4-a8b1-ffac861306a3'])


def test_get_test_token_dict():
    token_dict = multiconer_submission.get_test_token_dict()
    assert len(token_dict) == 247947
    assert list(token_dict)[0] == 'd9b2af7f-1110-468e-a64c-75a3c7d07f6a'
    predictions = multiconer_submission.get_predictions_without_nesting()
    assert list(predictions)[0] == list(token_dict)[0]


def test_get_num_samples_in_test_file():
    assert multiconer_submission.get_num_samples_in_test_data() == 247947
