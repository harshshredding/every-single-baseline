from preprocessors.genia_preprocessor import get_first_sample_soup, get_text, get_annos, get_samples
from preprocessors.genia_preprocessor import get_parent_annos
from structs import DatasetSplit


def test_something():
    first_sample_soup = get_first_sample_soup()
    assert "IL-2 gene" in str(first_sample_soup)
    assert "requires reactive oxygen" in str(first_sample_soup)


def test_get_text():
    first_sample_soup = get_first_sample_soup()
    first_sample_text = get_text(first_sample_soup)
    print(first_sample_text)
    assert "IL-2 gene" in first_sample_text
    assert "requires reactive oxygen" in first_sample_text


def test_get_annos():
    first_sample_soup = get_first_sample_soup()
    annos = get_annos(first_sample_soup)
    assert len(annos) == 6
    assert any([anno.extraction == 'IL-2 gene' for anno in annos])
    assert any([anno.label_type == 'G#DNA_domain_or_region' for anno in annos])
    assert len([anno
                for anno in annos
                if anno.label_type == 'G#other_name']) == 2


def test_get_samples():
    all_train_samples = get_samples(DatasetSplit.train)
    assert "IL-2 gene" in all_train_samples[0].text
    assert "Activation of the" in all_train_samples[1].text
    assert "primary T lymphocytes we show" in all_train_samples[2].text
    assert len(all_train_samples) == 9273
    all_valid_samples = get_samples(DatasetSplit.valid)
    assert len(all_valid_samples) == 4636


def test_get_parent_annos():
    first_sample_soup = get_first_sample_soup()
    annos = get_annos(first_sample_soup)
    parent_annos = get_parent_annos(annos)
    assert len(parent_annos) == 4
    assert parent_annos[0].label_type == 'dna'
    assert parent_annos[3].label_type == 'protein'
    print(parent_annos)
