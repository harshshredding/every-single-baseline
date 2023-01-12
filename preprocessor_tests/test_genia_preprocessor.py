from preprocessors.genia_preprocessor import get_first_sample_soup, get_text, get_annos


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
