import util
from preamble import *


def test_preprocessed_data():
    samples = util.read_samples('./preprocessed_data/multiconer_fine_test_samples.json')
    first_sample = samples[0]
    assert 'the species was described' in first_sample.text
    assert len(first_sample.annos.external) == 14
