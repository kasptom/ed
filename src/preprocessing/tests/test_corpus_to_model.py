from unittest import TestCase

from src.preprocessing.create_corpus import create_corpus
from src.preprocessing.corpus_to_model import corpus_to_model


class TestCorpus_to_model(TestCase):
    def test_corpus_to_model(self):
        corpus, negatives_no, positives_no = create_corpus()
        model = corpus_to_model(corpus)
        print(model.wv.similarity('old', 'yes'))
        print(model.wv.similarity('young', 'yes'))
        print(model.wv.similarity('man', 'woman'))
