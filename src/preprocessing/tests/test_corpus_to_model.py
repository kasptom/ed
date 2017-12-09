from unittest import TestCase

from src.preprocessing.create_corpus import create_corpus
from src.preprocessing.load_model import corpus_to_model


class TestCorpus_to_model(TestCase):
    def test_corpus_to_model(self):
        corpus = create_corpus()
        model = corpus_to_model(corpus)
