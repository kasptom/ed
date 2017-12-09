from unittest import TestCase

from src.preprocessing.w2v_preprocessor import corpus_to_vectors


class TestCorpus_to_vectors(TestCase):
    def test_corpus_to_vectors(self):
        (x_tr, y_tr), (x_tst, y_tst) = corpus_to_vectors()
        print(len(x_tr))
        print(len(y_tr))
