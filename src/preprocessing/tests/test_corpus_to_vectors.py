from unittest import TestCase

from gensim import corpora

from src.preprocessing.create_corpus import create_corpus
from src.preprocessing.w2v_preprocessor import corpus_to_vectors, _tfidf


class TestCorpus_to_vectors(TestCase):
    def test_corpus_to_vectors(self):
        (x_tr, y_tr), (x_tst, y_tst) = corpus_to_vectors()
        print(len(x_tr))
        print(len(y_tr))

    def test_tfidf(self):
        corpus, n, p = create_corpus()
        tfidf = _tfidf(corpus)
        dictionary = corpora.Dictionary(corpus)
        print(corpus[12])
        print(tfidf[dictionary.doc2bow(corpus[12])])
