from unittest import TestCase

from src.preprocessing.create_corpus import create_corpus_and_labels
from src.preprocessing.w2v_loader import create_w2v_from_corpus, load_google_w2v_model


class TestCorpusW2VModels(TestCase):
    def test_corpus_to_model(self):
        corpus, labels = create_corpus_and_labels()
        model = create_w2v_from_corpus(corpus)
        google_model = load_google_w2v_model()

        print(model.wv.similarity('old', 'yes'))
        print(model.wv.similarity('young', 'yes'))
        print(model.wv.similarity('man', 'woman'))

        print("woman - king + man = ...")
        most_similar = google_model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
        print(most_similar)
