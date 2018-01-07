from unittest import TestCase

from src.preprocessing.create_corpus import create_corpus_and_labels
from src.preprocessing.doc2vec_preprocessor import _dictionary
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

        google_most_similar = google_model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
        print("google: ", google_most_similar)

        most_similar = model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
        print("model: ", most_similar)

    def test_google_model_word_occurrence_percentage(self):
        corpus, labels = create_corpus_and_labels()
        dictionary = _dictionary(corpus)

        google_model = load_google_w2v_model()
        model = create_w2v_from_corpus(corpus)

        counter = 0.0
        counter_2 = 0.0

        for word in dictionary.token2id.keys():
            if word in google_model:
                counter += 1
            if word in model.wv.vocab:
                counter_2 += 1

        print("[%] of words in google model", round((counter / len(dictionary.token2id.keys())) * 100, 2))
        print("[%] of words in generated model", round((counter_2 / len(dictionary.token2id.keys())) * 100, 2))
        print(len(dictionary))
        print(len(dictionary.token2id.keys()))
