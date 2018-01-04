import logging
from gensim import models
from gensim.models import Word2Vec

from src.preprocessing.configuration import WORD_NUMERIC_VECTOR_SIZE, CORPUS_FILES, GOOGLE_NEWS_WORD_LIMIT
from src.utils.get_file import full_path

_WORD2VEC_MODEL_FILENAME = full_path("data/w2v_" + CORPUS_FILES['label'] + "_model")


def corpus_to_model(corpus):
    try:
        model = Word2Vec.load(_WORD2VEC_MODEL_FILENAME)
    except FileNotFoundError:
        print("File does not exist - creating the model")
        model = Word2Vec(corpus, size=WORD_NUMERIC_VECTOR_SIZE)
        model.train(corpus, total_examples=len(corpus), epochs=10)
        model.save(_WORD2VEC_MODEL_FILENAME)
    print("train loss: %d" % model.get_latest_training_loss())

    return model


def load_google_w2v_model(words_limit=GOOGLE_NEWS_WORD_LIMIT):
    logging.debug("Lading google word2vec ...")
    google_model = models.KeyedVectors.load_word2vec_format(
        full_path("data/google/GoogleNews-vectors-negative300.bin"),
        binary=True,
        limit=words_limit
    )
    logging.debug("Finished google word2vec loading")
    return google_model
