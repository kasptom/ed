import logging
from gensim import models
from gensim.models import Word2Vec

from src.configuration import WORD_NUMERIC_VECTOR_SIZE, DATA_SET, GOOGLE_NEWS_WORD_LIMIT, get_w2v_file_name
from src.utils.get_file import full_path, create_file_and_folders_if_not_exist


def create_w2v_from_corpus(corpus):
    word2vec_model_file_name = get_w2v_file_name(DATA_SET["label"])
    try:
        model = Word2Vec.load(word2vec_model_file_name)
    except FileNotFoundError:
        print("File does not exist - creating the w2v model")
        model = Word2Vec(corpus, size=WORD_NUMERIC_VECTOR_SIZE, min_count=1)
        model.train(corpus, total_examples=len(corpus), epochs=10)

        create_file_and_folders_if_not_exist(word2vec_model_file_name)

        model.save(word2vec_model_file_name)

    return model


_google_model = None


def load_google_w2v_model(words_limit=GOOGLE_NEWS_WORD_LIMIT):
    global _google_model
    if _google_model is not None:
        return _google_model

    logging.info("Lading google word2vec ...")
    google_model = models.KeyedVectors.load_word2vec_format(
        full_path("data/google/GoogleNews-vectors-negative300.bin"),
        binary=True,
        limit=words_limit
    )
    logging.info("Finished google word2vec loading")
    _google_model = google_model
    return google_model
