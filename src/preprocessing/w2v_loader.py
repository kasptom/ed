import logging
from gensim import models
from gensim.models import Word2Vec

from src.configuration import WORD_NUMERIC_VECTOR_SIZE, CORPUS_FILES, GOOGLE_NEWS_WORD_LIMIT, get_w2v_file_name
from src.utils.get_file import full_path, create_file_and_folders_if_not_exist


def create_w2v_from_corpus(corpus=None):
    word2vec_model_file_name = get_w2v_file_name(CORPUS_FILES["label"])
    try:
        model = Word2Vec.load(word2vec_model_file_name)
    except FileNotFoundError:
        print("File does not exist - creating the w2v model")
        model = Word2Vec(corpus, size=WORD_NUMERIC_VECTOR_SIZE)
        model.train(corpus, total_examples=len(corpus), epochs=10)

        create_file_and_folders_if_not_exist(word2vec_model_file_name)

        model.save(word2vec_model_file_name)
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
