from gensim import models
from gensim.models import Word2Vec

from src.preprocessing.configuration import WORD_NUMERIC_VECTOR_SIZE
from src.utils.get_file import full_path

# _WORD2VEC_MODEL_FILENAME = full_path("data/w2v_model")
_WORD2VEC_MODEL_FILENAME = full_path("data/w2v_imdb_model")


def corpus_to_model(corpus):
    google_model = models.KeyedVectors.load_word2vec_format(full_path("data/google/GoogleNews-vectors-negative300.bin"), binary=True)
    try:
        model = Word2Vec.load(_WORD2VEC_MODEL_FILENAME)
    except FileNotFoundError:
        print("File does not exist - creating the model")
        model = Word2Vec(corpus, size=WORD_NUMERIC_VECTOR_SIZE)
        model.train(corpus, total_examples=len(corpus), epochs=10)
        model.save(_WORD2VEC_MODEL_FILENAME)
    print("train loss: %d" % model.get_latest_training_loss())

    return model, google_model
