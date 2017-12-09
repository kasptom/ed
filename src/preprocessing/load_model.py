from gensim.models import Word2Vec

from src.utils.get_file import full_path

_WORD2VEC_MODEL_FILENAME = full_path("data/w2v_model")


def corpus_to_model(corpus):
    try:
        model = Word2Vec.load(_WORD2VEC_MODEL_FILENAME)
    except FileNotFoundError:
        print("File does not exist - creating the model")
        model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=4)
        model.save(_WORD2VEC_MODEL_FILENAME)
    print(model.wv.similarity('old', 'yes'))
    print(model.wv.similarity('young', 'yes'))
    print(model.wv.similarity('man', 'woman'))

    return corpus
