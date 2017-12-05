from typing import List

import numpy as np
from gensim.models import Word2Vec

STOP_LIST = set('for a of the and to in'.split())
TEST_DATA_PERCENTAGE = 30
_WORD2VEC_MODEL_FILENAME = "../data/w2v_model"


def corpus_to_vectors():
    negative_corpus = _load_corpus(filename="../data/rt-polaritydata/rt-polarity.neg")
    positive_corpus = _load_corpus(filename="../data/rt-polaritydata/rt-polarity.pos")
    negatives_number = len(negative_corpus)
    positives_number = len(positive_corpus)

    corpus = negative_corpus
    corpus.append(positive_corpus)

    try:
        model = Word2Vec.load(_WORD2VEC_MODEL_FILENAME)
    except FileNotFoundError:
        print("File does not exist - creating the model")
        model = Word2Vec(corpus, size=100)
        model.save(_WORD2VEC_MODEL_FILENAME)
    print(model.similarity('old', 'grandfather'))
    print(model.similarity('young', 'grandfather'))

    X = np.array([_document_to_vector(document=document, model=model) for document in corpus])
    Y = np.concatenate((np.zeros(negatives_number), np.ones(positives_number)), axis=0)

    train_samples_count = int(round(positives_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
    x_train_pos = [X[i] for i in range(train_samples_count)]
    y_train_pos = [1 for _ in range(train_samples_count)]
    x_test_pos = [X[i] for i in range(train_samples_count, positives_number)]
    y_test_pos = [1 for _ in range(train_samples_count, positives_number)]

    train_samples_count = int(round(negatives_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
    x_train_neg = [X[i] for i in range(positives_number, positives_number + train_samples_count)]
    y_train_neg = [0 for _ in range(train_samples_count)]
    x_test_neg = [X[i] for i in range(positives_number + train_samples_count, positives_number + negatives_number)]
    y_test_neg = [0 for _ in range(train_samples_count, negatives_number)]

    x_train = x_train_pos + x_train_neg
    y_train = y_train_pos + y_train_neg

    x_test = x_test_pos + x_test_neg
    y_test = y_test_pos + y_test_neg

    result = (x_train, y_train), (x_test, y_test)
    return result


def _load_corpus(filename: str):
    documents = []
    with open(filename, encoding='utf-8', errors='ignore') as data_file:
        iter_file = iter(data_file)
        for line in iter_file:
            documents.append(line)
    # remove common words and tokenize
    return [[word for word in document.lower().split() if word not in STOP_LIST]
            for document in documents]


def _document_to_vector(document: List[str], model: Word2Vec):
    return np.array([
        np.mean(model[word] if word in model.wv.vocab else np.zeros(model.vector_size) for word in document)
    ])


# corpus_to_vectors()
