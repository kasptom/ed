from typing import List

import numpy as np
from gensim.models import Word2Vec

from src.preprocessing.create_corpus import create_corpus
from src.preprocessing.load_model import corpus_to_model

TEST_DATA_PERCENTAGE = 30


def corpus_to_vectors():
    corpus, neg_number, pos_number = create_corpus()

    model = corpus_to_model(corpus=corpus)

    x_vec = np.array([_document_to_vector(document=document, model=model) for document in corpus])
    y_vec = np.concatenate((np.zeros(neg_number), np.ones(pos_number)), axis=0)

    train_samples_count = int(round(pos_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
    x_train_pos = [x_vec[i] for i in range(train_samples_count)]
    y_train_pos = [1 for _ in range(train_samples_count)]
    x_test_pos = [x_vec[i] for i in range(train_samples_count, pos_number)]
    y_test_pos = [1 for _ in range(train_samples_count, pos_number)]

    train_samples_count = int(round(neg_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
    x_train_neg = [x_vec[i] for i in range(pos_number, pos_number + train_samples_count)]
    y_train_neg = [0 for _ in range(train_samples_count)]
    x_test_neg = [x_vec[i] for i in range(pos_number + train_samples_count, pos_number + neg_number)]
    y_test_neg = [0 for _ in range(train_samples_count, neg_number)]

    x_train = x_train_pos + x_train_neg
    y_train = y_train_pos + y_train_neg

    x_test = x_test_pos + x_test_neg
    y_test = y_test_pos + y_test_neg

    result = (x_train, y_train), (x_test, y_test)
    return result


def _document_to_vector(document: List[str], model: Word2Vec):
    word_vectors = []
    for word in document:
        if word in model.wv.vocab:
            pass
            # word_vectors.append(np.) TODO
    np.mean(model[word] if word in model.wv.vocab else np.zeros(model.vector_size) for word in document)
    return np.array(word_vectors)


corpus_to_vectors()
