from typing import List

import logging
import numpy as np
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import Word2Vec

from src.preprocessing.configuration import TEST_DATA_PERCENTAGE, USE_GOOGLE_W2V
from src.preprocessing.corpus_to_model import corpus_to_model, load_google_w2v_model
from src.preprocessing.create_corpus import create_corpus


def corpus_to_vectors():
    corpus, neg_number, pos_number = create_corpus()

    model = corpus_to_model(corpus=corpus)
    google_model = load_google_w2v_model() if USE_GOOGLE_W2V else None

    tfidf, dictionary = _tfidf(corpus)

    document_vectors = documents_to_vector_from_w2v(corpus, google_model, model, tfidf)

    x_vec = np.concatenate(tuple(document_vectors), axis=0)
    y_vec = np.concatenate((np.zeros(neg_number), np.ones(pos_number)), axis=0)

    train_samples_count = int(round(pos_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
    x_train_pos = [x_vec[i] for i in range(train_samples_count)]
    y_train_pos = [y_vec[i] for i in range(train_samples_count)]
    x_test_pos = [x_vec[i] for i in range(train_samples_count, pos_number)]
    y_test_pos = [y_vec[i] for i in range(train_samples_count, pos_number)]

    train_samples_count = int(round(neg_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
    x_train_neg = [x_vec[i] for i in range(pos_number, pos_number + train_samples_count)]
    y_train_neg = [y_vec[i] for i in range(pos_number, pos_number + train_samples_count)]
    x_test_neg = [x_vec[i] for i in range(pos_number + train_samples_count, pos_number + neg_number)]
    y_test_neg = [y_vec[i] for i in range(pos_number + train_samples_count, pos_number + neg_number)]

    x_train = np.array(x_train_pos + x_train_neg)
    y_train = np.array(y_train_pos + y_train_neg)

    x_test = np.array(x_test_pos + x_test_neg)
    y_test = np.array(y_test_pos + y_test_neg)

    result = (x_train, y_train), (x_test, y_test)

    # normalize (get rid of the negative values)
    global_minimum = abs(x_train.min()) if abs(x_train.min()) > abs(x_test.min()) else abs(x_test.min())
    x_train += global_minimum
    x_test += global_minimum
    return result


def documents_to_vector_from_w2v(corpus, google_model, model, tfidf):
    non_zero_vectos_all_documents = 0
    all_documents_words_count = 0

    if google_model:
        logging.debug("documents to vectors - using google model")
    vectorized_documents = []
    for document in corpus:
        all_documents_words_count += len(document)
        document_vector, non_zero_vectors = _document_to_vector(
            document=document,
            model=model,
            google_model=google_model,
            tfidf=tfidf)
        non_zero_vectos_all_documents += non_zero_vectors
        vectorized_documents.append([document_vector])

    print("percentage of appearances of words existing in used model: %.2f" %
          ((non_zero_vectos_all_documents / all_documents_words_count) * 100))
    return vectorized_documents


def _tfidf(corpus):
    dictionary = corpora.Dictionary(corpus)
    corpus_numeric = [dictionary.doc2bow(document) for document in corpus]
    tfidf = TfidfModel(corpus=corpus_numeric)
    return tfidf, dictionary


def _document_to_vector(document: List[str], model: Word2Vec, google_model: Word2Vec, tfidf):
    word_vectors = []
    counter_words_in_dictionary = 0
    for word in document:
        if google_model:
            if create_with_google_model(google_model, model, tfidf, word, word_vectors):
                counter_words_in_dictionary += 1
        else:
            if create_with_self_trained_model(model, tfidf, word, word_vectors):
                counter_words_in_dictionary += 1
    return np.mean(word_vectors, 0), counter_words_in_dictionary


def create_with_google_model(google_model, model, tfidf, word, word_vectors):
    word_in_google_model = word in google_model
    word_in_model = word in model
    if word_in_model and word_in_google_model:
        word_vectors.append(google_model.wv.word_vec(word))
        # * tfidf.idfs[model.wv.vocab[word].index])
    else:
        word_vectors.append(np.zeros(google_model.vector_size))
    return word_in_google_model


def create_with_self_trained_model(model, tfidf, word, word_vectors):
    word_in_model = word in model
    if word_in_model:
        word_vectors.append(model.wv.word_vec(word) * tfidf.idfs[model.wv.vocab[word].index])
    else:
        word_vectors.append(np.zeros(model.vector_size))
    return word_in_model
