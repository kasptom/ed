from typing import List

import logging
import numpy as np
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import Word2Vec

from src.configuration import TEST_DATA_PERCENTAGE, USE_GOOGLE_W2V, CORPUS_FILES, get_tfidf_file_name, \
    get_dictionary_file_name
from src.preprocessing.w2v_loader import create_w2v_from_corpus, load_google_w2v_model
from src.preprocessing.create_corpus import create_corpus_and_labels
from src.utils.get_file import create_file_and_folders_if_not_exist

BATCH_SIZE = 100


def corpus_to_vectors():
    corpus, labels = create_corpus_and_labels()

    model = create_w2v_from_corpus(corpus=corpus)
    google_model = load_google_w2v_model() if USE_GOOGLE_W2V else None

    dictionary = _dictionary(corpus)
    tfidf = _tfidf(corpus, dictionary)

    document_vectors = documents_to_vector_from_w2v(corpus, google_model, model, tfidf)

    train_samples_count = round(BATCH_SIZE * (100 - TEST_DATA_PERCENTAGE) / 100, 0)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    doc_iter = iter(document_vectors)
    label_iter = iter(labels)
    counter = 0
    while True:
        try:
            document_vector = next(doc_iter)
        except StopIteration:
            break
        if counter < train_samples_count:
            x_train.append([document_vector])
        else:
            x_test.append([document_vector])
        counter = (counter + 1) % BATCH_SIZE

    counter = 0
    while True:
        try:
            label = next(label_iter)
        except StopIteration:
            break
        if counter < train_samples_count:
            y_train.append(label)
        else:
            y_test.append(label)
        counter = (counter + 1) % BATCH_SIZE

    x_train = np.concatenate(tuple(x_train), axis=0)
    y_train = np.array(y_train)

    x_test = np.concatenate(tuple(x_test), axis=0)
    y_test = np.array(y_test)

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
        logging.info("documents to vectors - using google model")
    vectorized_documents = []
    for document in corpus:
        all_documents_words_count += len(document)
        document_vector, non_zero_vectors = _document_to_vector(
            document=document,
            model=model,
            google_model=google_model,
            tfidf=tfidf)
        non_zero_vectos_all_documents += non_zero_vectors
        vectorized_documents.append(document_vector)

    print("percentage of appearances of words existing in used model: %.2f" %
          ((non_zero_vectos_all_documents / all_documents_words_count) * 100))
    return vectorized_documents


def _tfidf(corpus, dictionary):
    tfidf_file_name = get_tfidf_file_name(CORPUS_FILES["label"])
    try:
        tfidf = TfidfModel.load(tfidf_file_name)
    except FileNotFoundError:
        corpus_numeric = [dictionary.doc2bow(document) for document in corpus]
        tfidf = TfidfModel(corpus=corpus_numeric)
        print("File does not exist - creating the tfidf model")

        create_file_and_folders_if_not_exist(tfidf_file_name)
        tfidf.save(tfidf_file_name)

    return tfidf


def _dictionary(corpus):
    dictionary_file_name = get_dictionary_file_name(CORPUS_FILES["label"])
    dictionary = corpora.Dictionary(corpus)
    try:
        dictionary = corpora.Dictionary.load(dictionary_file_name)
    except FileNotFoundError:
        print("File does not exist - creating the tfidf model")
        create_file_and_folders_if_not_exist(dictionary_file_name)
        dictionary.save(dictionary_file_name)
    return dictionary


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
        word_vectors.append(google_model.wv.word_vec(word) * tfidf.idfs[model.wv.vocab[word].index])
    else:
        word_vectors.append(np.random.rand(google_model.vector_size))
    return word_in_google_model


def create_with_self_trained_model(model, tfidf, word, word_vectors):
    word_in_model = word in model
    if word_in_model:
        word_vectors.append(model.wv.word_vec(word) * tfidf.idfs[model.wv.vocab[word].index])
    else:
        word_vectors.append(np.zeros(model.vector_size))
    return word_in_model
