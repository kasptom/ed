import logging

import numpy as np
from gensim.models import Word2Vec

from src.configuration import USE_GOOGLE_W2V, DATA_SET, WORD_NUMERIC_VECTOR_SIZE, TEST_DATA_PERCENTAGE, \
    get_vector_labels_file_name, get_batch_file_name
from src.preprocessing.create_corpus import create_corpus_and_labels
from src.preprocessing.w2v_loader import load_google_w2v_model, create_w2v_from_corpus
from src.utils.get_file import create_file_and_folders_if_not_exist


def ensure_word_numeric_representation_created():
    labels_file_name = get_vector_labels_file_name(DATA_SET['label'])

    try:
        np.load(labels_file_name)
    except IOError:
        logging.info("word vector files does not exist - creating...")
        corpus, labels = create_corpus_and_labels()

        w2v_model = load_google_w2v_model() if USE_GOOGLE_W2V else create_w2v_from_corpus(corpus)

        # each time_step portion conforms to one document in the corpus
        time_steps = DATA_SET["time_steps"]

        for document_idx in range(len(corpus)):
            document_batch = document_to_batch(corpus[document_idx], w2v_model, time_steps)
            batch_file_name = get_batch_file_name(document_idx)
            create_file_and_folders_if_not_exist(batch_file_name)
            np.save(batch_file_name, document_batch)

        create_file_and_folders_if_not_exist(labels_file_name)

        np.save(labels_file_name, labels)
        logging.info("word vector files created")


def document_to_batch(document, model: Word2Vec, time_steps):
    """
    Converts the document to its numeric representation
    :param document:
    :param model:
    :param time_steps: maximum number of words that will be taken into account during vector computation
    :return:
    """
    words_vectors_batch = []

    counter = 0
    for word in document:
        if word in model:
            words_vectors_batch.append(model.wv[word])
            counter += 1
        if counter >= time_steps:
            break
    for _ in range(counter, time_steps):
        words_vectors_batch.append(np.zeros(WORD_NUMERIC_VECTOR_SIZE))

    return np.array(words_vectors_batch)
