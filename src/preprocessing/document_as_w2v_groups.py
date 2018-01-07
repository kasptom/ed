import numpy as np
from gensim.models import Word2Vec

from src.configuration import USE_GOOGLE_W2V, DATA_SET, WORD_NUMERIC_VECTOR_SIZE
from src.preprocessing.create_corpus import create_corpus_and_labels
from src.preprocessing.w2v_loader import create_w2v_from_corpus, load_google_w2v_model


def get_train_and_test_vectors():
    documents_vectors_batches, labels_batches = corpus_to_vector_batches()
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    batch_size = DATA_SET["batch_size"]
    sample_size = 100
    counter = 0

    prev_idx = 0
    for idx in range(batch_size, len(documents_vectors_batches) + batch_size, batch_size):
        if counter < sample_size:
            x_train += documents_vectors_batches[prev_idx:idx]
            y_train += (labels_batches[prev_idx:idx])
        else:
            x_test += (documents_vectors_batches[prev_idx:idx])
            y_test += (labels_batches[prev_idx:idx])
        prev_idx = idx
        counter = (counter + 1) % batch_size

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    #
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def corpus_to_vector_batches():
    corpus, labels = create_corpus_and_labels()

    # model = create_w2v_from_corpus(corpus=corpus)
    google_model = load_google_w2v_model() if USE_GOOGLE_W2V else None

    # each batch conforms to one document in the corpus
    corpus_vector_batches = []
    labels_batches = []
    batch_size = DATA_SET["batch_size"]

    for (document, label) in zip(corpus, labels):
        document_batch, label_batch = document_to_batch(document, label, google_model, batch_size)
        corpus_vector_batches += document_batch
        labels_batches += label_batch

    return corpus_vector_batches, labels_batches


def document_to_batch(document, label, model: Word2Vec, batch_size):
    words_vectors_batch = []
    label_batch = []

    counter = 0
    for word in document:
        if word in model:
            words_vectors_batch.append(model.wv[word])
            label_batch.append(np.array(label))
            counter += 1
        if counter >= batch_size:
            break
    for _ in range(counter, batch_size):
        words_vectors_batch.append(np.zeros(WORD_NUMERIC_VECTOR_SIZE))
        label_batch.append(np.array(label))

    return words_vectors_batch, label_batch


get_train_and_test_vectors()
