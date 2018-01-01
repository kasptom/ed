WORD_NUMERIC_VECTOR_SIZE = 300
EPOCHS_NUMBER = 3
DROPOUT = 0.2
RECURRENT_DROPOUT = 0.2
BATCH_SIZE = 32
TEST_DATA_PERCENTAGE = 30

CORPUS_FILES_IMDB = {
    "label": "imdb",
    "positive": "data/imdb.pos",
    "negative": "data/imdb.neg"
}

CORPUS_FILES_RT_POLARITY = {
    "label": "rt-polarity",
    "positive": "data/rt-polaritydata/rt-polarity.pos",
    "negative": "data/rt-polaritydata/rt-polarity.neg"
}

CORPUS_FILES = CORPUS_FILES_IMDB


def print_configuration():
    print("----------------Configuration----------------")
    print(
        "corpus: %s, "
        "vector size: %d, "
        "epochs number: %d, "
        "dropout: %.2f, "
        "recurrent dropout: %.2f, "
        "batch size: %d, "
        "test data percentage: %.2f" %
        (CORPUS_FILES["label"], WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, BATCH_SIZE,
         TEST_DATA_PERCENTAGE))
    print("---------------------------------------------")
