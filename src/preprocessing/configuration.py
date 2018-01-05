WORD_NUMERIC_VECTOR_SIZE = 300
TEST_DATA_PERCENTAGE = 30

GOOGLE_NEWS_WORD_LIMIT = 500000
USE_GOOGLE_W2V = False

CORPUS_FILES_IMDB = {
    "label": "imdb",
    "positive": "data/imdb.pos",
    "negative": "data/imdb.neg",
    "batch_size": 64,
    "dropout": 0.5,
    "recurrent_dropout": 0.3,
    "epochs": 3
}

CORPUS_FILES_RT_POLARITY = {
    "label": "rt-polarity",
    "positive": "data/rt-polaritydata/rt-polarity.pos",
    "negative": "data/rt-polaritydata/rt-polarity.neg",
    "batch_size": 32,
    "dropout": 0.5,
    "recurrent_dropout": 0.5,
    "epochs": 3
}

CORPUS_FILES = CORPUS_FILES_IMDB
BATCH_SIZE = CORPUS_FILES["batch_size"]
DROPOUT = CORPUS_FILES["dropout"]
RECURRENT_DROPOUT = CORPUS_FILES["recurrent_dropout"]
EPOCHS_NUMBER = 15


def print_configuration():
    print("----------------Configuration----------------")
    print(
        "corpus: %s, "
        "vector size: %d, "
        "epochs number: %d, "
        "dropout: %.2f, "
        "recurrent dropout: %.2f, "
        "batch size: %d, "
        "test data percentage: %.2f "
        "use google w2v: %s " %
        (CORPUS_FILES["label"], WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, BATCH_SIZE,
         TEST_DATA_PERCENTAGE, USE_GOOGLE_W2V))
    print("---------------------------------------------")
