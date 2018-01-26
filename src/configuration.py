from src.utils.get_file import full_path

WORD_NUMERIC_VECTOR_SIZE = 300
TEST_DATA_PERCENTAGE = 30

# GOOGLE_NEWS_WORD_LIMIT = 500000
GOOGLE_NEWS_WORD_LIMIT = None

DATA_SET_IMDB = {
    "label": "imdb",
    "positive": "data/imdb.pos",
    "negative": "data/imdb.neg",
    "time_steps": 350,
    "batch_size": 64,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "epochs": 14,
    "use_google_w2v": True
}

DATA_SET_RT_POLARITY = {
    "label": "rt-polarity",
    "positive": "data/rt-polaritydata/rt-polarity.pos",
    "negative": "data/rt-polaritydata/rt-polarity.neg",
    "time_steps": 35,
    "batch_size": 20,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "epochs": 5,
    "use_google_w2v": True
}

DATA_SET = DATA_SET_RT_POLARITY
TIME_STEPS = DATA_SET["time_steps"]
BATCH_SIZE = DATA_SET['batch_size']
DROPOUT = DATA_SET["dropout"]
RECURRENT_DROPOUT = DATA_SET["recurrent_dropout"]
USE_GOOGLE_W2V = DATA_SET["use_google_w2v"]
EPOCHS_NUMBER = DATA_SET["epochs"]


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
        (DATA_SET["label"], WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, TIME_STEPS,
         TEST_DATA_PERCENTAGE, USE_GOOGLE_W2V))
    print("---------------------------------------------")


def get_network_model_snapshot(corpus_label: str):
    return full_path("data/lstm-net/" + corpus_label + ".h5")


def get_tfidf_file_name(corpus_label: str):
    return full_path("data/tfidfs/" + corpus_label + "_tfidf_model")


def get_dictionary_file_name(corpus_label: str):
    return full_path("data/dicts/" + corpus_label + "_dict")


def get_w2v_file_name(corpus_label: str):
    return full_path("data/word2vecs/" + corpus_label + "_model")


def get_vector_words_directory(corpus_label: str):
    return full_path("data/vector_words/" + corpus_label + "_words_timestep" + str(DATA_SET['batch_size']))


def get_batch_file_name(document_idx):
    document_idx_str = "%010d" % document_idx
    return get_vector_words_directory(DATA_SET['label']) + "/" + document_idx_str + ".npy"


def get_vector_labels_file_name(corpus_label: str):
    return full_path("data/vector_words/" + corpus_label + "_labels.npy")


def get_csv_log_file_name(corpus_label: str):
    return full_path(
        "log/" + corpus_label + "_timestep" + str(DATA_SET['batch_size']) + "_drout_" + str(
            DATA_SET['dropout']) + "_rdrout_" +
        str(DATA_SET["recurrent_dropout"]) + ".csv")
