from src.utils.get_file import full_path

WORD_NUMERIC_VECTOR_SIZE = 300
TEST_DATA_PERCENTAGE = 30
EPOCH_PATIENCE = 3

# GOOGLE_NEWS_WORD_LIMIT = 500000
GOOGLE_NEWS_WORD_LIMIT = None

DATA_SET_IMDB = {
    "label": "imdb",
    "positive": "data/imdb.pos",
    "negative": "data/imdb.neg",
    "time_steps": 150,
    "max_time_steps": 350,
    "batch_size": 64,
    "dropout": 0.4,
    "recurrent_dropout": 0.4,
    "epochs": 40,
    "use_google_w2v": True
}

DATA_SET_RT_POLARITY = {
    "label": "rt-polarity",
    "positive": "data/rt-polaritydata/rt-polarity.pos",
    "negative": "data/rt-polaritydata/rt-polarity.neg",
    "time_steps": 35,
    "max_time_steps": 35,
    "batch_size": 20,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "epochs": 40,
    "use_google_w2v": True
}

DATA_SET_TREC = {
    "label": "trec",
    "corpus_file": "data/trec/trec.corp",
    "time_steps": 10,
    "max_time_steps": 35,
    "batch_size": 20,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "epochs": 40,
    "use_google_w2v": True
}

DATA_SET = DATA_SET_IMDB
TIME_STEPS = DATA_SET["time_steps"]
BATCH_SIZE = DATA_SET['batch_size']
DROPOUT = DATA_SET["dropout"]
RECURRENT_DROPOUT = DATA_SET["recurrent_dropout"]
USE_GOOGLE_W2V = DATA_SET["use_google_w2v"]
EPOCHS_NUMBER = DATA_SET["epochs"]


def print_configuration():
    print("----------------Configuration----------------")
    print(configuration_string())
    print("---------------------------------------------")


def configuration_string():
    return ("corpus: %s\\n "
            "vector size: %d\n "
            "epochs limit: %d\n "
            "dropout: %.2f\n "
            "recurrent dropout: %.2f\n "
            "time steps: %d\n "
            "batch size: %d\n "
            "test data percentage: %.2f\n "
            "use google w2v: %s\n"
            "stop after %d epoch(s) if no progress" %
            (DATA_SET["label"], WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, TIME_STEPS,
             BATCH_SIZE,
             TEST_DATA_PERCENTAGE, USE_GOOGLE_W2V, EPOCH_PATIENCE))


def get_network_model_snapshot(corpus_label: str):
    return full_path("lstm-net/{}_timestep{}_drout{}_rdrout{}_batch{}.h5".format(
        corpus_label,
        str(DATA_SET['time_steps']),
        str(DATA_SET['dropout']),
        str(DATA_SET["recurrent_dropout"]),
        str(DATA_SET['batch_size'])
    ))


def get_tfidf_file_name(corpus_label: str):
    return full_path("data/tfidfs/" + corpus_label + "_tfidf_model")


def get_dictionary_file_name(corpus_label: str):
    return full_path("data/dicts/" + corpus_label + "_dict")


def get_w2v_file_name(corpus_label: str):
    return full_path("data/word2vecs/" + corpus_label + "_model")


def get_vector_words_directory(corpus_label: str):
    return full_path("data/vector_words/" + corpus_label + "_words_max_timestep" + str(DATA_SET['max_time_steps']))


def get_vector_words_directory_for_dataset(corpus_label: str, dataset):
    return full_path("data/vector_words/" + corpus_label + "_words_max_timestep" + str(dataset['max_time_steps']))


def get_batch_file_name(document_idx):
    """
    Name of the file with the vectorised version of the document (stored as a group of
    MAX_TIME_STEPS x WORD_NUMERIC_VECTOR_SIZE) numpy vectors
    :param document_idx:
    :return:
    """
    document_idx_str = "%010d" % document_idx
    return get_vector_words_directory(DATA_SET['label']) + "/" + document_idx_str + ".npy"


def get_batch_file_name_for_dataset(document_idx, data_set):
    """
    Name of the file with the vectorised version of the document (stored as a group of
    MAX_TIME_STEPS x WORD_NUMERIC_VECTOR_SIZE) numpy vectors
    :param document_idx:
    :param data_set:
    :return:
    """
    document_idx_str = "%010d" % document_idx
    return get_vector_words_directory_for_dataset(data_set['label'], data_set) + "/" + document_idx_str + ".npy"


def get_vector_labels_file_name(corpus_label: str):
    return full_path("data/vector_words/" + corpus_label + "_labels.npy")


def get_csv_log_file_name(corpus_label: str):
    return full_path(
        "log/{}_timestep{}_drout{}_rdrout{}_batch{}.csv".format(
            corpus_label,
            str(DATA_SET['time_steps']),
            str(DATA_SET['dropout']),
            str(DATA_SET["recurrent_dropout"]),
            str(DATA_SET["batch_size"]))
    )


def get_summaries_file_name(corpus_label: str):
    return full_path("log/{}_summary.log".format(corpus_label))
