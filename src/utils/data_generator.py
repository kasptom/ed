from os import listdir

import numpy as np

from src.configuration import DATA_SET, get_vector_labels_file_name, TEST_DATA_PERCENTAGE, get_vector_words_directory

prev_val_counter = 0
files_counter = 0


def get_train_samples_count():
    _data_path = get_vector_words_directory(DATA_SET['label'])
    _files = [f for f in listdir(_data_path)]
    return len(_files) * ((100 - TEST_DATA_PERCENTAGE) / 100)


def get_test_samples_count():
    _data_path = get_vector_words_directory(DATA_SET['label'])
    _files = [f for f in listdir(_data_path)]
    return len(_files) * TEST_DATA_PERCENTAGE / 100


def get_train_generator(batch_size=DATA_SET['batch_size']):
    data_path = get_vector_words_directory(DATA_SET['label'])
    files = [f for f in listdir(data_path)]
    labels = np.load(get_vector_labels_file_name(DATA_SET['label']))

    global prev_val_counter
    global files_counter

    while files_counter < len(files) * ((100 - TEST_DATA_PERCENTAGE) / 100):
        x = []
        y = []
        files_counter += batch_size
        for idx in range(prev_val_counter, files_counter):
            x.append(np.load(data_path + "/" + files[idx]))
            y.append(np.array(labels[idx]))
        prev_val_counter = files_counter
        yield (np.array(x), np.array(y))


def get_test_generator(batch_size=DATA_SET['batch_size']):
    data_path = get_vector_words_directory(DATA_SET['label'])
    files = [f for f in listdir(data_path)]
    labels = np.load(get_vector_labels_file_name(DATA_SET['label']))

    global prev_val_counter
    global files_counter

    while files_counter < len(files):
        x = []
        y = []
        files_counter += batch_size
        for idx in range(prev_val_counter, files_counter):
            x.append(np.load(data_path + "/" + files[idx]))
            y.append(np.array(labels[idx]))
        prev_val_counter = files_counter
        yield (np.array(x), np.array(y))
