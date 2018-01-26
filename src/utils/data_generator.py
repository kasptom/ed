from os import listdir

import numpy as np

from src.configuration import DATA_SET, get_vector_labels_file_name, TEST_DATA_PERCENTAGE, get_vector_words_directory

prev_val_counter = 0
files_counter = 0


class DataGenerator:
    def __init__(self):
        self.data_path = get_vector_words_directory(DATA_SET['label'])
        self.files = [f for f in listdir(self.data_path)]
        self.labels = np.load(get_vector_labels_file_name(DATA_SET['label']))

        self.train_samples_count = int(round(len(self.files) * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
        self.test_samples_count = len(self.files) - self.train_samples_count

        self.train_files = self.files[:self.train_samples_count]
        self.train_labels = self.labels[:self.train_samples_count]

        self.test_files = self.files[self.train_samples_count:]
        self.test_labels = self.labels[self.train_samples_count:]

    def get_train_samples_count(self):
        return self.train_samples_count

    def get_test_samples_count(self):
        return self.test_samples_count

    def get_train_generator(self, batch_size=DATA_SET['batch_size']):
        idx = 0
        while True:
            x = []
            y = []
            counter = 0
            while counter < batch_size:
                x.append(np.load(self.data_path + "/" + self.train_files[idx]))
                y.append(np.array(self.train_labels[idx]))
                idx = (idx + 1) % self.train_samples_count
                counter += 1
            yield (np.array(x), np.array(y))

    def get_test_generator(self, batch_size=DATA_SET['batch_size']):
        idx = 0
        while True:
            x = []
            y = []
            counter = 0
            while counter < batch_size:
                x.append(np.load(self.data_path + "/" + self.test_files[idx]))
                y.append(np.array(self.test_labels[idx]))
                idx = (idx + 1) % self.test_samples_count
                counter += 1
            yield (np.array(x), np.array(y))
