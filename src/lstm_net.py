# coding=utf-8
from __future__ import print_function

import time

import numpy as np
from keras import callbacks

from src import configuration
from src.configuration import EPOCHS_NUMBER, get_network_model_snapshot, DATA_SET, get_csv_log_file_name, BATCH_SIZE, \
    EPOCH_PATIENCE
from src.preprocessing.document_as_w2v_groups import ensure_word_numeric_representation_created
from src.utils.data_generator import DataGenerator
from src.utils.get_file import create_file_and_folders_if_not_exist
from src.utils.log_summary import log_summary

"""
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
"""


class LstmNet:
    def __init__(self):
        self.callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_PATIENCE),
                          callbacks.CSVLogger(get_csv_log_file_name(DATA_SET['label']))]
        self.model = self.build_model()
        self.data_generator = self.create_data_generator()

    def build_model(self):
        raise NotImplementedError()

    def create_data_generator(self):
        raise NotImplementedError()

    def train_network(self):
        np.random.seed(7)

        print('Loading data...')
        start = time.time()

        ensure_word_numeric_representation_created()

        end = time.time()
        print("time elapsed: ", end - start, " seconds")

        print('Build model...')
        start = time.time()

        create_file_and_folders_if_not_exist(get_csv_log_file_name(DATA_SET['label']))

        print(self.model.summary())

        configuration.print_configuration()

        end = time.time()
        print("time elapsed: ", end - start, " seconds")

        print('Train...')
        start = time.time()
        self.model.fit_generator(self.data_generator.get_train_generator(),
                                 epochs=EPOCHS_NUMBER,
                                 validation_data=self.data_generator.get_test_generator(),
                                 steps_per_epoch=self.data_generator.get_train_samples_count() // BATCH_SIZE,
                                 callbacks=self.callbacks,
                                 validation_steps=self.data_generator.get_test_samples_count() // BATCH_SIZE,
                                 workers=8)

        score, acc = self.model.evaluate_generator(self.data_generator.get_test_generator(),
                                                   steps=self.data_generator.get_test_samples_count() // BATCH_SIZE)

        create_file_and_folders_if_not_exist(get_network_model_snapshot(DATA_SET['label']))
        self.model.save(get_network_model_snapshot(DATA_SET['label']))

        print('Score: %f' % score)
        print('Test accuracy: %f%%' % (acc * 100))
        end = time.time()
        print("time elapsed: ", end - start, " seconds")

        log_summary(score, (acc * 100), end - start)
