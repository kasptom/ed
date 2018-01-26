# coding=utf-8
from __future__ import print_function

import time

import numpy as np
from keras import callbacks
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

from src import configuration
from src.configuration import WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, \
    TIME_STEPS, get_network_model_snapshot, DATA_SET, get_csv_log_file_name, BATCH_SIZE, EPOCH_PATIENCE
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

np.random.seed(7)

print('Loading data...')
start = time.time()

ensure_word_numeric_representation_created()

data_generator = DataGenerator()

end = time.time()
print("time elapsed: ", end - start, " seconds")

print('Build model...')
start = time.time()

create_file_and_folders_if_not_exist(get_csv_log_file_name(DATA_SET['label']))

callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_PATIENCE),
             callbacks.CSVLogger(get_csv_log_file_name(DATA_SET['label']))]

model = Sequential()
model.add(
    LSTM(200, input_shape=(TIME_STEPS, WORD_NUMERIC_VECTOR_SIZE), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

configuration.print_configuration()

end = time.time()
print("time elapsed: ", end - start, " seconds")

print('Train...')
start = time.time()
model.fit_generator(data_generator.get_train_generator(),
                    epochs=EPOCHS_NUMBER,
                    validation_data=data_generator.get_test_generator(),
                    samples_per_epoch=data_generator.get_train_samples_count() // BATCH_SIZE,
                    callbacks=callbacks,
                    validation_steps=data_generator.get_test_samples_count() // BATCH_SIZE)

score, acc = model.evaluate_generator(data_generator.get_test_generator(),
                                      steps=data_generator.get_test_samples_count() // BATCH_SIZE)

create_file_and_folders_if_not_exist(get_network_model_snapshot(DATA_SET['label']))
model.save(get_network_model_snapshot(DATA_SET['label']))

print('Score: %f' % score)
print('Test accuracy: %f%%' % (acc * 100))
end = time.time()
print("time elapsed: ", end - start, " seconds")

log_summary(score, (acc * 100), end - start)
