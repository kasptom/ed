# coding=utf-8
from __future__ import print_function

import time

import numpy as np
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from src import configuration
from src.configuration import WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, \
    TIME_STEP
from src.preprocessing.document_as_w2v_groups import get_train_and_test_vectors

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

(x_train, y_train), (x_test, y_test) = get_train_and_test_vectors()

end = time.time()
print("time elapsed: ", end - start, " seconds")

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
start = time.time()

model = Sequential()
model.add(LSTM(200, input_shape=(TIME_STEP, WORD_NUMERIC_VECTOR_SIZE), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

configuration.print_configuration()

end = time.time()
print("time elapsed: ", end - start, " seconds")

print('Train...')
start = time.time()
model.fit(x_train, y_train,
          batch_size=32,
          epochs=EPOCHS_NUMBER,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=32)

print('Score: %f' % score)
print('Test accuracy: %f%%' % (acc * 100))
end = time.time()
print("time elapsed: ", end - start, " seconds")
