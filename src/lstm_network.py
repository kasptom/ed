# coding=utf-8
from __future__ import print_function

import numpy as np
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from src.preprocessor import Preprocessor

"""
Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
"""

np.random.seed(7)

top_words = 5000
max_review_length = 100

print('Loading data...')
(x_train, y_train), (x_test, y_test) = Preprocessor.fetch_data()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('Train...')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Score: %d' % score)
print('Test accuracy: %d%%' % acc)
