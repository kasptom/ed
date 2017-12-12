# coding=utf-8
from __future__ import print_function

import numpy as np
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential

from src.preprocessing.configuration import WORD_NUMERIC_VECTOR_SIZE
from src.preprocessing.w2v_preprocessor import corpus_to_vectors

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

max_features = WORD_NUMERIC_VECTOR_SIZE
batch_size = 32

print('Loading data...')

(x_train, y_train), (x_test, y_test) = corpus_to_vectors()

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
embedding_vecor_length = 64
model = Sequential()
model.add(
    Embedding(input_dim=max_features, batch_input_shape=(None, max_features), batch_size=batch_size, output_dim=128)
)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test)

print('Score: %f' % score)
print('Test accuracy: %f%%' % (acc * 100))
