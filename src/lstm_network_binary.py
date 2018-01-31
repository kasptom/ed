from keras import Sequential
from keras.layers import LSTM, Dense

from src.configuration import TIME_STEPS, RECURRENT_DROPOUT, WORD_NUMERIC_VECTOR_SIZE, DROPOUT
from src.lstm_net import LstmNet
from src.utils.data_generator import DataGenerator


class LstmBinaryNet(LstmNet):

    def build_model(self):
        model = Sequential()
        model.add(
            LSTM(200, input_shape=(TIME_STEPS, WORD_NUMERIC_VECTOR_SIZE), dropout=DROPOUT,
                 recurrent_dropout=RECURRENT_DROPOUT))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_data_generator(self):
        return DataGenerator()
