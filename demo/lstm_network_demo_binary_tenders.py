import keras
import numpy as np
from gensim import utils
from gensim.models import Word2Vec

from src.configuration import DATA_SET_TENDERS
from src.preprocessing.create_corpus import STOP_LIST
from src.preprocessing.document_as_w2v_groups import document_to_batch
from src.utils.get_file import full_path

sentiment = {0: "negative", 1: "positive"}


def check_tender_lstm_network():
    w2v_model = Word2Vec.load(full_path('data/word2vecs/tenders_model'))
    model = keras.models.load_model(
        full_path("lstm-net-backups/tenders_timestep150_drout0.4_rdrout0.4_batch64_1.h5")
    )

    with open(full_path(DATA_SET_TENDERS['positive'])) as positive_corpus:
        positives_count = 0
        overall_count = 0
        for line in positive_corpus:
            tokens_pos = list(utils.tokenize(line, deacc=True, lower=True))
            document_pos = list(filter(lambda x: x not in STOP_LIST, tokens_pos))
            x_pos = np.array([document_to_batch(document_pos, w2v_model, 150)])
            overall_count += 1
            if evaluate(model, x_pos) == 'positive':
                positives_count += 1
            print(evaluate(model, x_pos))
        print('positives detected: ' + str(positives_count) + '/' + str(overall_count))


def evaluate(model, document_batch):
    return sentiment[round(model.predict(document_batch)[0][0], 0)]


check_tender_lstm_network()
