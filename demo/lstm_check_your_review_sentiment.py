import keras
import numpy as np
from gensim import utils

from src.preprocessing.create_corpus import STOP_LIST
from src.preprocessing.document_as_w2v_groups import document_to_batch
from src.preprocessing.w2v_loader import load_google_w2v_model
from src.utils.get_file import full_path

sentiment = {0: "negative", 1: "positive"}


def review_your_review():
    print("Wait for the google w2v model to load...")
    w2v_model = load_google_w2v_model()
    net_model = keras.models.load_model(full_path("lstm-net-backups/imdb_timestep150_drout0.4_rdrout0.4_batch64.h5"))
    print("done")

    while True:
        line = input("Type in your review or \"quit\" to finish then press ENTER: ")

        if line == 'quit':
            break

        tokens_line = list(utils.tokenize(line, deacc=True, lower=True))
        document_review = list(filter(lambda x: x not in STOP_LIST, tokens_line))
        line_numeric = np.array([document_to_batch(document_review, w2v_model, 150)])

        print('I think this review is: ' + evaluate(net_model, line_numeric))
    print('Good bye')


def evaluate(model, document_batch):
    return sentiment[round(model.predict(document_batch)[0][0], 0)]


review_your_review()
