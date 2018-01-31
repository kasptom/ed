import keras
import numpy as np

from src.preprocessing.create_corpus import STOP_LIST
from src.preprocessing.document_as_w2v_groups import document_to_batch
from src.preprocessing.w2v_loader import load_google_w2v_model
from src.utils.get_file import full_path
from gensim import utils

sentiment = {0: "negative", 1: "positive"}


def check_imdb_lstm_network():
    model = load_google_w2v_model()
    line_negative = \
        "OK, I see that the movie has many naysayers. I was one of them when I saw the film in 1972, and I was " \
        "only fifteen at the time. I could go on and on about the film's myriad failures. It is contrived, " \
        "self-important, at times even poorly staged. Which brings me to my point. A lot of people seem to forget " \
        "that Coppola did not win Best Director-- Bob Fosse (for 'Cabaret') did, and deservedly so. He did a much " \
        "better job. That is one of the eight Oscars that 'Cabaret' won.The other seven just happen to be Art " \
        "Direction, Cinematography, Sound, Editing,Original Score, Best Supporting Actor, and Best Actress. " \
        "So when the time came to open the envelope and announce Best Picture, the Award goes instead to a film " \
        "that, by that point, had won only two statues (for Actor and Adapted Screenplay). How does any movie win " \
        "eight Academy Awards and fail to grab Best Picture? With that in mind, 'The Godfather' is not merely " \
        "arrogant film-making. Its history and legacy,both--just like its protagonists-- are just downright " \
        "larcenous."
    line_positive = \
        "The Godfather is an extravaganza, nigh flawless, a cinematic magnum opus, ubiquitously " \
        "acclaimed for its brilliance and for being in a league of its own. The Godfather doesn't depict " \
        "poetic justice but rather portrays the triumph of perspicacious potency over abject " \
        "vulnerability. The Godfather is known, not for its cogency but for its eloquence. The movie " \
        "being star-studded is decorated with a plethora of supernal performances and it won't be a " \
        "hyperbole that almost every actor gave an Oscar worthy performance. Marlon Brando is " \
        "exceptionally brilliant in his sterling portrayal of Vito Corleone and so is Al Pacino in his " \
        "remarkable portrayal of Michael Corleone. The grandeur of Don Vito Corleone ironically " \
        "lies in his austerity and inexorable equanimity. The grandiosity of the movie is such, that " \
        "even the biggest complement made about it may sound like a picayune remark. The Godfather " \
        "may most aptly be described as an obituary of humanity, a requiem of mankind, owing to the " \
        "pervasive violence and the brutality that it portrays in an utmost sanguinary fashion. In a nutshell, the " \
        "movie has transcended all the limits of mortality only to achieve apotheosis."

    tokens_neg = list(utils.tokenize(line_negative, deacc=True, lower=True))
    document_neg = list(filter(lambda x: x not in STOP_LIST, tokens_neg))
    x_neg = np.array([document_to_batch(document_neg, model, 150)])

    tokens_pos = list(utils.tokenize(line_positive, deacc=True, lower=True))
    document_pos = list(filter(lambda x: x not in STOP_LIST, tokens_pos))
    x_pos = np.array([document_to_batch(document_pos, model, 150)])

    model = keras.models.load_model(full_path("lstm-net_BAK/imdb_timestep150_drout0.4_rdrout0.4_batch64.h5"))
    print(evaluate(model, x_neg))
    print(evaluate(model, x_pos))


def evaluate(model, document_batch):
    return sentiment[round(model.predict(document_batch)[0][0], 0)]


check_imdb_lstm_network()
