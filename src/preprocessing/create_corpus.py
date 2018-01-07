from random import randrange

from gensim import utils

from src.configuration import CORPUS_FILES
from src.utils.get_file import full_path

STOP_LIST = set('for a of the and to in'.split())
PRINT_STATS = True


def create_corpus_and_labels():
    negative_corpus = _load_corpus(filename=full_path(CORPUS_FILES["negative"]))
    positive_corpus = _load_corpus(filename=full_path(CORPUS_FILES["positive"]))
    negatives_number = len(negative_corpus)
    positives_number = len(positive_corpus)

    corpus = []
    labels = []
    for i in range(min(negatives_number, positives_number)):
        corpus.append(negative_corpus[i])
        labels.append([0])
        corpus.append(positive_corpus[i])
        labels.append([1])

    remainder = negative_corpus if negatives_number > positives_number else positive_corpus
    label = 0 if negatives_number > positives_number else 1
    remainder = remainder[min(negatives_number, positives_number):]

    labels += [label for _ in range(abs(positives_number - negatives_number))]
    corpus += remainder

    if PRINT_STATS:
        print("average document length %d, negatives: %d, positives %d, total %d"
              % get_corpus_stats(corpus, negatives_number, positives_number))
        print(corpus[randrange(len(corpus))])
    return corpus, labels


def _load_corpus(filename: str):
    documents_tokenized = []
    # porter_stemmer = PorterStemmer()

    with open(filename, encoding='utf-8', errors='ignore') as data_file:
        iter_file = iter(data_file)
        for line in iter_file:
            # stemmed_line = porter_stemmer.stem_sentence(line)
            tokens = list(utils.tokenize(line, deacc=True, lower=True))
            documents_tokenized.append(tokens)
    return documents_tokenized


def get_corpus_stats(corpus, negatives_number, positives_number):
    average_document_length = 0
    documents_count = len(corpus)

    for document in corpus:
        average_document_length += len(document)

    average_document_length /= negatives_number + positives_number
    return average_document_length, negatives_number, positives_number, documents_count
