from random import randrange

from src.utils.get_file import full_path

STOP_LIST = set('for a of the and to in'.split())
PRINT_STATS = True


def create_corpus():
    negative_corpus = _load_corpus(filename=full_path("data/rt-polaritydata/rt-polarity.neg"))
    positive_corpus = _load_corpus(filename=full_path("data/rt-polaritydata/rt-polarity.pos"))
    negatives_number = len(negative_corpus)
    positives_number = len(positive_corpus)
    corpus = negative_corpus + positive_corpus

    if PRINT_STATS:
        print("average document length %d, negatives: %d, positives %d, total %d"
              % get_corpus_stats(corpus, negatives_number, positives_number))
        print(corpus[randrange(len(corpus))])
    return corpus, negatives_number, positives_number


def _load_corpus(filename: str):
    documents = []
    with open(filename, encoding='utf-8', errors='ignore') as data_file:
        iter_file = iter(data_file)
        for line in iter_file:
            documents.append(line)
    # remove common words and tokenize
    return [[word for word in document.lower().split() if word not in STOP_LIST]
            for document in documents]


def get_corpus_stats(corpus, negatives_number, positives_number):
    average_document_length = 0
    documents_count = len(corpus)

    for document in corpus:
        average_document_length += len(document)

    average_document_length /= negatives_number + positives_number
    return average_document_length, negatives_number, positives_number, documents_count
