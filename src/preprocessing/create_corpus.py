from src.utils.get_file import full_path

STOP_LIST = set('for a of the and to in'.split())


def create_corpus():
    negative_corpus = _load_corpus(filename=full_path("data/rt-polaritydata/rt-polarity.neg"))
    positive_corpus = _load_corpus(filename=full_path("data/rt-polaritydata/rt-polarity.pos"))
    negatives_number = len(negative_corpus)
    positives_number = len(positive_corpus)
    corpus = negative_corpus + positive_corpus
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
