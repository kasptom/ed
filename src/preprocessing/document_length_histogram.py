import matplotlib.pyplot as plt

from src.configuration import DATA_SET
from src.preprocessing.create_corpus import create_corpus_and_labels


def show_document_length_histogram(bin_count=30):
    corpus, labels = create_corpus_and_labels()
    sentence_lengths = []
    for document in corpus:
        sentence_lengths.append(len(document))

    plt.title(DATA_SET['label'])
    plt.xlabel("word count")
    plt.ylabel("document count")

    plt.hist(sentence_lengths, bins=bin_count)
    plt.show()


show_document_length_histogram(60)
