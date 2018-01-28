import os

from src.configuration import get_vector_labels_file_name, get_vector_words_directory
from src.utils.get_file import full_path


def trec_preprocess():
    # see http://cogcomp.org/Data/QA/QC/
    trec_text_file = full_path("data/trec/trec.txt")
    labels_file_name = get_vector_labels_file_name('trec')
    word_vectors_directory = get_vector_words_directory('trec')
    corpus_file_name = "data/trec/trec.corp"

    create_corpus_file(trec_text_file)


def create_corpus_file(trec_text_file):
    labels = {}
    buffer = []
    with open(trec_text_file, 'r', encoding='utf-8', errors='ignore') as file:
        iter_file = iter(file)
        for line in iter_file:
            label = line[:str(line).find(":")]
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1
            line = line[str(line).find(" ") + 1:]

            # buffer += line
    print(labels)
trec_preprocess()
