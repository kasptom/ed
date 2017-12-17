import os

from src.utils.get_file import full_path


def imdb_preprocess():
    base_directory = full_path("data/imdb/")

    pos_subdirectories = ["test/pos", "train/pos"]
    neg_subdirectories = ["test/neg", "train/neg"]

    neg_file_path = full_path("data/imdb.neg")
    pos_file_path = full_path("data/imdb.pos")

    create_corpus_file(base_directory, neg_file_path, neg_subdirectories)
    create_corpus_file(base_directory, pos_file_path, pos_subdirectories)


def create_corpus_file(base_directory, neg_file_path, neg_subdirectories):
    neg_file = open(neg_file_path, "w+")
    for subdirectory in neg_subdirectories:
        directory = os.fsdecode(base_directory + subdirectory)
        for file_name in os.listdir(directory):
            buffer = ""

            with open(directory + "/" + file_name, 'r', encoding='utf-8', errors='ignore') as file:
                iter_file = iter(file)
                for line in iter_file:
                    buffer += " " + line
            neg_file.write(buffer + "\n")
    neg_file.close()
    print(base_directory)


imdb_preprocess()
