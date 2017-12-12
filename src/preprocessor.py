import itertools

from gensim import corpora, matutils

from src.my_corpus import MyCorpus, STOP_LIST

FILE_SEPARATOR = "/"
TEST_DATA_PERCENTAGE = 30.0


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def preprocess(filename):
        documents = []
        with open(filename, encoding='utf-8', errors='ignore') as data_file:
            iter_file = iter(data_file)
            for line in iter_file:
                documents.append(line)
        # remove common words and tokenize
        texts = [[word for word in document.lower().split() if word not in STOP_LIST]
                 for document in documents]
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [[token for token in text if frequency[token] > 1]
                 for text in texts]

        dictionary = corpora.Dictionary(texts)
        dictionary_filename = filename.split(FILE_SEPARATOR)[-1]
        dictionary_path = "../tmp/" + dictionary_filename + ".dict"
        dictionary.save(dictionary_path)  # store the dictionary, for future reference
        print("File saved in: " + dictionary_path)

    @staticmethod
    def fetch_data():
        dictionary_neg = corpora.Dictionary.load("../tmp/rt-polarity.neg.dict")
        dictionary_pos = corpora.Dictionary.load("../tmp/rt-polarity.pos.dict")

        print(dictionary_neg)
        print(dictionary_pos)

        corpus_pos = MyCorpus("../data/rt-polaritydata/rt-polarity.pos", dictionary_pos)
        corpus_neg = MyCorpus("../data/rt-polaritydata/rt-polarity.neg", dictionary_neg)
        loaded_neg_corpus = [vector for vector in corpus_neg]
        loaded_pos_corpus = [vector for vector in corpus_pos]

        dict_neg2pos = dictionary_pos.merge_with(dictionary_neg)
        print(dict_neg2pos)
        # now we can merge corpora from the two incompatible dictionaries into one
        merged_corpus = itertools.chain(loaded_pos_corpus, dict_neg2pos[loaded_neg_corpus])

        positives_number = len(loaded_pos_corpus)
        negatives_number = len(loaded_neg_corpus)

        corpus_matrix = matutils.corpus2dense(merged_corpus, num_terms=len(dictionary_pos.keys()))
        corpus_matrix = corpus_matrix.transpose()

        train_samples_count = int(round(positives_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
        x_train_pos = [corpus_matrix[i] for i in range(train_samples_count)]
        y_train_pos = [1 for _ in range(train_samples_count)]
        x_test_pos = [corpus_matrix[i] for i in range(train_samples_count, positives_number)]
        y_test_pos = [1 for _ in range(train_samples_count, positives_number)]

        train_samples_count = int(round(negatives_number * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
        x_train_neg = [corpus_matrix[i] for i in range(positives_number, positives_number + train_samples_count)]
        y_train_neg = [0 for _ in range(train_samples_count)]
        x_test_neg = [corpus_matrix[i] for i in
                      range(positives_number + train_samples_count, positives_number + negatives_number)]
        y_test_neg = [0 for _ in range(train_samples_count, negatives_number)]

        x_train = x_train_pos + x_train_neg
        y_train = y_train_pos + y_train_neg

        x_test = x_test_pos + x_test_neg
        y_test = y_test_pos + y_test_neg

        result = (x_train, y_train), (x_test, y_test)
        return result

# Preprocessor.preprocess(filename="../data/rt-polaritydata/rt-polarity.neg")
# Preprocessor.preprocess(filename="../data/rt-polaritydata/rt-polarity.pos")
# Preprocessor.fetch_data()
