from gensim import corpora

from src.my_corpus import MyCorpus, STOP_LIST

FILE_SEPARATOR = "/"


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def preprocess(filename):
        documents = []
        with open(filename) as data_file:
            iter_file = iter(data_file)
            for line in iter_file:
                documents.append(line)
        # remove common words and tokenize
        texts = [[unicode(word, errors='ignore') for word in document.lower().split() if word not in STOP_LIST]
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


# Preprocessor.preprocess(filename="../data/rt-polaritydata/rt-polarity.neg")
# Preprocessor.preprocess(filename="../data/rt-polaritydata/rt-polarity.pos")

dictionary_neg = corpora.Dictionary.load("../tmp/rt-polarity.neg.dict")
dictionary_pos = corpora.Dictionary.load("../tmp/rt-polarity.pos.dict")

print(dictionary_neg)
print(dictionary_pos)

corpus = MyCorpus("../data/rt-polaritydata/rt-polarity.neg", dictionary_neg)

for vector in corpus:  # load one vector into memory at a time
    print(vector)
