import numpy as np


def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'dont', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmatian', 'is', 'so', 'cute', 'i', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'wordless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    class_vec = [0, 1, 0, 1, 0, 1]

    return posting_list, class_vec


def create_vocal_list(data_set):
    vocal_set = set([])
    for document in data_set:
        vocal_set |= set(document)
    return list(vocal_set)


def set_of_words2vec(vocal_list, input_set):
    return_vec = [0] * len(vocal_list)
    for word in input_set:
        if word in vocal_list:
            return_vec[vocal_list.index(word)] = 1
    return return_vec


# naiwny klasyfikator Bayesowski - prawdopodobienstwo termu w danej klasie
def train_nbb(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_of_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p_0_num = np.zeros(num_of_words)
    p_1_num = np.zeros(num_of_words)
    p_0_denom = 0.0
    p_1_denom = 0.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p_1_num += train_matrix[i]
            p_1_denom += sum(train_matrix[i])
        else:
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])
    p_1_vec = p_1_num / p_1_denom
    p_0_vec = p_0_num / p_0_denom
    return p_0_vec, p_1_vec, p_abusive


def classify_nb(vec_2_classify, p0vec, p1vec, pclass1):
    p1 = sum(vec_2_classify * p1vec) + np.log(pclass1)
    p0 = sum(vec_2_classify * p0vec) + np.log(1 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


def test_nb():
    list_of_posts, list_of_classes = load_data_set()
    vocabulary = create_vocal_list(list_of_posts)
    train_matrix = [set_of_words2vec(vocabulary, sentence) for sentence in list_of_posts]
    p0, p1, p_abusive = train_nbb(train_matrix, list_of_classes)
    test_entry = ['love', 'my', 'dalmatian']
    this_doc = np.array(set_of_words2vec(vocabulary, test_entry))
    print "test entry classified as ", classify_nb(this_doc, p0, p1, p_abusive)

    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(vocabulary, test_entry))
    print "test entry classified as ", classify_nb(this_doc, p0, p1, p_abusive)


test_nb()
# top 5 error - nie jest w tej klasie co powinna ale znajduje sie w 5 najbardziej prawdopodobnych
# top 1 error - ...

# srednia harmoniczna - precission record
# metryka f1

# za tydzien ML i deep learning
# inne przyklady klasyfikatorow
    # Najlbizsi sasiedzi. Sprawdzamy do jakiego punktu mamy najblizej w przestrzeni
    # k-nearest neighbours
    # drzewa decyzyjne (przypadek szczegolny: random forest)

# entropia, index gn