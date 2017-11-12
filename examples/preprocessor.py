"""
    source: https://radimrehurek.com/gensim/tutorial.html
"""
import logging
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# corpus of nine documents and twelve features
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]

# initialize transformation
tfidf = models.TfidfModel(corpus=corpus)

# convert the document to another vector representation
vec = [(0, 1), (4, 1)]
print(tfidf[vec])

# transform the whole corpus via TfIdf and index it, in preparation for similarity queries
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

# query the similarity of our query vector vec against every document in the corpus
sims = index[tfidf[vec]]
print(list(enumerate(sims)))
