import nltk
import numpy
from sklearn.base import BaseEstimator, TransformerMixin


class NBOWVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, aggregation, embeddings=None, word2idx=None,
                 stopwords=True):
        self.aggregation = aggregation
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.dim = embeddings[0].size
        self.stopwords = stopwords
        self.stops = set(nltk.corpus.stopwords.words('english'))

    def aggregate_vecs(self, vectors):
        feats = []
        for method in self.aggregation:
            if method == "sum":
                feats.append(numpy.sum(vectors, axis=0))
            if method == "mean":
                feats.append(numpy.mean(vectors, axis=0))
            if method == "min":
                feats.append(numpy.amin(vectors, axis=0))
            if method == "max":
                feats.append(numpy.amax(vectors, axis=0))
        return numpy.hstack(feats)

    def transform(self, X, y=None):
        docs = []
        for doc in X:
            vectors = []
            for word in doc:
                if word not in self.word2idx:
                    continue
                if not self.stopwords and word in self.stops:
                    continue
                vectors.append(self.embeddings[self.word2idx[word]])
            if len(vectors) == 0:
                vectors.append(numpy.zeros(self.dim))
            feats = self.aggregate_vecs(numpy.array(vectors))
            docs.append(feats)
        return docs

    def fit(self, X, y=None):
        return self
