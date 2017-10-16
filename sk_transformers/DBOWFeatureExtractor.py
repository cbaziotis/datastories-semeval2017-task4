import nltk
import numpy as np
from ekphrasis.utils.nlp import doc_ngrams, find_negations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

# @profile
from utilities.evaluation import most_discriminative_features


class DBOWFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, aggregation,
                 word_vectors=None,
                 word_indices=None,
                 negation=False, context_diff=None,
                 neg_comma=False, neg_modals=False,
                 idf_weight=False, fs_weight=False,
                 stopwords=True,
                 window=1):
        self.idf_weight = idf_weight
        self.fs_weight = fs_weight
        self.stopwords = stopwords
        self.aggregation = aggregation
        self.negation = negation
        self.context_diff = context_diff
        self.neg_comma = neg_comma
        self.neg_modals = neg_modals
        self.window = window
        self.tfidf_weights = None
        self.fscores = None
        self.vectorizer = None

        self.word_vectors = word_vectors
        self.word_indices = word_indices

        self.vec_dim = word_vectors[0].size
        self.stops = set(nltk.corpus.stopwords.words('english'))

    def context_difference(self, aff, neg):
        if self.context_diff == "normal":
            return aff - neg
        elif self.context_diff == "abs":
            # an = np.linalg.norm(aff, ord=1)
            # nn = np.linalg.norm(neg, ord=1)
            # s = np.sign(an-nn)
            diff = np.absolute(aff - neg)
            # return np.multiply(diff, s)
            return diff
        else:
            return []

    def aggregate_vectors(self, vectors, operation):

        if len(vectors) > (self.window - 1):
            vectors = doc_ngrams(vectors, n_from=self.window, n_to=self.window)
            vectors = [np.hstack([vec for vec in v]) for v in vectors]

            if operation == "sum":
                return np.sum(np.array(vectors), axis=0)
            if operation == "mean":
                return np.mean(np.array(vectors), axis=0)
            if operation == "min":
                return np.amin(np.array(vectors), axis=0)
            if operation == "max":
                return np.amax(np.array(vectors), axis=0)
            if operation == "minmax":
                max_vec = np.amax(np.array(vectors), axis=0)
                min_vec = np.amin(np.array(vectors), axis=0)
                return np.hstack([min_vec, max_vec])

        elif len(vectors) > 0:
            vec = np.array([item for sublist in vectors for item in sublist])
            return np.hstack(
                [vec, np.zeros(self.vec_dim * (self.window - len(vectors)))]
            )

        else:
            if operation == "minmax":
                return np.zeros(2 * self.vec_dim * self.window)
            else:
                return np.zeros(self.vec_dim * self.window)

    def extract_doc_vec(self, docs):
        feats = []
        append = feats.append
        for doc in docs:
            aff_vectors = []
            append_aff_vectors = aff_vectors.append
            neg_vectors = []
            append_neg_vectors = neg_vectors.append

            doc_terms = [term for term in doc if term[1] > 0]
            # print(len(doc), len(doc_terms), len(doc) - len(doc_terms))

            if not self.stopwords:
                doc_terms = [term for term in doc_terms
                             if term[0] not in self.stops]

            if self.idf_weight or self.fs_weight:
                if self.idf_weight:
                    doc_terms = [term for term in doc_terms
                                 if term[0] in self.tfidf_weights]
                else:
                    doc_terms = [term[0] for term in doc_terms
                                 if term[0] in self.fscores]

            if self.negation:
                negs = find_negations(doc,
                                      neg_comma=self.neg_comma,
                                      neg_modals=self.neg_modals)

            for i, term in enumerate(doc_terms):
                # vector = term[1]
                vector = self.word_vectors[term[1]]
                if self.idf_weight:
                    vector = np.multiply(vector, self.tfidf_weights[term[0]])

                if self.fs_weight:
                    vector = np.multiply(vector, self.fscores[term[0]])

                if self.negation and i in negs:
                    append_neg_vectors(vector)
                else:
                    append_aff_vectors(vector)

            # add the affirmative context vector
            aff_vec = self.aggregate_vectors(aff_vectors, self.aggregation)
            feat_stack = np.hstack([aff_vec])

            if self.negation:
                # add the negated context vector
                neg_vec = self.aggregate_vectors(neg_vectors, self.aggregation)
                feat_stack = np.hstack([feat_stack, neg_vec])

                if self.context_diff:
                    # add the difference between the two context vectors
                    feat_stack = np.hstack(
                        [feat_stack, self.context_difference(aff_vec, neg_vec)]
                    )

            append(feat_stack)

        return feats

    def index_text(self, sent, unk_policy="zero"):
        sent_words = []
        for token in sent:
            if token in self.word_indices:
                sent_words.append(self.word_indices[token])
            else:
                if unk_policy == "random":
                    sent_words.append(self.word_indices["<unk>"])
                elif unk_policy == "zero":
                    sent_words.append(0)
        return sent_words

    def fuse_text_vecs(self, X):
        Xs = []
        for sent in list(X):
            indexes = self.index_text(sent)
            merged = [(word, idx) for word, idx in zip(sent, indexes)]
            Xs.append(merged)
        return Xs

    def transform(self, X, y=None):
        # self.vec_dim = X[0][0][1].size
        X = self.fuse_text_vecs(X)
        return self.extract_doc_vec(X)  # actual feature extraction

    def fit(self, X, y=None):
        X = list(X)
        if self.idf_weight or self.fs_weight:
            count = CountVectorizer(
                lowercase=False,
                binary=False,
                tokenizer=lambda words: [word for word in words])
            _X = count.fit_transform(X)

            if self.idf_weight:
                tfidf_transformer = TfidfTransformer(sublinear_tf=True,
                                                     smooth_idf=True,
                                                     use_idf=True)
                tfidf_transformer.fit(_X)
                self.tfidf_weights = {name: score for name, score
                                      in zip(count.get_feature_names(),
                                             tfidf_transformer.idf_)}

            if self.fs_weight:
                mi = SelectKBest(score_func=mutual_info_classif, k="all")
                # mi = SelectKBest(score_func=chi2, k="all")
                mi.fit(_X, y)
                self.fscores = most_discriminative_features(count, mi)
        return self
