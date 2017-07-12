from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class CustomPreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, pp, to_list=False):
        self.pp = pp
        self.to_list = to_list

    # @cached(cache={})
    def pre_process_doc(self, doc):
        if isinstance(doc, tuple) or isinstance(doc, list):
            return [self.pp.pre_process_doc(d) for d in doc]
        else:
            return self.pp.pre_process_doc(doc)

    def pre_process_steps(self, X):
        for x in tqdm(X, desc="PreProcessing..."):
            yield self.pre_process_doc(x)

    def transform(self, X, y=None):
        if self.to_list:
            return list(self.pre_process_steps(X))
        else:
            return self.pre_process_steps(X)

    def fit(self, X, y=None):
        return self
