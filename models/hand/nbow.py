import random

from ekphrasis.classes.textpp import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.metrics import f1_score

from dataset.data_loader import SemEvalDataLoader
from sk_transformers.CustomPreProcessor import CustomPreProcessor
from sk_transformers.DBOWFeatureExtractor import DBOWFeatureExtractor
from utilities.data_loader import get_embeddings
from utilities.ignore_warnings import set_ignores

set_ignores()
import numpy

numpy.random.seed(1337)  # for reproducibility
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


def tok(text):
    return text


default_pp = TextPreProcessor(
    backoff=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
    include_tags={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons])

WV_CORPUS = "datastories.twitter"
WV_DIM = 300
embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

dbow = Pipeline([
    ('preprocess', CustomPreProcessor(default_pp, to_list=True)),
    ('ext', DBOWFeatureExtractor(aggregation="mean",
                                 negation=True,
                                 # context_diff="abs",
                                 word_vectors=embeddings,
                                 word_indices=word_indices,
                                 neg_comma=True, neg_modals=True,
                                 window=1,
                                 idf_weight=True, fs_weight=False,
                                 stopwords=False)),
    # ('normalizer', Normalizer(norm='l2', copy=False)),
    ('classifier', svm.LinearSVC(C=0.9, class_weight='balanced', loss='hinge', random_state=0)),
])

dataset = SemEvalDataLoader(verbose=False).get_data(task="A", years=None, datasets=None, only_semeval=True)
random.Random(42).shuffle(dataset)

X = [obs[1] for obs in dataset]
y = [obs[0] for obs in dataset]
# X = X[:100]
# y = y[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# print_dataset_statistics(y)
dbow.fit(X_train, y_train)

print("--------------------------")
y_predicted = dbow.predict(X_test)
print("baseline", f1_score(y_test, y_predicted, labels=['positive', 'negative'], average='macro') * 100)

dbow.fit(X_train, y_train)

print("--------------------------")
y_predicted = dbow.predict(X_test)
print("dbow", f1_score(y_test, y_predicted, labels=['positive', 'negative'], average='macro') * 100)
