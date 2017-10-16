from ekphrasis.classes.textpp import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import Normalizer, StandardScaler

from dataset.data_loader import SemEvalDataLoader
from sk_transformers.CustomPreProcessor import CustomPreProcessor
from sk_transformers.DBOWFeatureExtractor import DBOWFeatureExtractor
from utilities.data_loader import get_embeddings
from utilities.ignore_warnings import set_ignores

set_ignores()
import numpy

numpy.random.seed(1337)  # for reproducibility
from sklearn.pipeline import Pipeline


def eval_clf(model, X, y, X_test, y_test):
    model.fit(X, y)
    y_p = model.predict(X_test)

    f1 = round(f1_score(y_test, y_p, labels=['positive', 'negative'],
                        average='macro'), 3)
    recall = round(recall_score(y_test, y_p, average='macro'), 3)
    precision = round(precision_score(y_test, y_p, average='macro'), 3)
    print("f1", f1)
    print("recall", recall)
    print("precision", precision)
    print("{} & {} & {}".format(recall, f1, precision))


def tok(text):
    return text


default_pp = TextPreProcessor(
    backoff=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url',
             'date', 'number'],
    include_tags={"hashtag", "allcaps", "elongated", "repeated", 'emphasis',
                  'censored'},
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

# bot = Pipeline([
#     ('preprocess', CustomPreProcessor(default_pp, to_list=True)),
#     ('fw', TfidfVectorizer(tokenizer=tok,
#                            min_df=5,
#                            # max_df=0.9,
#                            sublinear_tf=True, binary=True,
#                            lowercase=False)),
#     ('classifier', svm.LinearSVC(C=0.6, class_weight='balanced',
#                                  loss='hinge',
#                                  random_state=0)),
#     # ('classifier', LogisticRegression(C=0.6, class_weight='balanced',
#     #                                   random_state=0, n_jobs=-1)),
# ])

dbow = Pipeline([
    ('preprocess', CustomPreProcessor(default_pp, to_list=True)),
    ('ext', DBOWFeatureExtractor(aggregation="mean",
                                 negation=True,
                                 # context_diff="abs",
                                 word_vectors=embeddings,
                                 word_indices=word_indices,
                                 neg_comma=False, neg_modals=True,
                                 # idf_weight=True, fs_weight=False,
                                 stopwords=True)),
    # ('normalizer', StandardScaler()),
    ('normalizer2', Normalizer(norm='l2', copy=False)),
    ('classifier', svm.LinearSVC(C=0.6, class_weight='balanced',
                                 loss='hinge',
                                 random_state=0)),
])

train_set = SemEvalDataLoader(verbose=False).get_data(task="A",
                                                      years=None,
                                                      datasets=None,
                                                      only_semeval=True)
X = [obs[1] for obs in train_set]
y = [obs[0] for obs in train_set]

test_data = SemEvalDataLoader(verbose=False).get_gold(task="A")
X_test = [obs[1] for obs in test_data]
y_test = [obs[0] for obs in test_data]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     stratify=y,
#                                                     random_state=42)

# print_dataset_statistics(y)
print("-----------------------------")
print("LinearSVC")
eval_clf(dbow, X, y, X_test, y_test)
# eval_clf(bot, X, y, X_test, y_test)
print("-----------------------------")
