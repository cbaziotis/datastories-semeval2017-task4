import numpy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score, f1_score, \
    precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR

from modules.CustomPreProcessor import CustomPreProcessor
from modules.NBOWVectorizer import NBOWVectorizer


def eval_reg(y_hat, y):
    results = {
        "pearson": pearsonr([float(x) for x in y_hat],
                            [float(x) for x in y])[0]
    }

    return results


def eval_clf(y_test, y_p):
    results = {
        "f1": f1_score(y_test, y_p, average='macro'),
        "recall": recall_score(y_test, y_p, average='macro'),
        "precision": precision_score(y_test, y_p, average='macro'),
        "accuracy": accuracy_score(y_test, y_p)
    }

    return results


def eval_mclf(y, y_hat):
    results = {
        "jaccard": jaccard_similarity_score(numpy.array(y),
                                            numpy.array(y_hat)),
        "f1-macro": f1_score(numpy.array(y), numpy.array(y_hat),
                             average='macro'),
        "f1-micro": f1_score(numpy.array(y), numpy.array(y_hat),
                             average='micro')
    }

    return results


def bow_model(task, max_features=10000):
    if task == "clf":
        algo = LogisticRegression(C=0.6, random_state=0,
                                  class_weight='balanced')
    elif task == "reg":
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError("invalid task!")

    word_features = TfidfVectorizer(ngram_range=(1, 1),
                                    tokenizer=lambda x: x,
                                    analyzer='word',
                                    min_df=5,
                                    # max_df=0.9,
                                    lowercase=False,
                                    use_idf=True,
                                    smooth_idf=True,
                                    max_features=max_features,
                                    sublinear_tf=True)
    preprocessor = TextPreProcessor(
        backoff=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
                 'url',
                 'date', 'number'],
        include_tags={"hashtag", "allcaps", "elongated", "repeated",
                      'emphasis',
                      'censored'},
        fix_html=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons])

    model = Pipeline([
        ('preprocess', CustomPreProcessor(preprocessor, to_list=True)),
        ('bow-feats', word_features),
        ('normalizer', Normalizer(norm='l2')),
        ('clf', algo)
    ])

    return model


def nbow_model(task, embeddings, word2idx):
    if task == "clf":
        algo = LogisticRegression(C=0.6, random_state=0,
                                  class_weight='balanced')
    elif task == "reg":
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError("invalid task!")

    embeddings_features = NBOWVectorizer(aggregation=["mean"],
                                         embeddings=embeddings,
                                         word2idx=word2idx,
                                         stopwords=False)

    preprocessor = TextPreProcessor(
        backoff=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
                 'url',
                 'date', 'number'],
        include_tags={"hashtag", "allcaps", "elongated", "repeated",
                      'emphasis',
                      'censored'},
        fix_html=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons])

    model = Pipeline([
        ('preprocess', CustomPreProcessor(preprocessor, to_list=True)),
        ('embeddings-feats', embeddings_features),
        ('normalizer', Normalizer(norm='l2')),
        ('clf', algo)
    ])

    return model
