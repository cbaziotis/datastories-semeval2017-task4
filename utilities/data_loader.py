"""
Created by Christos Baziotis.
"""
import random

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from kutilities.helpers.data_preparation import print_dataset_statistics, \
    labels_to_categories, categories_to_onehot
from sklearn.cross_validation import train_test_split

from dataset.data_loader import SemEvalDataLoader
from sklearn.pipeline import Pipeline
import numpy
from embeddings.WordVectorsManager import WordVectorsManager
from modules.CustomPreProcessor import CustomPreProcessor
from modules.EmbeddingsExtractor import EmbeddingsExtractor


def prepare_dataset(X, y, pipeline, y_one_hot=True, y_as_is=False):
    try:
        print_dataset_statistics(y)
    except:
        pass

    X = pipeline.fit_transform(X)

    if y_as_is:
        try:
            return X, numpy.asarray(y, dtype=float)
        except:
            return X, y

    # 1 - Labels to categories
    y_cat = labels_to_categories(y)

    if y_one_hot:
        # 2 - Labels to one-hot vectors
        return X, categories_to_onehot(y_cat)

    return X, y_cat


def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(corpus, dim).read()
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    wv_map = {}
    pos = 0
    # +1 for zero padding token and +1 for unk
    emb_matrix = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > 199:
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector

    # add unknown token
    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)

    return emb_matrix, wv_map


def prepare_text_only_dataset(X, pipeline):
    X = pipeline.fit_transform(X)
    return X


def data_splits(dataset, final=False):
    '''
    Splits a dataset in parts
    :param dataset:
    :param final: Flag that indicates if we want a split for tha final submission or for normal training
    :return:
    '''
    if final:
        # 95% training and 5% validation
        train_ratio = 0.95
        train_split_index = int(len(dataset) * train_ratio)

        training = dataset[:train_split_index]
        test = dataset[train_split_index:]

        return training, test
    else:
        # 80% training, 10% validation and 10% testing
        train_ratio = 0.8
        val_test_ratio = 0.5
        train_split_index = int(len(dataset) * train_ratio)
        val_test_split_index = int(
            (len(dataset) - train_split_index) * val_test_ratio)

        training = dataset[:train_split_index]
        rest = dataset[train_split_index:]
        validation = rest[:val_test_split_index]
        test = rest[val_test_split_index:]

        return training, validation, test


class Task4Loader:
    """
    Task 4: Sentiment Analysis in Twitter
    """

    def __init__(self, word_indices, text_lengths, subtask="A", silver=False,
                 **kwargs):

        self.word_indices = word_indices

        filter_classes = kwargs.get("filter_classes", None)
        self.y_one_hot = kwargs.get("y_one_hot", True)

        self.pipeline = Pipeline([
            ('preprocess', CustomPreProcessor(TextPreProcessor(
                backoff=['url', 'email', 'percent', 'money', 'phone', 'user',
                         'time', 'url', 'date', 'number'],
                include_tags={"hashtag", "allcaps", "elongated", "repeated",
                              'emphasis', 'censored'},
                fix_html=True,
                segmenter="twitter",
                corrector="twitter",
                unpack_hashtags=True,
                unpack_contractions=True,
                spell_correct_elong=False,
                tokenizer=SocialTokenizer(lowercase=True).tokenize,
                dicts=[emoticons]))),
            ('ext', EmbeddingsExtractor(word_indices=word_indices,
                                        max_lengths=text_lengths,
                                        add_tokens=(False,
                                                    True) if subtask != "A" else True,
                                        unk_policy="random"))])

        # loading data
        print("Loading data...")
        dataset = SemEvalDataLoader(verbose=False).get_data(task=subtask,
                                                            years=None,
                                                            datasets=None,
                                                            only_semeval=True)
        random.Random(42).shuffle(dataset)

        if filter_classes:
            dataset = [d for d in dataset if d[0] in filter_classes]

        self.X = [obs[1] for obs in dataset]
        self.y = [obs[0] for obs in dataset]
        print("total observations:", len(self.y))

        print("-------------------\ntraining set stats\n-------------------")
        print_dataset_statistics(self.y)
        print("-------------------")

        if silver:
            print("Loading silver data...")
            dataset = SemEvalDataLoader().get_silver()
            self.silver_X = [obs[1] for obs in dataset]
            self.silver_y = [obs[0] for obs in dataset]
            print("total observations:", len(self.silver_y))

    def load_train_val_test(self, only_test=False):
        X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y,
                                                            test_size=0.3,
                                                            stratify=self.y,
                                                            random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest,
                                                        test_size=0.5,
                                                        stratify=y_rest,
                                                        random_state=42)

        if not only_test:
            print("\nPreparing training set...")
            training = prepare_dataset(X_train, y_train, self.pipeline,
                                       self.y_one_hot)
            print("\nPreparing validation set...")
            validation = prepare_dataset(X_val, y_val, self.pipeline,
                                         self.y_one_hot)
        print("\nPreparing test set...")
        testing = prepare_dataset(X_test, y_test, self.pipeline,
                                  self.y_one_hot)

        if only_test:
            return testing
        else:
            return training, validation, testing

    def load_final(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.1,
                                                            stratify=self.y,
                                                            random_state=27)
        print("\nPreparing training set...")
        training = prepare_dataset(X_train, y_train, self.pipeline,
                                   self.y_one_hot)
        print("\nPreparing test set...")
        testing = prepare_dataset(X_test, y_test, self.pipeline,
                                  self.y_one_hot)
        return training, testing
