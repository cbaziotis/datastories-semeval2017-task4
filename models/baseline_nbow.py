import numpy

from dataset.data_loader import SemEvalDataLoader
from utilities.data_loader import get_embeddings
from utilities.sklearn import eval_clf, nbow_model

numpy.random.seed(1337)  # for reproducibility


def tok(text):
    return text


WV_CORPUS = "datastories.twitter"
WV_DIM = 300
embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

train_set = SemEvalDataLoader(verbose=False).get_data(task="A",
                                                      years=None,
                                                      datasets=None,
                                                      only_semeval=True)
X = [obs[1] for obs in train_set]
y = [obs[0] for obs in train_set]

test_data = SemEvalDataLoader(verbose=False).get_gold(task="A")
X_test = [obs[1] for obs in test_data]
y_test = [obs[0] for obs in test_data]

print("-----------------------------")
print("LinearSVC")
nbow = nbow_model("clf", embeddings, word_indices)
nbow.fit(X, y)
results = eval_clf(nbow.predict(X_test), y_test)
for res, val in results.items():
    print("{}: {:.3f}".format(res, val))
print("-----------------------------")
