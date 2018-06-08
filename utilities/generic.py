import html
import itertools
from collections import Counter
from functools import reduce

import numpy
from kutilities.helpers.data_preparation import get_labels_to_categories_map
from sklearn.metrics import mean_absolute_error

flatten = lambda lst: reduce(
    lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i],
    lst, [])


def isplit(iterable, splitters):
    return [list(g) for k, g in
            itertools.groupby(iterable, lambda x: x in splitters) if not k]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def clean_text(text):
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def macro_mae(y_test, y_pred, classes):
    cat_to_class_mapping = {v: int(k) for k, v in
                            get_labels_to_categories_map(classes).items()}
    _y_test = [cat_to_class_mapping[y] for y in y_test]
    _y_pred = [cat_to_class_mapping[y] for y in y_pred]

    c = Counter(_y_pred)
    print(c)

    classes = set(_y_test)
    micro_m = {}
    for c in classes:
        class_sentences = [(t, p) for t, p in zip(_y_test, _y_pred) if t == c]
        yt = [y[0] for y in class_sentences]
        yp = [y[1] for y in class_sentences]
        micro_m[c] = mean_absolute_error(yt, yp)

    # pprint.pprint(sorted(micro_m.items(), key=lambda x: x[1], reverse=True))

    return numpy.mean(list(micro_m.values()))
