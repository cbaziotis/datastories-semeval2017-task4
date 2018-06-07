from collections import Counter

import matplotlib.pyplot as plt
import numpy
import seaborn as sns


def get_class_weights2(y, smooth_factor=0):
    """
    Returns the normalized weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


def onehot_to_categories(y):
    """
    Transform categorical labels to one-hot vectors
    :param y: list of one-hot vectors, ex. [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], ...]
    :return: list of categories, ex. [0, 2, 1, 2, 0, ...]
    """
    return numpy.asarray(y).argmax(axis=-1)


def get_class_labels(y):
    """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: sorted unique class labels
    """
    return numpy.unique(y)


def get_labels_to_categories_map(y):
    """
    Get the mapping of class labels to numerical categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: dictionary with the mapping
    """
    labels = get_class_labels(y)
    return {l: i for i, l in enumerate(labels)}


def print_dataset_statistics(y):
    """
    Returns the normalized weights for each class based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)
    print("Total:", len(y))
    # statistics = {
    #     c: str(counter[c]) + " (%.2f%%)" % (counter[c] / float(len(y)) * 100.0)
    #     for c in sorted(counter.keys())}
    statistics = {
        c: str("(%.2f%%)" % (counter[c] / float(len(y)) * 100.0))
        for c in sorted(counter.keys())}
    return statistics


TASK = "CE"
training = numpy.load(TASK + "_training.pickle")
testing = numpy.load(TASK + "_testing.pickle")
data = numpy.concatenate((training, testing), axis=0)
print()

smooth_factor = 0.1


def get_weights(sf):
    print()
    print("--------------")
    print("a={}".format(sf))
    if TASK == "BD":
        classes = ['positive', 'negative']
        class_weights = get_class_weights2(data, smooth_factor=sf)
        cat_to_class_mapping = {v: k for k, v in
                                get_labels_to_categories_map(classes).items()}
    else:
        classes = ["-2", "-1", "0", "1", "2"]
        class_weights = get_class_weights2(onehot_to_categories(data),
                                           smooth_factor=sf)
        cat_to_class_mapping = {v: int(k) for k, v in
                                get_labels_to_categories_map(classes).items()}

    print("Class weights:",
          {cat_to_class_mapping[c]: round(w, 3) for c, w in
           class_weights.items()})

    print("Class weights:",
          [(cat_to_class_mapping[c], round(w, 3)) for c, w in
           class_weights.items()])

    print("sum:", sum(class_weights.values()))

    weights = [(cat_to_class_mapping[c], w) for c, w in
               sorted(class_weights.items())]
    percs = print_dataset_statistics(onehot_to_categories(data))
    print(percs)
    print("--------------")
    percs = [(cat_to_class_mapping[c], w) for c, w in
             sorted(percs.items())]

    # x = [str(x[0]) + "\n" + str(p[1]) for x, p in zip(weights, percs)]
    x = [str(x[0]) for x, p in zip(weights, percs)]
    y = [x[1] for x in weights]

    return x, y


sns.set(style="white", palette="bright")
rs = numpy.random.RandomState(7)

sfs = [.0, .1, .5, 1, 5, 10]
plt_data = [get_weights(sf) for sf in sfs]

# Set up the matplotlib figure
fig, axs = plt.subplots(ncols=3, )

for ax, (x, y), sf in zip(axs, plt_data, sfs):
    sns.barplot(x=x, y=y, ax=ax)
    ax.set_title('smoothing factor α={}'.format(sf))

# Define some hatches
hatches = ['-', 'x', '.', '/', '\\', '*', 'o']

# Loop over the bars
for ax in axs:
    # for i, thisbar in enumerate(ax.patches):
    #     thisbar.set_hatch(hatches[i])
    ax.set(xlabel='classes', ylabel='weights')
    # ax.tick_params(labelsize=8)

fig.set_size_inches(8, 4)
# fig.subplots_adjust(wspace=0.5, hspace=0.05)
fig.tight_layout(pad=0.4, w_pad=2, h_pad=1.0)
# fig.tight_layout()
plt.savefig('smoothig_{}.png'.format(TASK), bbox_inches='tight')
plt.show()
