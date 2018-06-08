import pickle
from collections import Counter

import numpy
from keras.callbacks import ModelCheckpoint
from kutilities.callbacks import MetricsCallback, WeightsCallback, \
    PlottingCallback
from kutilities.helpers.data_preparation import get_labels_to_categories_map, \
    get_class_weights2, onehot_to_categories
from sklearn.metrics import precision_score, accuracy_score, \
    mean_absolute_error
from sklearn.metrics import recall_score

from dataset.data_loader import SemEvalDataLoader
from models.nn_models import target_RNN
from utilities.data_loader import get_embeddings, Task4Loader, prepare_dataset

numpy.random.seed(1337)  # for reproducibility

# specify the word vectors file to use.
# for example, WC_CORPUS = "own.twitter" and WC_DIM = 300,
# correspond to the file "datastories.twitter.300d.txt"
WV_CORPUS = "datastories.twitter"
WV_DIM = 300

# Flag that sets the training mode.
# - if FINAL == False,  then the dataset will be split in {train, val, test}
# - if FINAL == True,   then the dataset will be split in {train, val}.
# Even for training the model for the final submission a small percentage
# of the labeled data will be kept for as a validation set for early stopping
FINAL = True

# If True, the SemEval gold labels will be used as the testing set in order to perform Post-mortem analysis
POST_MORTEM = True

text_max_length = 50
target_max_length = 6
TASK = "CE"  # Specify the Subtask. It is needed to correctly load the data

############################################################################
# PERSISTENCE
############################################################################
# if True save model checkpoints, as well as the corresponding word indices
# you HAVE tp set PERSIST = True, in order to be able to use the trained model later
PERSIST = False
best_model = lambda: "cp_model_task4_sub{}.hdf5".format(TASK)
best_model_word_indices = lambda: "cp_model_task4_sub{}_word_indices.pickle".format(
    TASK)

############################################################################
# LOAD DATA
############################################################################
embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

if TASK == "BD":
    loader = Task4Loader(word_indices,
                         text_lengths=(target_max_length, text_max_length),
                         subtask=TASK,
                         filter_classes={"positive", "negative"},
                         y_one_hot=False)
    classes = ['positive', 'negative']
else:
    loader = Task4Loader(word_indices,
                         text_lengths=(target_max_length, text_max_length),
                         subtask=TASK)
    classes = ["-2", "-1", "0", "1", "2"]

if PERSIST:
    pickle.dump(word_indices, open(best_model_word_indices(), 'wb'))

if FINAL:
    print("\n > running in FINAL mode!\n")
    training, testing = loader.load_final()
else:
    training, validation, testing = loader.load_train_val_test()

if POST_MORTEM:
    print("\n > running in Post-Mortem mode!\n")
    gold_data = SemEvalDataLoader().get_gold(task=TASK)
    gX = [obs[1] for obs in gold_data]
    gy = [obs[0] for obs in gold_data]
    gold = prepare_dataset(gX, gy, loader.pipeline, loader.y_one_hot)

    validation = testing
    testing = gold
    FINAL = False

############################################################################
# NN MODEL
############################################################################
print("Building NN Model...")
nn_model = target_RNN(embeddings,
                      tweet_max_length=text_max_length,
                      aspect_max_length=target_max_length,
                      noise=0.2,
                      activity_l2=0.001,
                      drop_text_rnn_U=0.2,
                      drop_text_input=0.3,
                      drop_text_rnn=0.3,
                      drop_target_rnn=0.2,
                      use_final=True,
                      bi=True,
                      final_size=64,
                      drop_final=0.5,
                      lr=0.001,
                      rnn_cells=64,
                      attention="context",
                      clipnorm=.1,
                      classes=len(classes))

print(nn_model.summary())

############################################################################
# CALLBACKS
############################################################################

# define metrics and class weights
if TASK == "BD":
    cat_to_class_mapping = {v: k for k, v in
                            get_labels_to_categories_map(classes).items()}
    metrics = {
        "accuracy": (lambda y_test, y_pred: accuracy_score(y_test, y_pred)),
        "recall": (lambda y_test, y_pred: recall_score(y_test, y_pred,
                                                       average='macro')),
        "precision": (lambda y_test, y_pred: precision_score(y_test, y_pred,
                                                             average='macro'))
    }
else:
    cat_to_class_mapping = {v: int(k) for k, v in
                            get_labels_to_categories_map(classes).items()}


    def macro_mae(y_test, y_pred):
        _y_test = [cat_to_class_mapping[y] for y in y_test]
        _y_pred = [cat_to_class_mapping[y] for y in y_pred]

        c = Counter(_y_pred)
        print(c)

        classes = set(_y_test)
        micro_m = {}
        for c in classes:
            class_sentences = [(t, p) for t, p in zip(_y_test, _y_pred) if
                               t == c]
            yt = [y[0] for y in class_sentences]
            yp = [y[1] for y in class_sentences]
            micro_m[c] = mean_absolute_error(yt, yp)
        # pprint.pprint(sorted(micro_m.items(), key=lambda x: x[1], reverse=True))
        return numpy.mean(list(micro_m.values()))


    metrics = {
        "macro_mae": macro_mae,
        "micro_mae": (
            lambda y_test, y_pred: mean_absolute_error(y_test, y_pred)),
    }

_datasets = {}
_datasets["1-train"] = (training[0], training[1]),
_datasets["2-val"] = (validation[0], validation[1]) if not FINAL else (
    testing[0], testing[1])
if not FINAL:
    _datasets["3-test"] = (testing[0], testing[1])

metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
weights = WeightsCallback(parameters=["W"], stats=["raster", "mean", "std"])

if TASK == "BD":
    plotting = PlottingCallback(grid_ranges=(0.75, 1), height=4,
                                benchmarks={"ρ": 0.797, "α": 0.87})
    checkpointer = ModelCheckpoint(filepath=best_model(), monitor='val.recall',
                                   mode="max", verbose=1, save_best_only=True)
else:
    plotting = PlottingCallback(grid_ranges=(0.4, 1.), height=4,
                                benchmarks={"MAE_M": 0.719, "MAE_m": 0.58})
    checkpointer = ModelCheckpoint(filepath=best_model(),
                                   monitor='val.macro_mae', mode="min",
                                   verbose=1, save_best_only=True)

_callbacks = []
_callbacks.append(metrics_callback)
_callbacks.append(plotting)
_callbacks.append(weights)

############################################################################
# APPLY CLASS WEIGHTS
############################################################################
if TASK == "BD":
    class_weights = get_class_weights2(training[1], smooth_factor=0)
else:
    class_weights = get_class_weights2(onehot_to_categories(training[1]),
                                       smooth_factor=0.1)

print("Class weights:",
      {cat_to_class_mapping[c]: w for c, w in class_weights.items()})

history = nn_model.fit(training[0], training[1],
                       validation_data=(
                           validation[0], validation[1]) if not FINAL else (
                           testing[0], testing[1]),
                       nb_epoch=50, batch_size=64, class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history,
            open("hist_task4_sub{}.pickle".format(TASK), "wb"))
