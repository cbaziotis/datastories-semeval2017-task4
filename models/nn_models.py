from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, Flatten, \
    RepeatVector, MaxoutDense, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D, concatenate, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from kutilities.layers import AttentionWithContext, Attention, MeanOverTime
from sklearn import preprocessing


def embeddings_layer(max_length, embeddings, trainable=False, masking=False,
                     scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn


def build_attention_RNN(embeddings, classes, max_length, unit=LSTM, cells=64,
                        layers=1, **kwargs):
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)

    model = Sequential()
    model.add(embeddings_layer(max_length=max_length, embeddings=embeddings,
                               trainable=False, masking=True, scale=False,
                               normalize=False))

    if noise > 0:
        model.add(GaussianNoise(noise))
    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        model.add(get_RNN(unit, cells, bi, return_sequences=rs,
                          dropout_U=dropout_rnn_U))
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))

    if attention == "memory":
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == "simple":
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        # model.add(Highway())
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy')
    return model


def target_RNN(wv, tweet_max_length, aspect_max_length, classes=2, **kwargs):
    ######################################################
    # HyperParameters
    ######################################################
    noise = kwargs.get("noise", 0)
    trainable = kwargs.get("trainable", False)
    rnn_size = kwargs.get("rnn_size", 75)
    rnn_type = kwargs.get("rnn_type", LSTM)
    final_size = kwargs.get("final_size", 100)
    final_type = kwargs.get("final_type", "linear")
    use_final = kwargs.get("use_final", False)
    drop_text_input = kwargs.get("drop_text_input", 0.)
    drop_text_rnn = kwargs.get("drop_text_rnn", 0.)
    drop_text_rnn_U = kwargs.get("drop_text_rnn_U", 0.)
    drop_target_rnn = kwargs.get("drop_target_rnn", 0.)
    drop_rep = kwargs.get("drop_rep", 0.)
    drop_final = kwargs.get("drop_final", 0.)
    activity_l2 = kwargs.get("activity_l2", 0.)
    clipnorm = kwargs.get("clipnorm", 5)
    bi = kwargs.get("bi", False)
    lr = kwargs.get("lr", 0.001)

    attention = kwargs.get("attention", "simple")
    #####################################################
    shared_RNN = get_RNN(rnn_type, rnn_size, bi=bi, return_sequences=True,
                         dropout_U=drop_text_rnn_U)

    input_tweet = Input(shape=[tweet_max_length], dtype='int32')
    input_aspect = Input(shape=[aspect_max_length], dtype='int32')

    # Embeddings
    tweets_emb = embeddings_layer(max_length=tweet_max_length, embeddings=wv,
                                  trainable=trainable, masking=True)(
        input_tweet)
    tweets_emb = GaussianNoise(noise)(tweets_emb)
    tweets_emb = Dropout(drop_text_input)(tweets_emb)

    aspects_emb = embeddings_layer(max_length=aspect_max_length, embeddings=wv,
                                   trainable=trainable, masking=True)(
        input_aspect)
    aspects_emb = GaussianNoise(noise)(aspects_emb)

    # Recurrent NN
    h_tweets = shared_RNN(tweets_emb)
    h_tweets = Dropout(drop_text_rnn)(h_tweets)

    h_aspects = shared_RNN(aspects_emb)
    h_aspects = Dropout(drop_target_rnn)(h_aspects)
    h_aspects = MeanOverTime()(h_aspects)
    h_aspects = RepeatVector(tweet_max_length)(h_aspects)

    # Merge of Aspect + Tweet
    representation = concatenate([h_tweets, h_aspects])

    # apply attention over the hidden outputs of the RNN's
    att_layer = AttentionWithContext if attention == "context" else Attention
    representation = att_layer()(representation)
    representation = Dropout(drop_rep)(representation)

    if use_final:
        if final_type == "maxout":
            representation = MaxoutDense(final_size)(representation)
        else:
            representation = Dense(final_size, activation=final_type)(
                representation)
        representation = Dropout(drop_final)(representation)

    ######################################################
    # Probabilities
    ######################################################
    probabilities = Dense(1 if classes == 2 else classes,
                          activation="sigmoid" if classes == 2 else "softmax",
                          activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=[input_aspect, input_tweet], output=probabilities)

    loss = "binary_crossentropy" if classes == 2 else "categorical_crossentropy"
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss=loss)
    return model


def cnn_simple(wv, sent_length, **params):
    model = Sequential()
    model.add(
        embeddings_layer(max_length=sent_length, embeddings=wv, masking=False))

    model.add(Conv1D(activation="relu",
                     filters=80, kernel_size=4, padding="valid"))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer="adam", loss='categorical_crossentropy')
    return model


def cnn_multi_filters(wv, sent_length, nfilters, nb_filters, **kwargs):
    noise = kwargs.get("noise", 0)
    trainable = kwargs.get("trainable", False)
    drop_text_input = kwargs.get("drop_text_input", 0.)
    drop_conv = kwargs.get("drop_conv", 0.)
    activity_l2 = kwargs.get("activity_l2", 0.)

    input_text = Input(shape=(sent_length,), dtype='int32')

    emb_text = embeddings_layer(max_length=sent_length, embeddings=wv,
                                trainable=trainable, masking=False)(input_text)
    emb_text = GaussianNoise(noise)(emb_text)
    emb_text = Dropout(drop_text_input)(emb_text)

    pooling_reps = []
    for i in nfilters:
        feat_maps = Convolution1D(nb_filter=nb_filters,
                                  filter_length=i,
                                  border_mode="valid",
                                  activation="relu",
                                  subsample_length=1)(emb_text)
        pool_vecs = MaxPooling1D(pool_length=2)(feat_maps)
        pool_vecs = Flatten()(pool_vecs)
        # pool_vecs = GlobalMaxPooling1D()(feat_maps)
        pooling_reps.append(pool_vecs)

    representation = concatenate(pooling_reps)

    representation = Dropout(drop_conv)(representation)

    probabilities = Dense(3, activation='softmax',
                          activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=input_text, output=probabilities)
    model.compile(optimizer="adam", loss='categorical_crossentropy')

    return model
