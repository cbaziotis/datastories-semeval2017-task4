from operator import itemgetter

import numpy as np


def get_class_coefficients(estimator):
    class_dict = {cls: coefs for cls, coefs in
                  zip(estimator._final_estimator.classes_, estimator._final_estimator.coef_)}
    return class_dict


def get_features_coefficients(estimator, vectorizer, feature_selector, ):
    if feature_selector:
        features = np.asarray(vectorizer.get_feature_names())[feature_selector.get_support()]
    else:
        features = np.asarray(vectorizer.get_feature_names())

    features_coefficients = {
        f: {cls: coef[i] for cls, coef in zip(estimator._final_estimator.classes_, estimator._final_estimator.coef_)}
        for i, f in enumerate(features)}

    features_coefficients_order = {k: sorted(v, key=v.get, reverse=True) for k, v in features_coefficients.items()}

    return features_coefficients, features_coefficients_order


def classifiers_most_informative_features(vectorizer, clf, n=20):
    """
    Prints the n most informative features
    :param vectorizer:
    :param clf:
    :param n:
    :return:
    """
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%20s\t\t\t\t%.4f\t%20s" % (coef_1, fn_1, coef_2, fn_2))


def most_discriminative_features(vectorizer, selector):
    """
    Returns the n most discriminative features from feature selection step
    """
    feature_mask = selector.get_support()
    features = {name: score for name, score in
                zip(np.asarray(vectorizer.get_feature_names())[feature_mask],
                    np.asarray(selector.scores_)[feature_mask])}
    return features


def most_informative_features(model, vectorizer, feature_selector, classifier, text=None, n=20):
    """
    Accepts a Pipeline with a classifier and a Vectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.

    Note that this function will only work on linear models with coefs_
    """

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn = zip(coefs[:n], coefs[:-(n + 1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)
