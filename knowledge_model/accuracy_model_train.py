"""Train an accuracy model on feature data read from stdin.

For more detail on the expected format of the input data, see FeaturesetFields
below, or accuracy_model_featureset.py.
"""

import optparse
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import accuracy_model_util
import regression_util

# Minimum number of data samples required to fit a model.  Exercises which
# do not have at least this many problems attempted will not have parameters,
# and call to predict() will end up emitting None in production.
MIN_SAMPLES = 5000
# Max number of data samples used to fit a model.  We can cap it if we want
# faster training and are confident that MAX_SAMPLES will be enough data to
# avoid overfitting.
MAX_SAMPLES = 50000
# The amount of data witheld for testing.  This can be an integer number of
# samples, or a fraction (expressed as a decimal between 0.0 and 1.0).  Note
# that the choice of a fixed size will affect analysis of model performance
# that averages across exercises by either equal-weighting performance of
# each exercise or weighthing proportional to frequency of attempts.
TEST_SIZE = 500


class FeaturesetFields:
    """Specify the column indices of specific data in the input file."""
    exercise = 0
    correct = 1
    baseline_prediction = 2
    num_previous_exs = 3
    problem_number = 4

    # The following feature types may or may not be present in the input data,
    # but if they are, these are the column slices where they should be.

    # The output/prediction of webapp:exercises/accuracy_model.py
    features_baseline = slice(2, 3)
    # The internal features of webapp:exercises/accuracy_model.py
    features_custom = slice(6, 12)
    # "Random" features.
    features_random = slice(12, None)
    # MIRT features
    features_mirt = slice(12, None)


idx = FeaturesetFields()


class Dataset:
    def __init__(self, correct, baseline_prediction, features):
        self.correct = correct
        self.baseline_prediction = baseline_prediction
        self.features = features


def roc_curve(correct, prediction_prob):
    thresholds = np.arange(-0.01, 1.02, 0.01)
    true_pos = np.zeros(thresholds.shape)
    true_neg = np.zeros(thresholds.shape)
    tot_true = np.max([np.float(np.sum(correct)), 1])
    tot_false = np.max([np.float(np.sum(np.logical_not(correct))), 1])

    for i in range(thresholds.shape[0]):
        threshold = thresholds[i]
        pred1 = prediction_prob >= threshold
        pred0 = prediction_prob < threshold
        if np.sum(tot_true) > 0:
            true_pos[i] = np.sum(correct[pred1]) / tot_true
        if np.sum(tot_false) > 0:
            true_neg[i] = np.sum(np.logical_not(correct[pred0])) / tot_false

    return {"thresholds": thresholds,
            "true_pos": true_pos,
            "true_neg": true_neg}


def print_roc_curve(roc_curve):
    num_points = len(roc_curve['thresholds'])
    for t in range(num_points):
        print "rocline,",  # a known line prefix useful for grepping results
        print "%s," % roc_curve['thresholds'][t],
        print "%s," % roc_curve['true_pos'][t],
        print "%s" % roc_curve['true_neg'][t]


def quantile(x, q):
    if len(x.shape) != 1:
        return None
    x = x.tolist()
    x.sort()
    return x[int((len(x) - 1) * float(q))]


def quantiles(x, quantiles):
    return [quantile(x, q) for q in quantiles]


def preprocess_data(lines, options):
    # this step is critical- currently the input is sorted not only on exercise
    # but also subsequent fields, which could introduce a ton of bias
    np.random.shuffle(lines)

    # TODO(jace): extract columns individually; avoid string column
    # turn it into a numpy array
    lines = np.asarray(lines)
    print >> sys.stderr, "Preprocessing, shape = " + str(lines.shape)

    # and split it up into the different components
    correct = lines[:, idx.correct].astype('int')
    baseline_prediction = lines[:, idx.baseline_prediction].astype('float')

    # Start features with a bias vector, unless we were told not to.
    use_ones = 0 if options.no_bias else 1
    features = np.ones((correct.shape[0], use_ones))

    # and add additional features as specified by command line args
    feature_list = options.feature_list.split(",")

    def append_features(slice_obj):
        return np.append(features, lines[:, slice_obj].astype('float'), axis=1)

    if 'baseline' in feature_list:
        features = append_features(idx.features_baseline)
    if 'custom' in feature_list:
        features = append_features(idx.features_custom)
    if 'random' in feature_list:
        features = append_features(idx.features_random)
    if 'mirt' in feature_list:
        features = append_features(idx.features_mirt)

    print >> sys.stderr, "Computing for %s, " % lines[0, 0],
    print >> sys.stderr, "feature shape = %s" % str(features.shape)

    # clean up any NaN.... TODO(jace) - is this necessary?
    correct = np.nan_to_num(correct)
    features = np.nan_to_num(features)
    baseline_prediction = np.nan_to_num(baseline_prediction)

    N = lines.shape[0]
    if TEST_SIZE < 1.0:
        training_cutoff = int(N * (1 - TEST_SIZE))
    else:
        training_cutoff = N - TEST_SIZE

    def dataset(start_index, end_index):
        return Dataset(
                correct[start_index:end_index],
                baseline_prediction[start_index:end_index],
                features[start_index:end_index, :])

    data_train = dataset(0, training_cutoff)
    data_test = dataset(training_cutoff, N)

    return data_train, data_test


def fit_logistic_log_regression(lines, options):

    data_train, data_test = preprocess_data(lines, options)

    # do parameter estimation
    theta = regression_util.logistic_log_regression(
            data_train.features, data_train.correct)

    prediction = regression_util.sigmoid(np.dot(data_test.features, theta))

    print >> sys.stderr, "mean correct = %.3f, mean predict = %.3f" % (
            np.mean(data_test.correct), np.mean(prediction))
    return {"theta": theta,
            "labels": data_test.correct,
            "predictions": prediction,
            "ROC": roc_curve(data_test.correct, prediction)}


def fit_random_forest(lines, options):

    data_train, data_test = preprocess_data(lines, options)

    rf = RandomForestClassifier(n_estimators=100, min_samples_split=2)
    rf.fit(data_train.features, data_train.correct)

    prediction = rf.predict_proba(data_test.features)
    prediction = prediction[:, 1]  # probability correct

    print >> sys.stderr, "mean correct = %.3f, mean predict = %.3f" % (
            np.mean(data_test.correct), np.mean(prediction))
    return {"theta": None,
            "labels": data_test.correct,
            "predictions": prediction,
            "ROC": roc_curve(data_test.correct, prediction)}


def fit_model(models, model_key, lines, options):
    print >> sys.stderr, "Model_key " + model_key
    if len(lines) >= MIN_SAMPLES:

        if len(lines) >= MAX_SAMPLES:
            lines = lines[:MAX_SAMPLES]

        # TODO(jace): why is this check necessary?
        if 'representing_numbers' in model_key or (
                'rotation_of_polygons' in model_key):
            return

        if options.classifier == 'logistic_log':
            model = fit_logistic_log_regression(lines, options)
        elif options.classifier == 'random_forest':
            model = fit_random_forest(lines, options)
        else:
            sys.exit("ERROR: unsupported classifier '%s'" % options.classifier)

        models[model_key] = model
    else:
        print >> sys.stderr, "Insufficient data points (%d) for %s" % (
                len(lines), model_key)


def summarize_models(models):
    """Print a distribution summary and ROC curve for each model."""
    labels = None
    predictions = None
    for model_key in sorted(models.keys()):
        model = models[model_key]
        if labels is None:
            labels = model['labels']
            predictions = model['predictions']
        else:
            labels = np.concatenate((labels, model['labels']))
            predictions = np.concatenate((predictions, model['predictions']))

        # print some information about the range/distribution of predictions
        quants = quantiles(model['predictions'], [0.0, 0.1, 0.5, 0.9, 1.0])
        print "PREDICT_DIST,%s," % model_key,
        print ",".join([str(q) for q in quants])

    print_roc_curve(roc_curve(labels, predictions))


def output_models(models, options):
    """Output a pickled file containing all the trained models."""
    assert options.rand_comp_file, "The rand_comp_file arg must be provided."
    with open(options.rand_comp_file, 'r') as infile:
        random_components = pickle.load(infile)

    model_thetas = {k: v['theta'] for k, v in models.iteritems()}

    output = {"components": random_components, "thetas": model_thetas}
    assert options.output_model_file, "Specify output model file."
    with open(options.output_model_file, 'w') as outfile:
        pickle.dump(output, outfile)


def get_cmd_line_options():
    parser = optparse.OptionParser()

    parser.add_option("-c", "--classifier", default='logistic_log',
            help="Type of classifier.  'logistic_log' or 'random_forest'.")
    parser.add_option("-f", "--feature_list", default='baseline',
            help="Comma seprated list to feature types to use. Choose from "
                 "baseline, custom, random, and mirt. If you want to run "
                 "with nothing but a bias feature, use 'none'.")
    parser.add_option("-n", "--no_bias", dest="no_bias", action="store_true",
            help="Whether to omit using a bias in the feature set.  By "
                 "default a bias unit is included.")
    parser.add_option("-r", "--rand_comp_file",
            help="Name of a file to optionally write the random components "
                 "to.")
    parser.add_option("-o", "--output_model_file",
            help="Name of a file you optionally want to write a pickeled "
                 "file of all the models to.")
    parser.add_option("-t", "--topic",
            help="If given, then models are not fit per exercise; rather "
                 "all exercises are treated as part of one 'topic' and one "
                 "model is fit for all of the data.")

    options, _ = parser.parse_args()
    return options


def main():
    options = get_cmd_line_options()

    models = {}

    prev_key = None
    lines = []
    for line in sys.stdin:

        row = accuracy_model_util.linesplit.split(line.strip())

        model_key = options.topic if options.topic else row[idx.exercise]

        if model_key != prev_key and prev_key:
            fit_model(models, prev_key, lines, options)
            lines = []

        prev_key = model_key

        # NOTE: if you care to filter for say, heavy or light users, perhaps
        # to analyze if there is loss of information from collisions in
        # random features, you can filter the samples based the the number
        # of distinct exercises that came before.  For production, we use
        # everything (by taking num_previous_exercises >= 0)
        if int(row[idx.num_previous_exs]) >= 0:
            lines.append(row)

    # one last time for the final model_key
    fit_model(models, model_key, lines, options)

    summarize_models(models)

    if options.output_model_file:
        output_models(models, options)

if __name__ == '__main__':
    main()

