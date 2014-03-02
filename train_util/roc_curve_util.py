"""Bare bones utility script to take an input file or list of input files and
plot ROC curves for each one on a single figure.

Usage:
  cat roc_file | plot_roc_curves.py
    OR
  plot_roc_curves.py *_roc_file

Right now the input files are assumed to be CSV data, with the first column
the correctness on an exercise, and the second column the predicted
probability correct on that exercise.  Each file contains data for a different
curve.

TODO(jace): Maybe take command line args to override the column index
assumption.
"""
import fileinput
import itertools
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np

PLOT_LINES = ["-+", "--D", "-.s", ":*", "-^", "--|", "-._", ":"]


def get_correct_predicted(datapoints, as_string=False):
    """Parse each comma-separated line"""
    if as_string:
        linesplit = re.compile('[, ]')
        datapoints = [linesplit.split(point) for point in datapoints]
    try:
        datapoints = np.asarray(datapoints)
        correct = datapoints[:, 0].astype('float')
        predicted = datapoints[:, 1].astype('float')
    except IndexError:
        # deal with the case where the last row has the wrong number
        # of columns -- eg, if you are looking at a csv file as it's
        # being written
        datapoints = datapoints[:-1]
        datapoints = np.asarray(datapoints)
        correct = datapoints[:, 0].astype('float')
        predicted = datapoints[:, 1].astype('float')

    return correct, predicted


def calc_roc_curve(correct, predicted):
    """Calculate true positive and negative values for various cutoffs"""
    thresholds = np.arange(-0.01, 1.02, 0.01)
    true_pos = np.zeros(thresholds.shape)
    true_neg = np.zeros(thresholds.shape)
    tot_true = np.max([np.float(np.sum(correct)), 1])
    tot_false = np.max([np.float(np.sum(np.logical_not(correct))), 1])

    for i in range(thresholds.shape[0]):
        threshold = thresholds[i]
        pred1 = predicted >= threshold
        pred0 = predicted < threshold
        if np.sum(tot_true) > 0:
            true_pos[i] = np.sum(correct[pred1]) / tot_true
        if np.sum(tot_false) > 0:
            true_neg[i] = np.sum(np.logical_not(correct[pred0])) / tot_false

    return true_pos, true_neg


def draw_roc_curve(name, lines, as_string=False):
    """Plot each point along a roc curve on a pyplot window"""
    line_cycler = itertools.cycle(PLOT_LINES)
    correct, predicted = get_correct_predicted(lines, as_string)
    true_pos, true_neg = calc_roc_curve(correct, predicted)

    # grab the base of the filename
    name = name.split('/')[-1].split('.')[0]

    if name.startswith('_'):
        warnings.warn("Warning.  If name starts with an underscore, "
                      "the label won't display.")

    plt.plot(1 - true_neg,
             true_pos,
             next(line_cycler),
             label=name)


def add_roc_labels():
    """Have pyplot show the correct labels"""
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()


def main():
    """Read in files and display them on a pyplot window"""
    plt.figure(1)

    lines = []
    filename = None
    for line in fileinput.input():
        if not filename:
            filename = fileinput.filename()
        if fileinput.isfirstline() and len(lines):
            draw_roc_curve(filename, lines, as_string=True)
            filename = fileinput.filename()
            lines = []
        lines.append(line)

    draw_roc_curve(fileinput.filename(), lines)

    add_roc_labels
    plt.show()

if __name__ == '__main__':
    main()
