#!/usr/bin/env python
"""An interactive utility to explore the workings of your IRT model.

Usage:
./adaptive_pretest.py /path/to/model.json

Then enter 1's and 0's
according to whether the student answered correctly, and optionally the amount
of time it took them to respond. Enter numbers representing how long the
problem took, if you choose. The output will include the model's estimation of
the student's accuracy on each of the assessment items as well as their overall
score ()
"""
import json
import sys

import mirt.engine
import mirt.mirt_engine
import mirt.mirt_util


def interactive_test(test_engine):
    """A simple command line interface to mirt parameters."""

    history = []
    use_time = raw_input("Use time as a feature? [y/n]: ")
    # Accept any answer starting with y or Y as a yes
    time = use_time[0] in ['y', 'Y']

    while not test_engine.is_complete(history):
        exercise = test_engine.next_suggested_item(history).item_id
        print "\nQuestion #%d, Exercise type: %s" % (len(history), exercise)
        correct = int(raw_input("Enter 1 for correct, 0 for incorrect: "))
        correct = correct if correct == 1 else 0
        if time:
            print "How much time did it take?"
            time = float(raw_input("Enter time in seconds: "))
        response = mirt.engine.ItemResponse.new(
            correct=correct, exercise=exercise)
        history.append(response.data)
        print "Current score is now %.4f (stdev=%.4f." % (
            test_engine.score(history), test_engine.abilities_stdev)
        print "Progress is now %.4f." % test_engine.progress(history)

    print json.dumps(
        test_engine.estimated_exercise_accuracies(history), indent=4)
    print test_engine.abilities
    print test_engine.score(history)


def main(model_file):
    """Starts an interactive session with a given parameter student

    Arguments:
        Takes the parameter file name for the mirt json parameter
        file of interest.

        Is interactive at the command line with options to enter
        various questions and see what the resulting question asked
        will be.
    """
    data = mirt.mirt_util.json_to_data(model_file)
    parameters = data['params']
    interactive_test(mirt.mirt_engine.MIRTEngine(parameters))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        exit("Usage: python %s json_model_file" % sys.argv[0])
