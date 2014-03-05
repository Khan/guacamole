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


class TestEngine(object):
    """An engine to administer an interactive test"""

    def __init__(self, test_engine):
        self.engine = test_engine
        self.history = []

    def print_current_score(self):
        """Print the current score for the current history"""
        print "Current score is now %.4f." % (
            self.engine.score(self.history))
        if len(self.engine.abilities) > 1:
            print "(mean std=%.4f, max std=%.4f)." % (
                min(self.engine.abilities_stdev),
                max(self.engine.abilities_stdev))

    def interactive_test(self, num_exercises=5):
        """A simple command line interface to mirt parameters."""

        use_time = False  # raw_input("Use time as a feature? [y/N]: ")
        # Accept any answer starting with y or Y as a yes
        time = use_time and use_time[0] in ['y', 'Y']

        while (not self.engine.is_complete(self.history)
               and len(self.history) < num_exercises):
            exercise = self.engine.next_suggested_item(self.history).item_id
            print "\nQuestion #%d, Exercise type: %s" % (
                len(self.history), exercise)
            correct = int(raw_input("Enter 1 for correct, 0 for incorrect: "))
            correct = correct if correct == 1 else 0
            if time:
                print "How much time did it take?"
                time = float(raw_input("Enter time in seconds: "))

            response = mirt.engine.ItemResponse.new(
                correct=correct, exercise=exercise)

            self.history.append(response.data)
            self.print_current_score()

            print "Progress is now %.4f." % self.engine.progress(self.history)

    def print_outcome(self):
        """Print the status of a current test that's been taken"""
        print 'Estimated Exercise Accuracies:'
        print json.dumps(
            self.engine.estimated_exercise_accuracies(self.history), indent=4)
        print 'Score:'
        print self.engine.score(self.history)


def main(model_file, num_exercises):
    """Starts an interactive session with a given parameter student

    Arguments:
        Takes the parameter file name for the mirt json parameter
        file of interest.

        Is interactive at the command line with options to enter
        various questions and see what the resulting question asked
        will be.
    """
    data = mirt.mirt_util.json_to_data(model_file)
    engine = TestEngine(mirt.mirt_engine.MIRTEngine(data))
    engine.interactive_test(num_exercises=num_exercises)
    engine.print_outcome()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        exit("Usage: python %s json_model_file" % sys.argv[0])
