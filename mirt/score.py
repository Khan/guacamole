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
import mirt.engine
import mirt.mirt_engine
import mirt.mirt_util
from train_util.model_training_util import FieldIndexer


class ScoreEngine(object):
    """An engine score a student on a an interactive test"""

    def __init__(self, test_engine):
        self.engine = test_engine
        self.history = []

    def update_history(self, history):
        history.extend(history)

    def print_score(self):
        """Print the current score for the current history"""
        print self.engine.score(self.history)


def get_student_responses(students_filepath, data_format='simple'):
    """Given a set of student histories in a file, convert to item responses

    The students file should be indexable by an indexer.
    """
    history = []
    current_user = ''
    indexer = FieldIndexer.get_for_slug(data_format)
    with open(students_filepath, 'r') as outfile:
        for line in outfile:
            line = line.split(',')
            user = line[indexer.user]
            correct = line[indexer.correct]
            exercise = line[indexer.exercise]
            if history and user != current_user:
                yield user, history
                current_user = user
                history = []
            response = mirt.engine.ItemResponse.new(
                correct=correct, exercise=exercise)
            history.append(response)


def score_students(model_file, students_filepath):
    """Prints the estimated score for each student in the data file

    Arguments:
        Takes the parameter file name for the mirt json parameter
        file of interest.

        Is interactive at the command line with options to enter
        various questions and see what the resulting question asked
        will be.
    """
    data = mirt.mirt_util.json_to_data(model_file)
    students = get_student_responses(students_filepath)
    for name, history in students:
        print name,
        engine = ScoreEngine(mirt.mirt_engine.MIRTEngine(data))
        engine.update_history(history)
        engine.print_score()
