"""A variety of utilities to visualize the results from item response theory
training
"""
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import mirt_util
from train_util import roc_curve_util


def show_roc(predictions):
    """Display an roc curve given a predictions and classifications

    Predictions is a dictionary with keys consisting of model names,
    and values consisting of predictions made by those models.
    """
    plt.figure(1)
    for model, classifications in predictions.iteritems():
        roc_curve_util.draw_roc_curve(model, classifications)
    roc_curve_util.add_roc_labels()
    plt.show()


def show_exercises(parameter_file):
    """Display a sigmoid for each exercise."""
    plt.figure(1)
    data = mirt_util.json_to_data(parameter_file)
    parameters = data['params']
    exercise_ind_dict = parameters.exercise_ind_dict

    def eval_conditional_probability(x, parameters, exercise_ind):
        """Evaluate the conditional probability of answering each question
        accurately for a student with ability x
        """
        return mirt_util.conditional_probability_correct(
            np.ones((parameters.num_abilities, 1)) * x,
            parameters,
            exercise_ind)

    abilities_to_plot = np.arange(-3, 3, .01)
    exercises, indices = exercise_ind_dict.keys(), exercise_ind_dict.values()
    exercise_plots = defaultdict(list)
    for ability in abilities_to_plot:
        conditional_probs = eval_conditional_probability(
            ability,
            parameters,
            exercise_ind_dict.values())
        for exercise in exercises:
            exercise_plots[exercise].append(conditional_probs[
                exercises.index(exercise)])
    for exercise in exercises:
        plt.plot(abilities_to_plot, exercise_plots[exercise], label=exercise)
    plt.xlabel('Student Ability')
    plt.ylabel('P(Answer Correctly)')
    plt.title('Two parameter IRT model')
    plt.legend(loc='best', prop={'size': 6})
    plt.show()


def print_report(parameter_file):
    """Print interpretable results given a json file"""
    data = mirt_util.json_to_data(parameter_file)
    parameters = data['params']
    print 'Generating Report for %s' % parameter_file
    print "%50s\t%s\t\t" % ('Exercise', 'Bias'),
    for i in range(parameters.num_abilities):
        print 'Dim. %s\t' % (i + 1),
    print
    exercises = parameters.exercise_ind_dict.keys()
    exercises_to_parameters = [(ex, parameters.get_params_for_exercise(ex))
                               for ex in exercises]
    # Sort by the difficulty bias
    exercises_to_parameters.sort(key=lambda x: x[-1][-1])
    for ex, param in exercises_to_parameters:
        print "%50s\t%.4f\t" % (ex, param[-1]),
        for p in param[:-1]:
            print "\t%.4f\t" % p,
        print
