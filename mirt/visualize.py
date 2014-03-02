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
    print abilities_to_plot.size
    print np.array(exercise_plots.values())[0].size
    for exercise in exercises:
        plt.plot(abilities_to_plot, exercise_plots[exercise], label=exercise)
    plt.show()
