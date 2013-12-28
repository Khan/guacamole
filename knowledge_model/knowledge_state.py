from collections import defaultdict
import cPickle
import math
import numpy
import os

from accuracy_model import model

RANDOM_FEATURE_LENGTH = 100
PROFICIENCY_THRESHOLD = .91

# _PARAMS is lazily loaded when required.
# knowledge_params.pickle is auto-generated from the model training process,
# currently in the analytics repository at src/accuracy_model_train.py.
# TODO(eliana) - Actually auto-generate this from the model training process.
#               (Complete with cPickle.HIGHEST PROTOCOL, which is missing)
# TODO(eliana) - use instance_cache for locking in case we go multi-thread
# TODO(eliana) - maybe find a way to only load part of this? We are often
#                looking at all of it at once though, so maybe not.
_PARAMS = None


def thetas(key):
    """Loads knowledge params if necessary and finds the key in theta"""
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = cPickle.load(open(
            os.path.join('exercises', 'knowledge_params.pickle'), 'r'))
    return _PARAMS['thetas'].get(key, None)


def components(key):
    """Loads knowledge params if necessary and finds the key in components"""
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = cPickle.load(open(
            os.path.join('exercises', 'knowledge_params.pickle'), 'r'))
    return _PARAMS['components'].get(key, None)


def sigmoid(x):
    x = min(max(x, -100.0), 100.0)  # clamp for sanity
    return 1.0 / (1.0 + math.exp(-x))


class KnowledgeState(object):
    """Handles state-maintenance and computation for knowledge models.

    All state data is kept in a single dictionary.
    """

    def __init__(self):
        self.data = {}
        # # note the following gets put in self.data through an @property
        # self.random_features = numpy.zeros((RANDOM_FEATURE_LENGTH, 1))
        self.accuracy_models = defaultdict(model.AccuracyModel)

    @property
    def random_features(self):
        """Return the 'random features' that capture past exercise history."""
        return self.data['random_features']

    def update(self, exercise, problem_type, problem_number, correct):
        """Compute updated feature values given a new problem attempt."""
        if problem_number != 1:
            return

        rand_component_key = (exercise, problem_type, correct)

        if components(rand_component_key) is None:
            # This exercise/ptype combo wasn't present during model training,
            # so we can't update the random features.
            return

        self.random_features += components(rand_component_key)

    def predict(self, exercise):
        """Predict the probability of getting the next problem correct."""

        model = self.accuracy_models[exercise]
        if thetas(exercise) is None:
            # No model is found for this exercise.  This can happen if this
            # is a new exercise and/or the model parameter file has not been
            # updated recently.
            return None

        # temporarily create a new array that is the concatentation of the
        # random feature vector, the baseline model features, and a 1 for bias.
        features = [1.0]
        features += model.feature_vector()
        features += list(self.random_features)

        # convert to numpy
        features = numpy.array(features)

        return sigmoid(numpy.dot(features, thetas(exercise)))

    def proficient(self, exercise, problem_number,
                   topic_mode, num_missed):
        """Return a boolean for whether this user should be proficient."""
        return problem_number <= 5 and topic_mode and num_missed <= 1 and (
                self.predict(exercise) > PROFICIENCY_THRESHOLD)
