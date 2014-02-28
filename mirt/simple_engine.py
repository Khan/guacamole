"""A very simple engine that randomly suggests problems from exercise_ids"""

import random

from mirt import engine


class SimpleEngine(engine.Engine):
    """A very simple engine that always randomly chooses a question from all
    possible questions.
    """

    def __init__(self, model_data):
        super(SimpleEngine, self).__init__(model_data)
        self.model = model_data
        self.max_length = self.model['max_length']
        self.exercise_ids = self.model['exercise_ids']

    def next_suggested_item(self, history):
        """Randomly choose an item from all item ids."""
        # first argument is the item type, which in simple is always exercise
        ex = engine.ItemSuggestion(random.choice(self.exercise_ids))
        return ex

    def score(self, history):
        """Calculate percent correct"""
        if not history or not len(history):
            return 0.0

        correct = lambda ir: engine.ItemResponse(ir).correct
        total_correct = sum(correct(response) for response in history)

        return float(total_correct) / len(history)

    def readable_score(self, history):
        """Format percent correct as a string"""
        score = self.score(history)
        return format(score, ".0%")

    def progress(self, history):
        """Calculate fraction of assessment completed."""
        return min(float(len(history)) / self.max_length, 1.0)

    def estimated_exercise_accuracy(self, history, exercise_name):
        """The simple model does not estimate exercise accuracies."""
        return None

    def estimated_exercise_accuracies(self, history):
        """The simple model does not estimate exercise accuracies."""
        return None

    @staticmethod
    def validate_params(params):

        if ('exercise_ids' not in params or
                'max_length' not in params):
            raise engine.InvalidEngineParamsError()

        return params
