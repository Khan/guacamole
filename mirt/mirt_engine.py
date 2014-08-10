"""Extend the Engine model to provide a adaptive pretest engine."""
import numpy as np

import engine
import mirt_util


class MIRTEngine(engine.Engine):

    # ===== BEGIN: Engine interface implementation =====
    def __init__(self, model_data):
        """
        Args:
            model_data: Either a rich object containing the actual model
                parameters, or a file name to points to an .npz file (for
                offline use only).
            contextual_exercises: a list of the exercises names that the
                client would like the engine to choose questions from.
        """
        self.theta = model_data['params']
        self.exercise_ind_dict = self.theta.exercise_ind_dict

        self.num_abilities = self.theta.num_abilities
        self.abilities = np.zeros((self.num_abilities, 1))
        self.abilities_stdev = np.zeros((self.num_abilities, 1))

        self.max_length = model_data['max_length']
        self.max_time_taken = model_data['max_time_taken']

    def eligible_exercises(self, history):
        """An exercise may be seen at most once"""
        # Ensure we are only choosing from exercises the user has not seen
        seen_exs = set(h['exercise'] for h in history)
        return [e for e in self.exercises() if e not in seen_exs]

    def next_suggested_item(self, history):
        """Return an ItemSuggestion for this Engine's preferred next item."""
        # update ability estimates only once -- outside the loop
        self._update_abilities(history)

        max_info = float("-inf")
        max_info_ex = None
        for exercise in self.eligible_exercises(history):
            fisher_information = self.fisher_information(history, exercise)
            if fisher_information > max_info:
                max_info = fisher_information
                max_info_ex = exercise
        exercise = max_info_ex

        return engine.ItemSuggestion(exercise)

    def estimated_exercise_accuracies(self, history):
        """Returns a dictionary where the keys are all the exercise names
        known by the engine to be in the domain.
        """
        history = [h for h in history if h['exercise'] in self.exercises()]
        # for efficiency update ability estimates only once -- outside the loop
        self._update_abilities(history)

        return {ex: self.estimated_exercise_accuracy(history, ex, False)
                for ex in self.exercises()}

    def estimated_exercise_accuracy(self, history, exercise_name,
            update_abilities=True, ignore_analytics=False):
        """Returns the expected probability of getting a future question
        correct on the specified exercise.
        """
        if update_abilities:
            self._update_abilities(history, ignore_analytics=ignore_analytics)
        try:
            exercise_ind = mirt_util.get_exercise_ind(
                exercise_name, self.exercise_ind_dict)
        except KeyError:
            # If we don't have this exercise, predict the mean predicted
            # accuracy over all exercises we do have.
            return self.score(history)
        return mirt_util.conditional_probability_correct(
            self.abilities, self.theta, exercise_ind)[0]

    def score(self, history):
        """Returns a float that is the overall score on this assessment.
        Caller beware: may not be useful of valid is the assessment if the
        assessment has not been fully completed.  Check if is_complete().
        """
        # use lots of steps when estimating score to make
        # the score seem close to deterministic
        self._update_abilities(history, num_steps=2000)
        predicted_accuracies = np.asarray([
            self.estimated_exercise_accuracy(history, ex, False)
            for ex in self.exercises()], dtype=float)

        return np.mean(predicted_accuracies)

    def readable_score(self, history):
        """Return the score formatted as a percent"""
        score = self.score(history)
        return str(int(score * 100.0))

    def progress(self, history):
        """Return the portion of the problems completed."""
        return min(float(len(history)) / self.max_length, 1.0)

    # ===== END: Engine interface implementation =====
    def fisher_information(self, history, exercise_name):
        """Compute Fisher information for exercise at current ability."""
        p = self.estimated_exercise_accuracy(history, exercise_name, False)

        # "discrimination" parameter for this exercise.  Note this
        # implementation is only valid for the single dimensional case.
        a = self.theta.W_correct[self.exercise_ind_dict[exercise_name], :-1]

        # TODO(jascha) double check this formula for the multidimensional case
        fisher_info = np.sum(a ** 2) * p * (1. - p)

        return fisher_info

    def exercises(self):
        """Return the slug of all exercises as a list"""
        return self.exercise_ind_dict.keys()

    def is_complete(self, history):
        """Return True when the exercise is complete.

        This happens when it has reached its max length or has run out of
        eligible items
        """
        if super(MIRTEngine, self).is_complete(history):
            return True
        else:
            # The assessment is complete when there are no more possible
            # exercises
            if len(self.eligible_exercises(history)) == 0:
                return True
        return False

    def _update_abilities(self, history, use_mean=True, num_steps=200,
                          ignore_analytics=False):
        """Take the history and update the ability estimate."""
        # TODO(jace) - check to see if history has actually changed
        # to avoid needless re-estimation
        # If ignore_analytics is true, only learn from non-analytics cards
        # This is to evaluate the quality of various models for predicting
        # the analytics card.

        state = mirt_util.UserState()

        if history and ignore_analytics:
            history = [
                h for h in history if h['metadata'] and
                not h['metadata'].get('analytics')]
        ex = lambda h: engine.ItemResponse(h).exercise
        exercises = np.asarray([ex(h) for h in history])
        state.exercise_ind = mirt_util.get_exercise_ind(
            exercises, self.exercise_ind_dict)

        is_correct = lambda h: engine.ItemResponse(h).correct
        state.correct = np.asarray(
            [is_correct(h) for h in history]).astype(int)

        time_taken = lambda h: engine.ItemResponse(h).time_taken
        time_taken = np.asarray([time_taken(h) for h in history]).astype(float)
        state.log_time_taken = mirt_util.get_normalized_time(time_taken)

        state.abilities = self.abilities

        sample_abilities, _, mean_abilities, stdev = (
            mirt_util.sample_abilities_diffusion(
                self.theta, state, num_steps=num_steps))

        self.abilities = mean_abilities  # if use_mean else sample_abilities
        self.abilities_stdev = stdev
