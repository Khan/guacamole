"""Evaluate a history of student responses and predict the accuracy
of subsequent responses.
"""
import itertools
import math
import operator

from accuracy_model import params

# TODO(david): Find out what this actually is
PROBABILITY_FIRST_PROBLEM_CORRECT = 0.8

# Seeded on the mean correct of a sample of 1 million problem logs
# TODO(david): Allow these seeds to be adjusted or passed in, or at
#     least use a more accurate seed (one that corresponds to P(first
#     problem correct)).
EWMA_SEED = 0.9

# We only look at a sliding window of the past problems. This is to minimize
# space requirements as well as allow the user to recover faster.
MAX_HISTORY_KEPT = 20
MAX_HISTORY_BIT_MASK = (1 << MAX_HISTORY_KEPT) - 1


def bit_count(num):
    """
    Calculate the number of bits set to 1 in the binary of a number.
    TODO(david): This uses Kerninghan's method, which would not be very quick
    for dense 1s. Use numpy or some library.
    """
    count = 0
    while num:
        num &= num - 1
        count += 1
    return count


class AccuracyModel(object):
    """
    Predict the probability of the next problem correct using logistic
    regression.
    """

    # Bump this whenever you change the state we keep around so we can
    # reconstitute existing old AccuracyModel objects. Also remember to update
    # the function update_to_new_version accordingly.
    CURRENT_VERSION = 1

    def __init__(self):
        self.version = AccuracyModel.CURRENT_VERSION

        # A bit vector for keeping up to the last 32 problems done
        self.answer_history = 0

        # This is capped at MAX_HISTORY_KEPT
        self.total_done = 0

    def update(self, correct):
        """Update the total done and answer history

        Arguments:
            correct:
                Either a boolean or str or a list of booleans or strs
                representing accuracy of problem. Enter '1' or True if
                the problem is correct.
        """
        if hasattr(correct, '__iter__'):
            # TODO(david): This can definitely be made more efficient.
            for answer in correct:
                self.update(answer or answer == '1')
        else:
            self.total_done = min(self.total_done + 1, MAX_HISTORY_KEPT)
            self.answer_history = \
                ((self.answer_history << 1) | correct) & MAX_HISTORY_BIT_MASK

        return self

    def _get_recent_answer(self, index):
        """Returns either 1 or 0 for the correctness of most recent answer.
        index is 0-based where 0 is the most recent problem done.
        """
        return (self.answer_history >> index) & 1

    def answers(self):
        """Generator for the history in chronological order as bools.

        Note that if > MAX_HISTORY_KEPT problems are done, this will only
        return the last MAX_HISTORY_KEPT.
        """

        if not self.total_done:
            return

        index = 1 << (min(MAX_HISTORY_KEPT, self.total_done) - 1)
        while index:
            yield bool(self.answer_history & index)
            index /= 2  # Like >> bitshift, but turns 1 to 0 as well

    # http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    def exp_moving_avg(self, weight):
        """Return the exponential moving average of the history."""
        ewma = EWMA_SEED

        for i in reversed(xrange(self.total_done)):
            ewma = weight * self._get_recent_answer(i) + (1 - weight) * ewma

        return ewma

    def streak(self):
        """Return the most recent streak of accurate questions.
        """
        for i in xrange(self.total_done):
            if not self._get_recent_answer(i):
                return i

        return self.total_done

    def total_correct(self):
        """Caculate the total number of problems answered correctly."""
        mask = (1 << self.total_done) - 1
        return bit_count(self.answer_history & mask)

    def feature_vector(self):
        """Create a custom feature vector given the current history"""
        ewma_3 = self.exp_moving_avg(0.333)
        ewma_10 = self.exp_moving_avg(0.1)
        current_streak = self.streak()

        if self.total_done == 0:
            log_num_done = 0.0  # avoid log(0)
        else:
            log_num_done = math.log(self.total_done)

        # log (num_missed + 1)
        log_num_missed = math.log(self.total_done - self.total_correct() + 1)

        if self.total_done == 0:
            percent_correct = PROBABILITY_FIRST_PROBLEM_CORRECT
        else:
            percent_correct = float(self.total_correct()) / self.total_done

        features = [
            ewma_3,
            ewma_10,
            current_streak,
            log_num_done,
            log_num_missed,
            percent_correct,
        ]

        return features

    def weight_vector(self):
        """Import weights from a parameter file."""
        # Note: element order must match the ordering of feature_vector
        return [params.EWMA_3,
                params.EWMA_10,
                params.CURRENT_STREAK,
                params.LOG_NUM_DONE,
                params.LOG_NUM_MISSED,
                params.PERCENT_CORRECT]

    def weighted_features(self):
        """Return the combination of the features and weights"""
        return zip(self.feature_vector(), self.weight_vector())

    def predict(self):
        """
        Returns: the probability of the next problem correct using
        logistic regression.
        """
        # We don't try to predict the first problem (no user-exercise history)
        if self.total_done == 0:
            return PROBABILITY_FIRST_PROBLEM_CORRECT

        return AccuracyModel.logistic_regression_predict(
            params.INTERCEPT, self.weight_vector(), self.feature_vector())

    def is_struggling(self, param, minimum_accuracy, minimum_attempts):
        """ Whether or not this model detects that the student is struggling
        based on the history of answers thus far.

        param - This is an exponent which measures how fast we expect students
        to achieve proficiency and get out. The larger the number, the longer
        we allow them to experiment. This is only injected for experimentation
        purposes - it will be internalized later.

        minimum_accuracy - minimum accuracy required for proficiency

        minimum_attempts - minimum problems done before making a judgement
        """

        attempts = self.total_done
        if attempts < minimum_attempts:
            return False

        accuracy_prediction = self.predict()
        if accuracy_prediction >= minimum_accuracy:
            return False

        value = (attempts ** param) * (minimum_accuracy - accuracy_prediction)
        return value > 20.0

    # See http://en.wikipedia.org/wiki/Logistic_regression
    @staticmethod
    def logistic_regression_predict(intercept, weight_vector, X):
        # TODO(david): Use numpy's dot product fn when we support numpy
        dot_product = sum(itertools.imap(operator.mul, weight_vector, X))
        z = dot_product + intercept

        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def simulate(answer_history):
        model = AccuracyModel()
        model.update(answer_history)
        return model.predict()

    # The minimum number of problems correct in a row to be greater
    # than the given threshold
    @staticmethod
    def min_streak_till_threshold(threshold):
        model = AccuracyModel()

        for i in itertools.count(1):
            model.update(correct=True)

            if model.predict() >= threshold:
                return i
