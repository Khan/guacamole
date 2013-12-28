"""
Loose-bounded tests for the accuracy model - sanity tests as well as
tests for correct behavior expected of an accuracy model. Tries not to be
flaky.
"""

import unittest

from accuracy_model.model import AccuracyModel


class TestSequenceFunctions(unittest.TestCase):

    @staticmethod
    def to_bool_generator(str_sequence):
        return ((l == '1') for l in str_sequence)

    @staticmethod
    def model_from_str(str_sequence):
        model = AccuracyModel()
        model.update(TestSequenceFunctions.to_bool_generator(str_sequence))
        return model

    def is_struggling(self,
                      str_sequence,
                      minimum_accuracy=0.94,
                      minimum_attempts=5):
        model = TestSequenceFunctions.model_from_str(str_sequence)
        return model.is_struggling(
                param=1.8,
                minimum_accuracy=minimum_accuracy,
                minimum_attempts=minimum_attempts)

    def sim(self, str_sequence):
        """What is the predicted accuracy after simulating this sequence?"""
        gen = TestSequenceFunctions.to_bool_generator(str_sequence)
        return AccuracyModel.simulate(gen)

    def lt(self, seq_smaller, seq_larger):
        """Is the prediction for the first seq less than for the second?"""
        self.assertTrue(self.sim(seq_smaller) < self.sim(seq_larger))

    def test_bounded_in_0_1_interval(self):
        """Accuracy should never be under 0 or above 1"""
        sequences = ['', '1', '0', '000', '111', '1' * 100, '0' * 5000,
                     '1' * 5000, '1101111110111']
        for seq in sequences:
            self.assertTrue(0 <= self.sim(seq) <= 1.0)

    def test_more_is_stronger(self):
        """Longer streaks should give the model greater confidence, manifesting
        as more extreme predictions"""
        self.assertTrue(self.sim('') < self.sim('1') < self.sim('11') <
                        self.sim('111') < self.sim('1111'))

        self.assertTrue(self.sim('') > self.sim('0') > self.sim('00') >
                        self.sim('000') > self.sim('0000'))

    def test_weigh_recent_more(self):
        """Should weigh more recent answers more"""
        self.lt('0', '1')
        self.lt('10', '01')
        self.lt('110', '101')
        self.lt('010', '001')
        self.lt('1010', '0101')

    def test_sanity(self):
        # Some loose general bounds (very unscientific)... at least if these
        # are violated, we know something fishy's going on.
        self.assertTrue(self.sim('1' * 20) > 0.95)
        self.assertTrue(self.sim('0' * 20) < 0.8)
        self.assertTrue(0.3 < self.sim('0') < 0.8)
        self.assertTrue(0.4 < self.sim('') < 0.9)
        self.assertTrue(0.5 < self.sim('1') < 1.0)

    def test_expected_behavior(self):
        """After answering a question incorrectly, prediction is lower.
        After answering correctly, it is higher.
        """
        self.lt('11110', '1111')
        self.lt('11110', '111101')
        self.lt('0000', '00001')
        self.lt('0', '01')
        self.lt('10', '1')

    def test_struggling_minimum_questions(self):
        self.assertFalse(self.is_struggling('0', minimum_attempts=5))
        self.assertFalse(self.is_struggling('0' * 4, minimum_attempts=5))
        self.assertFalse(self.is_struggling('0' * 99, minimum_attempts=100))

    def test_struggling_minimum_accuracy(self):
        def assert_minimum_accuracy_met(s):
            self.assertFalse(self.is_struggling(s,
                                                minimum_accuracy=self.sim(s)))

        assert_minimum_accuracy_met('01011')
        assert_minimum_accuracy_met('0101010111')
        assert_minimum_accuracy_met('101010100110')
        assert_minimum_accuracy_met('11010001001100')
        assert_minimum_accuracy_met('101011ll000101001100')

    def test_struggling_sanity(self):
        # all correct -> not struggling
        for i in range(50):
            self.assertFalse(self.is_struggling('1' * i))

        # a heck of a lot incorrect -> struggling
        for i in range(20, 50):
            self.assertTrue(self.is_struggling('0' * i))

        # a heck of a lot of no progress -> struggling
        for i in range(10, 20):
            self.assertTrue(self.is_struggling('10' * i),
                            msg="Should be struggling on %s" % ('10' * i))
        for i in range(5, 10):
            self.assertTrue(self.is_struggling('110' * i),
                            msg="Should be struggling on %s" % ('110' * i))

    def test_conversion_of_history_to_bools(self):
        three_right_two_wrong = self.model_from_str('11100')
        self.assertEquals([True, True, True, False, False],
                          list(three_right_two_wrong.answers()))

        alternating = self.model_from_str('0101')
        self.assertEquals([False, True, False, True],
                          list(alternating.answers()))

        no_history = self.model_from_str('')
        self.assertEquals([], list(no_history.answers()))
