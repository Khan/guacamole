"""This file generates a sample training file from which data can be provided
to train an item response theory model. We generate a variety of students and
of assessment items, and then generate sample traces of assessments those
students could have taken. """
import numpy as np
import random
import string

import mirt.mirt_util as mirt_util


class Student(object):
    """Represent a student (test-taker) with certain ability vectors"""

    def __init__(self, student_id, num_abilities=1):
        self.student_id = student_id
        self.abilities = np.random.randn(num_abilities, 1) * 3


class AssessmentItem(mirt_util.Parameters):
    """Represent an assessment item with an id and difficulties"""

    def __init__(self, item_id, num_abilities=1):
        self.item_id = str(item_id)
        super(AssessmentItem, self).__init__(num_abilities, 1)

        # We initialize the couplings for correct/incorrect responses to a
        # random normal distribution, adding one to the dimensionality to
        # account for bias.
        self.W_correct = np.random.rand(1, num_abilities + 1)
        #self.W_correct[:1] = self.W_correct + 1


class Assessment(object):
    """An assessment is an ordered collection of assessment items"""

    def __init__(self, num_items, exercises):
        self.items = [AssessmentItem(item_id) for item_id in range(num_items)]
        # Use x.W_correct[0, 0]) as a proxy for difficulty
        self.items.sort(key=lambda x: x.W_correct[0, 1], reverse=True)
        for item_num, item in enumerate(self.items):
            item.item_id = exercises[len(exercises) * item_num / num_items]

    def get_items(self, num_items=0, randomize=False):
        """Draw a set of items from the assessment

        Default behavior is to return all items in given order. It is also
        possible to shuffle the items or to draw a random subsample

        Arguments:
            num_items: The number of items to return. If it's 0, we return all
                       items.
            randomize: A boolean. If it's true we shuffle the items before
                       returning them. If a subset of items are requested,
                       those will always be randomized.
        Returns:
            An ordered list of AssessmentItems
        """
        items = self.items
        if num_items and num_items <= len(self.items):
            items = random.choice(items, num_items)
        if randomize:
            random.shuffle(items)
        return items


class Response(object):
    """A student's response to an assessment item"""

    def __init__(self, item, correct):
        self.item = item
        self.correct = correct
        # TODO(eliana): Actually implement time_taken
        self.time_taken = int(random.random() * 20 + 1)


class StudentAssessment(object):
    """A student's assessment result, including responses

    This uses the assumptions of item response theory to generate a series
    of student responses to assessment items.
    """

    def __init__(self, student, assessment):
        self.student = student
        self.assessment = assessment
        self.responses = []

    def complete_assessment(self):
        """Fill in student performance on each assessment item"""
        for item in self.assessment.get_items():
            self.responses.append(Response(item, self.attempt(item)))

    def attempt(self, item):
        """Return whether the student responds correctly to the item.

        This is probabilistic - we get the probability that the student will
        respond correctly, and then generate a random number to see if they
        succeed.
        """
        p_correct = mirt_util.conditional_probability_correct(
            self.student.abilities,
            item,
            0)
        if random.random() <= p_correct:
            return True
        else:
            return False

    def response_strings(self):
        """Yield each response as a string suitable for training input"""
        for response in self.responses:
            yield string.join([
                self.student.student_id,
                response.item.item_id,
                str(response.time_taken),
                str(response.correct),
                ],
                ',')


def generate_sample_data(num_students=50, num_items=10, include_time=False,
                         num_abilities=1):
    """Generate a series of student assessments and simulate completing them
    """
    # Give the students real names for fun
    names = [name.strip() for name in open('sample_data/names.txt', 'r')]
    exercises = [name.strip() for name in open(
        'sample_data/exercise_names.txt', 'r')]
    random.shuffle(names)

    def get_name(student_id):
        return names[student_id % len(names)]
    students = [Student(get_name(student)) for student in range(num_students)]
    assessment = Assessment(num_items, exercises)
    completed_student_assessments = []
    for student in students:
        student_assessment = StudentAssessment(student, assessment)
        student_assessment.complete_assessment()
        completed_student_assessments.append(student_assessment)
    return completed_student_assessments


def print_sample_data(num_students=50, num_items=10, include_time=False,
                      num_abilities=1, data_file=None):
    """Generate sample assessment data and print it to stdout"""
    assessments = generate_sample_data(num_students=num_students,
                                       num_items=num_items,
                                       include_time=include_time,
                                       num_abilities=num_abilities)
    if data_file:
        with open(data_file, 'w') as data:
            for assessment in assessments:
                for response in assessment.response_strings():
                    data.write(response + '\n')
    else:
        for assessment in assessments:
            for response in assessment.response_strings():
                print response


def run(arguments):
    """Run with arguments"""
    print_sample_data(data_file=arguments.data_file,
                      num_students=arguments.num_students,
                      num_items=arguments.num_problems,
                      include_time=arguments.time,
                      num_abilities=arguments.abilities)

if __name__ == '__main__':
    print_sample_data()
