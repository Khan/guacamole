"""A variety of simple utilities used by guacamole that may come in
handy often.
"""
import errno
import os
import random


def sep_into_train_and_test(arguments, test_portion=.1):
    """Takes an input file and splits it into two files, a file for training
    a model and a file for testing a model. Supports multiple splits.
    TODO(eliana): Should probably eventually support n-fold x-validation
    This works in particular for time series data, and splits by the first
    field (assumed to be something like a user or a task id), rather than
    randomly distributing the lines.
    """
    data_file_name = os.path.expanduser(arguments.data_file)
    train_file_name = arguments.model_directory + 'train.responses'
    test_file_name = arguments.model_directory + 'test.responses'
    infile = open(data_file_name, 'r')
    train = open(train_file_name, 'w')
    test = open(test_file_name, 'w')
    current_user = ''
    for line in infile:
        user = line.split(',')[0]
        if user != current_user:
            current_user = user
            if random.random() < test_portion:
                current_file = test
            else:
                current_file = train
        current_file.write(line)


def mkdir_p(paths):
    """Emulates mkdir -p; makes a directory and its parents, with no complaints
    if the directory is already present
    """
    if type(paths) == str:
        paths = [paths]
    for path in paths:
        path = os.path.expanduser(path)
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


class FieldIndexer(object):
    """Describes the locations of the fields in a variety of data formats.
    Implement your own if you have a csv file you'd like to use features from
    and the fields are not in any of these orders.
    """
    def __init__(self, field_names):
        for i, field in enumerate(field_names):
            self.__dict__[field] = i

    @staticmethod
    def get_for_slug(slug):
        """Generate an indexer describing the locations of fields within data.
        """
        if slug == 'simple':
            return FieldIndexer(FieldIndexer.simple_fields)
        elif slug == 'topic_attempt_fields':
            return FieldIndexer(FieldIndexer.topic_attempt_fields)
        return FieldIndexer(FieldIndexer.plog_fields)

    def get_values(self):
        """"Return each value of the FieldIndexer"""
        return self.__dict__.values()

    def get_keys(self):
        """Return each key of the FieldIndexer"""
        return self.__dict__.keys()

    topic_attempt_fields = [
        'user', 'topic', 'exercise', 'time_done', 'time_taken',
        'problem_number', 'correct', 'scheduler_info', 'user_segment', 'dt']

    plog_fields = [
        'user', 'time_done', 'rowtype', 'exercise', 'problem_type', 'seed',
        'time_taken', 'problem_number', 'correct', 'number_attempts',
        'number_hints', 'eventually_correct', 'topic_mode', 'dt']

    simple_fields = ['user', 'exercise', 'time_taken', 'correct']


def sequential_problem_numbers(attempts, idx):
    """Take all problem logs for a user as a list of lists, indexed by idx,
    and make sure that problem numbers within an exercise are strictly
    increasing and never jump by more than one.
    """
    ex_prob_number = {}  # stores the current problem number for each exercise
    for attempt in attempts:

        ex = attempt[idx.exercise]
        prob_num = attempt[idx.problem_number]

        if ex not in ex_prob_number:
            ex_prob_number[ex] = prob_num
        else:
            if prob_num == ex_prob_number[ex] + 1:
                ex_prob_number[ex] = prob_num
            else:
                return False
    return True


def incomplete_history(attempts, idx):
    """Take all problem logs for a user as a list of lists.  The inner lists
    each represent a problem attempt, with items described and indexed by the
    idx argument.  This function returns True if we *know* we have an
    incomplete history for the user, by checking if the first problem seen
    for any exercise has a problem_number != 1.
    """
    exercises_seen = []
    for attempt in attempts:
        if attempt[idx.exercise] not in exercises_seen:
            if int(attempt[idx.problem_number]) != 1:
                return True
            exercises_seen.append(attempt[idx.exercise])
    return False


def valid_history(attempts, idx):
    """Validate that attempts on a problem have sequential problem numbers"""
    if not sequential_problem_numbers(attempts, idx):
        return False

    if incomplete_history(attempts, idx):
        return False

    return True
