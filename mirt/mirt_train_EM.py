"""This script trains, and then emits features generated using,
a multidimensional item response theory model.

USAGE:

  The KA website root and analytics directory must be on PYTHONPATH, e.g.,

  export PYTHONPATH=~/khan/website/stable:~/khan/analytics/src
  python mirt_train_EM.py -a 1 -n 75 -f PROD_RESPONSES -w 0 -o MIRT_NEW &> LOG

  Where PROD_RESPONSES is a number of UserAssessment data as formatted
  by get_user_assessment_data.py, MIRT_NEW is the root filename for
  output, and LOG is a logfile containing the stderr and stdout of
  this process.

"""
from collections import defaultdict
import copy
import datetime
import fileinput
import multiprocessing
import numpy as np
import optparse
import re
import sys

# necessary to do this after importing numpy to take advantage of
# multiple cores on unix
try:
    import affinity
    affinity.set_process_affinity_mask(0, 2 ** multiprocessing.cpu_count() - 1)
except NotImplementedError:
    # Affinity is only implemented for linux systems, and multithreading should
    # work on other systems.
    pass
except ImportError:
    # It's also OK if affinity is just not installed on the system
    pass

from mirt import mirt_util

# num_exercises and generate_exercise_ind are used in the creation of a
# defaultdict for mapping exercise names to an unique integer index
num_exercises = 0


def generate_exercise_ind():
    """Assign the next available index to an exercise name."""
    global num_exercises
    num_exercises += 1
    return num_exercises - 1


def get_cmd_line_options(arguments=None):
    """Retreive user specified parameters"""
    # TODO(eliana): Convert to argparse instead of optparse
    parser = optparse.OptionParser()
    parser.add_option("--time", action="store_true",
                      default=False,
                      help=("Include time in the model."))
    parser.add_option("-a", "--num_abilities", type=int, default=1,
                      help=("Number of hidden ability units"))
    parser.add_option("-s", "--sampling_num_steps", type=int, default=200,
                      help=("Number of sampling steps to use for "
                            "sample_abilities_diffusion"))
    parser.add_option("-l", "--sampling_epsilon", type=float, default=0.2,
                      help=("The length scale to use for sampling update "
                            "proposals"))
    parser.add_option("-n", "--num_epochs", type=int, default=10000,
                      help=("The number of EM iterations to do during "
                            "learning"))
    parser.add_option("-q", "--num_replicas", type=int, default=1,
                      help=("The number of copies of the data to train "
                            "on.  If there is too little training data, "
                            "increase this number in order to maintain "
                            "multiple samples from the abilities vector "
                            "for each student.  A sign that there is too "
                            "little training data is if the update step "
                            "length ||dcouplings|| remains large."))
    parser.add_option("-m", "--max_pass_lbfgs", type=int, default=5,
                      help=("The number of LBFGS descent steps to do per "
                            "EM iteration"))
    parser.add_option("-p", "--regularization", type=float, default=1e-5,
                      help=("The weight for an L2 regularizer on the "
                            "parameters.  This can be very small, but "
                            "keeps the weights from running away in a "
                            "weakly constrained direction."))
    parser.add_option("-w", "--workers", type=int, default=6,
                      help=("The number of processes to use to parallelize "
                            "this.  Set this to 0 to use one process, and "
                            "make debugging easier."))
    parser.add_option("-b", "--max_time_taken", type=int,
                      default=1e3,
                      help=("The maximum response time.  Longer responses "
                            "are set to this value."))
    parser.add_option("-f", "--file", type=str,
                      default='user_assessment.responses',
                      help=("The source data file"))
    parser.add_option("-o", "--output", type=str, default='',
                      help=("The root filename for output"))
    parser.add_option("-t", "--training_set_size", type=float, default=1.0,
                      help=("The fraction (expressed as a number beteween 0.0 "
                            "and 1.0) of the data to be used for training. "
                            "The remainder is held out for testing."))
    parser.add_option("-e", "--emit_features", action="store_true",
                      default=False,
                      help=("Boolean flag indicating whether to output "
                            "feature and prediction data. Often used to "
                            "analyze accuracy of predictions after model "
                            "training."))
    parser.add_option("-r", "--resume_from_file", default='',
                      help=("Name of a json file to bootstrap the couplings."
                            "WARNING: Not Fully Supported"))
    parser.add_option("-d", "--data_format", default='simple',
                      help=("The field indexer format of the input."))

    if arguments:
        options, _ = parser.parse_args(arguments)
    else:
        options, _ = parser.parse_args()

    if options.output == '':
        # default filename
        options.output = "mirt_file=%s_abilities=%d_time=%s" % (
            options.file, options.num_abilities,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return options


def emit_features(user_states, theta, options, split_desc):
    """Emit a CSV data file of correctness, prediction, and abilities."""
    f = open("%s_split=%s.csv" % (options.output, split_desc), 'w+')
    for user_state in user_states:
        # initialize
        abilities = np.zeros((options.num_abilities, 1))
        correct = user_state['correct']
        log_time_taken = user_state['log_time_taken']
        exercise_ind = user_state['exercise_ind']

        # NOTE: I currently do not output features for the first problem
        for i in xrange(1, correct.size):

            # TODO(jace) this should probably be the marginal estimation
            _, _, abilities, _ = mirt_util.sample_abilities_diffusion(
                theta, exercise_ind[:i], correct[:i], log_time_taken[:i],
                abilities_init=abilities, num_steps=200)
            prediction = mirt_util.conditional_probability_correct(
                abilities, theta, exercise_ind[i:(i + 1)])

            f.write("%d, " % correct[i])
            f.write("%.4f, " % prediction[-1])
            f.write(",".join(["%.4f" % a for a in abilities]))
            f.write('\n')

    f.close()


def main():
    options = get_cmd_line_options()
    run(options)


def run_programmatically(arguments):
    options = get_cmd_line_options(arguments)
    run(options)


def get_data_from_file(options, exercise_ind_dict):
    """Iterate through the input file to retrieve student responses
    Input: the options specified by the user
    Outputs: The user_states
    """
    user_states = []
    user_states_train = []
    user_states_test = []

    idx_pl = mirt_util.get_indexer(options)
    sys.stderr.write("loading data")
    prev_user = None
    attempts = []
    linesplit = re.compile('[\t,\x01]')
    for replica_num in range(options.num_replicas):
        # loop through all the training data, and create user objects
        for line in fileinput.input(options.file):
            # split on either tab or \x01 so the code works via Hive or pipe
            row = linesplit.split(line.strip())
            user = row[idx_pl.user]
            if prev_user and user != prev_user and len(attempts) > 0:
                # We're getting a new user, so perform the reduce operation
                # on our previous user
                user_state = mirt_util.UserState()
                user_state.add_data(attempts, exercise_ind_dict, options)
                user_states.append(user_state)
                attempts = []

            prev_user = user
            row[idx_pl.correct] = row[idx_pl.correct] in ('true', 'True', '1')
            row[idx_pl.time_taken] = float(row[idx_pl.time_taken])
            attempts.append(row)

        if len(attempts) > 0:
            # flush the data for the final user, too
            user_state = mirt_util.UserState()
            user_state.add_data(attempts, exercise_ind_dict, options)
            user_states.append(user_state)

        fileinput.close()
        # Reset prev_user so we have equal user_states from each replica
        prev_user = None

        # split into training and test
        if options.training_set_size < 1.0:
            training_cutoff = int(len(user_states) * options.training_set_size)
            user_states_train += copy.deepcopy(user_states[:training_cutoff])
            sys.stderr.write(str(len(user_states_train)) + '\n')
            if replica_num == 0:
                # we don't replicate the test data (only training data)
                user_states_test = copy.deepcopy(user_states[training_cutoff:])
            user_states = []

    # if splitting data into test/training, set user_states to training
    user_states = user_states_train if user_states_train else user_states

    return user_states, user_states_train, user_states_test


def run(options):
    sys.stderr.write("Starting main." + str(options) + '\n')  # DEBUG
    exercise_ind_dict = defaultdict(generate_exercise_ind)

    # Load information from the file
    user_states, user_states_train, user_states_test = get_data_from_file(
        options, exercise_ind_dict)

    sys.stderr.write("Training dataset, %d students\n" % (len(user_states)))

    # Initialize the parameters
    sys.stderr.write("%d exercises\n" % (num_exercises))
    # we won't be adding any more exercises, so we can cast this defaultdict to
    # a dict
    exercise_ind_dict = dict(exercise_ind_dict)

    model = mirt_util.MirtModel(
        options, num_exercises, exercise_ind_dict, user_states)

    # now do num_epochs EM steps
    for epoch in range(options.num_epochs):
        model.run_em_step(epoch)

    if options.emit_features:
        # TODO -- add a comment explaining what's going on here
        if options.training_set_size < 1.0:
            emit_features(user_states_test, model.theta, options, "test")
            emit_features(user_states_train, model.theta, options, "train")
        else:
            emit_features(user_states, model.theta, options, "full")


if __name__ == '__main__':
    main()
