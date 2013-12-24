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
import time

# necessary to do this after importing numpy to take advantage of
# multiple cores on unix
try:
    import affinity
    affinity.set_process_affinity_mask(0, 2 ** multiprocessing.cpu_count() - 1)
except NotImplementedError:
    # Affinity is only implemented for linux systems, and multithreading should
    # work on other systems.
    pass

from mirt import mirt_util
from train_util import model_training_util

# num_exercises and generate_exercise_ind are used in the creation of a
# defaultdict for mapping exercise names to an unique integer index
num_exercises = 0


def get_indexer(options):
    """Retrieve an object with data about the position of information in
    each row of data.
    """
    if options.data_format == 'simple':
        idx_pl = model_training_util.FieldIndexer(
            model_training_util.FieldIndexer.simple_fields)
    else:
        idx_pl = model_training_util.FieldIndexer(
            model_training_util.FieldIndexer.plog_fields)
    return idx_pl


def generate_exercise_ind():
    """Assign the next available index to an exercise name."""
    global num_exercises
    num_exercises += 1
    return num_exercises - 1


def sample_abilities_diffusion(args):
    """Sample the ability vector for this user, from the posterior over user
    ability conditioned on the observed exercise performance.
    use Metropolis-Hastings with Gaussian proposal distribution.

    This is just a wrapper around the corresponding function in mirt_util.
    """
    # TODO(jascha) make this a better sampler (eg, use the HMC sampler from
    # TMIRT)

    # make sure each student gets a different random sequence
    student_id = multiprocessing.current_process().ident
    if len(student_id) > 0:
        np.random.seed([student_id[0], time.time() * 1e9])
    else:
        np.random.seed([time.time() * 1e9])

    theta, state, options, user_index = args
    abilities = state['abilities']
    correct = state['correct']
    log_time_taken = state['log_time_taken']
    exercises_ind = state['exercises_ind']

    num_steps = options.sampling_num_steps

    abilities, Eabilities, _, _ = mirt_util.sample_abilities_diffusion(
            theta, exercises_ind, correct, log_time_taken,
            abilities, num_steps)

    return abilities, Eabilities, user_index


def get_cmd_line_options(arguments=None):
    """Retrieve options that parameterize the model training."""
    parser = optparse.OptionParser()
    parser.add_option("-a", "--num_abilities", type=int, default=1,
                      help=("Number of hidden ability units"))
    parser.add_option("-s", "--sampling_num_steps", type=int, default=50,
                      help=("Number of sampling steps to use for "
                            "sample_abilities_diffusion"))
    parser.add_option("-l", "--sampling_epsilon", type=float, default=0.1,
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
    parser.add_option("-z", "--correct_only", action="store_true",
                      default=False,
                      help=("Ignore response time, only model using "
                            "correctness."))
    parser.add_option("-r", "--resume_from_file", default='',
                      help=("Name of a .npz file to bootstrap the couplings."))
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


def create_user_state(lines, exercise_ind_dict, options):
    """Create a dictionary to hold training information for a single user."""
    idx_pl = get_indexer(options)
    correct = np.asarray([line[idx_pl.correct] for line in lines]
                         ).astype(int)
    time_taken = np.asarray([line[idx_pl.time_taken] for line in lines]
                            ).astype(int)
    time_taken[time_taken < 1] = 1
    time_taken[time_taken > options.max_time_taken] = options.max_time_taken
    exercises = [line[idx_pl.exercise] for line in lines]
    exercises_ind = [exercise_ind_dict[ex] for ex in exercises]
    exercises_ind = np.array(exercises_ind)
    abilities = np.random.randn(options.num_abilities, 1)

    # cut out any duplicate exercises in the training data for a single user
    # NOTE if you allow duplicates, you need to change the way the gradient
    # is computed as well.
    _, idx = np.unique(exercises_ind, return_index=True)
    exercises_ind = exercises_ind[idx]
    correct = correct[idx]
    time_taken = time_taken[idx]

    state = {'correct': correct,
             'log_time_taken': np.log(time_taken),
             'abilities': abilities,
             'exercises_ind': exercises_ind}

    return state


def emit_features(user_states, theta, options, split_desc):
    """Emit a CSV data file of correctness, prediction, and abilities."""
    with open(
            "%s_split=%s.csv" % (options.output, split_desc), 'w+') as outfile:

        for user_state in user_states:
            # initialize
            abilities = np.zeros((options.num_abilities, 1))
            correct = user_state['correct']
            log_time_taken = user_state['log_time_taken']
            exercises_ind = user_state['exercises_ind']

            # NOTE: I currently do not output features for the first problem
            for i in xrange(1, correct.size):

                # TODO(jace) this should probably be the marginal estimation
                _, _, abilities, _ = mirt_util.sample_abilities_diffusion(
                        theta, exercises_ind[:i], correct[:i],
                        log_time_taken[:i], abilities_init=abilities,
                        num_steps=200)
                prediction = mirt_util.conditional_probability_correct(
                        abilities, theta, exercises_ind[i:(i + 1)])

                print >>outfile, "%d," % correct[i],
                print >>outfile, "%.4f," % prediction[-1],
                print >>outfile, ",".join(["%.4f" % a for a in abilities])


def main():
    """Retrieve command line options and run training"""
    options = get_cmd_line_options()
    run(options)


def run_programmatically(arguments):
    """Take arguments from some other calling program and run training"""
    options = get_cmd_line_options(arguments)
    run(options)


def run(options):
    """Train a model with specified parameters."""
    print >>sys.stderr, "Starting main.", options  # DEBUG
    idx_pl = get_indexer(options)
    exercise_ind_dict = defaultdict(generate_exercise_ind)

    user_states = []
    user_states_train = []
    user_states_test = []

    print >>sys.stderr, "loading data"
    prev_user = None
    attempts = []
    # used to index the fields in with a line of text in the input data file
    linesplit = re.compile('[\t,\x01]')

    for replica_num in range(options.num_replicas):
        # loop through all the training data, and create user objects
        for row in fileinput.input(options.file):
            # split on either tab, comma, or \x01 so the code works via Hive
            # or pipe
            row = linesplit.split(row.strip())
            user = row[idx_pl.user]
            if prev_user and user != prev_user and len(attempts) > 0:
                # We're getting a new user, so perform the reduce operation
                # on our previous user
                user_states.append(create_user_state(
                        attempts, exercise_ind_dict, options))
                attempts = []
                prev_user = user
                row[idx_pl.correct] = row[idx_pl.correct] == 'true'
                row[idx_pl.time_taken] = float(row[idx_pl.time_taken])
                attempts.append(row)

            prev_user = user
            row[idx_pl.correct] = row[idx_pl.correct] == 'true'
            row[idx_pl.time_taken] = float(row[idx_pl.time_taken])
            attempts.append(row)

        if len(attempts) > 0:
            # flush the data for the final user, too
            user_states.append(create_user_state(
                    attempts, exercise_ind_dict, options))
            attempts = []

        fileinput.close()
        # Reset prev_user so we have equal user_states from each replica
        prev_user = None
        # split into training and test
        if options.training_set_size < 1.0:
            training_cutoff = int(len(user_states) * options.training_set_size)
            user_states_train += copy.deepcopy(user_states[:training_cutoff])
            print >>sys.stderr, len(user_states_train)
            if replica_num == 0:
                # we don't replicate the test data (only training data)
                user_states_test = copy.deepcopy(user_states[training_cutoff:])
            user_states = []

    # if splitting data into test/training sets, set user_states to training
    user_states = user_states_train if user_states_train else user_states

    print >>sys.stderr, "Training dataset, %d students" % (len(user_states))

    # initialize the parameters
    print >>sys.stderr, "%d exercises" % (num_exercises)

    mirt_model = mirt_util.MirtModel(
        options, num_exercises, dict(exercise_ind_dict),
        user_states)

    # now do num_epochs EM steps
    for epoch in range(options.num_epochs):
        mirt_model.run_em_step(epoch)

    if options.emit_features:
        if options.training_set_size < 1.0:
            emit_features(user_states_test, mirt_model.theta, options, "test")
            emit_features(
                user_states_train, mirt_model.theta, options, "train")
        else:
            emit_features(user_states, mirt_model.theta, options, "full")


if __name__ == '__main__':
    main()
