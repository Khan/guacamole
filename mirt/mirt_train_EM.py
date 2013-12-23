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
from multiprocessing import Pool
import numpy as np
import optparse
import scipy
import scipy.optimize
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

# used to index the fields in with a line of text in the input data file
linesplit = model_training_util.linesplit

# num_exercises and generate_exercise_ind are used in the creation of a
# defaultdict for mapping exercise names to an unique integer index
num_exercises = 0


def get_indexer(options):
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
    id = multiprocessing.current_process()._identity
    if len(id) > 0:
        np.random.seed([id[0], time.time() * 1e9])
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


def L_dL_singleuser(arg):
    """ calculate log likelihood and gradient wrt couplings of mIRT model
        for single user """
    theta, state, options = arg

    abilities = state['abilities'].copy()
    correct = state['correct']
    exercises_ind = state['exercises_ind']

    dL = mirt_util.Parameters(theta.num_abilities, len(exercises_ind))

    # pad the abilities vector with a 1 to act as a bias
    abilities = np.append(abilities.copy(),
                          np.ones((1, abilities.shape[1])),
                          axis=0)
    # the abilities to exercise coupling parameters for this exercise
    W_correct = theta.W_correct[exercises_ind, :]

    # calculate the probability of getting a question in this exercise correct
    Y = np.dot(W_correct, abilities)
    Z = mirt_util.sigmoid(Y)  # predicted correctness value
    Zt = correct.reshape(Z.shape)  # true correctness value
    pdata = Zt * Z + (1. - Zt) * (1. - Z)  # = 2*Zt*Z - Z + const
    dLdY = ((2. * Zt - 1.) * Z * (1. - Z)) / pdata

    L = -np.sum(np.log(pdata))
    dL.W_correct = -np.dot(dLdY, abilities.T)

    if not options.correct_only:
        # calculate the probability of taking time response_time to answer
        log_time_taken = state['log_time_taken']
        # the abilities to time coupling parameters for this exercise
        W_time = theta.W_time[exercises_ind, :]
        sigma = theta.sigma_time[exercises_ind].reshape((-1, 1))
        Y = np.dot(W_time, abilities)
        err = (Y - log_time_taken.reshape((-1, 1)))
        L += np.sum(err ** 2 / sigma ** 2) / 2.
        dLdY = err / sigma ** 2

        dL.W_time = np.dot(dLdY, abilities.T)
        dL.sigma_time = (-err ** 2 / sigma ** 3).ravel()

        # normalization for the Gaussian
        L += np.sum(0.5 * np.log(sigma ** 2))
        dL.sigma_time += 1. / sigma.ravel()

    return L, dL, exercises_ind


def L_dL(theta_flat, user_states, num_exercises, options, pool):
    """ calculate log likelihood and gradient wrt couplings of mIRT model """

    L = 0.
    theta = mirt_util.Parameters(options.num_abilities, num_exercises,
                                 vals=theta_flat.copy())

    nu = float(len(user_states))

    # note that the nu gets divided back out below, so the regularization term
    # does not end up with a factor of nu.
    L += options.regularization * nu * np.sum(theta_flat ** 2)
    dL_flat = 2. * options.regularization * nu * theta_flat
    dL = mirt_util.Parameters(theta.num_abilities, theta.num_exercises,
                              vals=dL_flat)

    # also regularize the inverse of sigma, so it doesn't run to 0
    L += np.sum(options.regularization * nu / theta.sigma_time ** 2)
    dL.sigma_time += -2. * options.regularization * nu / theta.sigma_time ** 3

    # TODO(jascha) this would be faster if user_states was divided into
    # minibatches instead of single students
    if pool is None:
        rslts = map(L_dL_singleuser,
                    [(theta, state, options) for state in user_states])
    else:
        rslts = pool.map(L_dL_singleuser,
                         [(theta, state, options) for state in user_states],
                         chunksize=100)
    for r in rslts:
        Lu, dLu, exercise_indu = r
        L += Lu
        dL.W_correct[exercise_indu, :] += dLu.W_correct
        dL.W_time[exercise_indu, :] += dLu.W_time
        dL.sigma_time[exercise_indu] += dLu.sigma_time

    if options.correct_only:
        dL.W_time[:, :] = 0.
        dL.sigma_time[:] = 0.

    dL_flat = dL.flat()

    # divide by log 2 so the answer is in bits instead of nats, and divide by
    # nu (the number of users) so that the magnitude of the log likelihood
    # stays reasonable even when trained on many users.
    L /= np.log(2.) * nu
    dL_flat /= np.log(2.) * nu

    return L, dL_flat


def emit_features(user_states, theta, options, split_desc):
    """Emit a CSV data file of correctness, prediction, and abilities."""
    f = open("%s_split=%s.csv" % (options.output, split_desc), 'w+')

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
                    theta, exercises_ind[:i], correct[:i], log_time_taken[:i],
                    abilities_init=abilities, num_steps=200)
            prediction = mirt_util.conditional_probability_correct(
                    abilities, theta, exercises_ind[i:(i + 1)])

            print >>f, "%d," % correct[i],
            print >>f, "%.4f," % prediction[-1],
            print >>f, ",".join(["%.4f" % a for a in abilities])

    f.close()


def check_grad(L_dL, theta, args=()):
    print >>sys.stderr, "Checking gradients."

    step_size = 1e-6

    f0, df0 = L_dL(theta.copy(), *args)
    # test gradients in random order. This lets us run check gradients on the
    # full size model, but still statistically test every type of gradient.
    test_order = range(theta.shape[0])
    np.random.shuffle(test_order)
    for ind in test_order:
        theta_offset = np.zeros(theta.shape)
        theta_offset[ind] = step_size
        f1, df1 = L_dL(theta.copy() + theta_offset, *args)
        df_true = (f1 - f0) / step_size

        # error in the gradient divided by the mean gradient
        rr = (df0[ind] - df_true) * 2. / (df0[ind] + df_true)

        print "ind", ind, "ind mod 3", np.mod(ind, 3),
        print "ind/3", np.floor(ind / 3.),
        print "df pred", df0[ind], "df true", df_true,
        print "(df pred - df true)*2/(df pred + df true)", rr


def main():
    options = get_cmd_line_options()
    run(options)


def run_programmatically(arguments):
    options = get_cmd_line_options(arguments)
    run(options)


def run(options):
    print >>sys.stderr, "Starting main.", options  # DEBUG
    idx_pl = get_indexer(options)
    pool = None
    if options.workers > 1:
        pool = Pool(options.workers)
    exercise_ind_dict = defaultdict(generate_exercise_ind)

    user_states = []
    user_states_train = []
    user_states_test = []

    print >>sys.stderr, "loading data"
    prev_user = None
    attempts = []
    for replica_num in range(options.num_replicas):
        # loop through all the training data, and create user objects
        for line in fileinput.input(options.file):
            # split on either tab or \x01 so the code works via Hive or pipe
            row = linesplit.split(line.strip())
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
    theta = mirt_util.Parameters(options.num_abilities, num_exercises)
    theta.sigma_time[:] = 1.
    # we won't be adding any more exercises
    exercise_ind_dict = dict(exercise_ind_dict)

    if options.resume_from_file:
        # HACK(jace): I need a cheap way
        # to output features from a previously trained model.  To use this
        # hacky version, pass --num_epochs 0 and you must pass the same
        # data file the model in resume_from_file was trained on.
        resume_from_model = np.load(options.resume_from_file)
        theta = resume_from_model['theta'][()]
        exercise_ind_dict = resume_from_model['exercise_ind_dict']
        print >>sys.stderr, "Loaded parameters from %s" % (
            options.resume_from_file)

    # now do num_epochs EM steps
    for epoch in range(options.num_epochs):
        print >>sys.stderr, "epoch %d, " % epoch,

        # Expectation step
        # Compute (and print) the energies during learning as a diagnostic.
        # These should decrease.
        Eavg = 0.
        # TODO(jascha) this would be faster if user_states was divided into
        # minibatches instead of single students
        if pool is None:
            rslts = map(sample_abilities_diffusion,
                        [(theta, user_states[ind], options, ind)
                            for ind in range(len(user_states))])
        else:
            rslts = pool.map(sample_abilities_diffusion,
                            [(theta, user_states[ind], options, ind)
                                for ind in range(len(user_states))],
                            chunksize=100)
        for r in rslts:
            abilities, El, ind = r
            user_states[ind]['abilities'] = abilities.copy()
            Eavg += El / float(len(user_states))
        print >>sys.stderr, "E joint log L + const %f, " % (
                -Eavg / np.log(2.)),

        # debugging info -- accumulate mean and covariance of abilities vector
        mn_a = 0.
        cov_a = 0.
        for state in user_states:
            mn_a += state['abilities'][:, 0].T / float(len(user_states))
            cov_a += (state['abilities'][:, 0] ** 2).T / (
                        float(len(user_states)))
        print >>sys.stderr, "<abilities>", mn_a,
        print >>sys.stderr, ", <abilities^2>", cov_a, ", ",

        # check_grad(L_dL, theta.flat(), args=(user_states,
        #     num_exercises, options, pool))

        # Maximization step
        old_theta_flat = theta.flat()
        #print "about to minimize"
        theta_flat, L, _ = scipy.optimize.fmin_l_bfgs_b(
            L_dL,
            theta.flat(),
            args=(user_states, num_exercises, options, pool),
            disp=0,
            maxfun=options.max_pass_lbfgs, m=100)
        theta = mirt_util.Parameters(options.num_abilities, num_exercises,
                                     vals=theta_flat)

        if options.correct_only:
            theta.sigma_time[:] = 1.
            theta.W_time[:, :] = 0.

        # Print debugging info on the progress of the training
        print >>sys.stderr, "M conditional log L %f, " % (-L),
        print >>sys.stderr, "reg penalty %f, " % (
                options.regularization * np.sum(theta_flat ** 2)),
        print >>sys.stderr, "||couplings|| %f, " % (
                np.sqrt(np.sum(theta.flat() ** 2))),
        print >>sys.stderr, "||dcouplings|| %f" % (
                np.sqrt(np.sum((theta_flat - old_theta_flat) ** 2)))

        # Maintain a consistent directional meaning of a
        # high/low ability esimtate.  We always prefer higher ability to
        # mean better performance; therefore, we prefer positive couplings.
        # So, compute the sign of the average coupling for each dimension.
        coupling_sign = np.sign(np.mean(theta.W_correct[:, :-1], axis=0))
        coupling_sign = coupling_sign.reshape((1, -1))
        # Then, flip ability and coupling sign for dimenions w/ negative mean.
        theta.W_correct[:, :-1] *= coupling_sign
        theta.W_time[:, :-1] *= coupling_sign
        for user_state in user_states:
            user_state['abilities'] *= coupling_sign.T

        # save state as a .npz
        np.savez("%s_epoch=%d.npz" % (options.output, epoch),
                 theta=theta,
                 exercise_ind_dict=exercise_ind_dict,
                 max_time_taken=options.max_time_taken)

        # save state as .csv - just for easy debugging inspection
        f1 = open("%s_epoch=%d.csv" % (options.output, epoch), 'w+')
        nms = sorted(exercise_ind_dict.keys(),
                key=lambda nm: theta.W_correct[exercise_ind_dict[nm], -1])

        print >>f1, 'correct bias,',
        for ii in range(options.num_abilities):
            print >>f1, "correct coupling %d," % ii,
        print >>f1, 'time bias,',
        for ii in range(options.num_abilities):
            print >>f1, "time coupling %d," % ii,
        print >>f1, 'time variance,',
        print >>f1, 'exercise name'
        for nm in nms:
            print >>f1, theta.W_correct[exercise_ind_dict[nm], -1], ',',
            for ii in range(options.num_abilities):
                print >>f1, theta.W_correct[exercise_ind_dict[nm], ii], ',',
            print >>f1, theta.W_time[exercise_ind_dict[nm], -1], ',',
            for ii in range(options.num_abilities):
                print >>f1, theta.W_time[exercise_ind_dict[nm], ii], ',',
            print >>f1, theta.sigma_time[exercise_ind_dict[nm]], ',',
            print >>f1, nm
        f1.close()

    if options.emit_features:
        if options.training_set_size < 1.0:
            emit_features(user_states_test, theta, options, "test")
            emit_features(user_states_train, theta, options, "train")
        else:
            emit_features(user_states, theta, options, "full")


if __name__ == '__main__':
    main()
