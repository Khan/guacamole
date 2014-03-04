"""A utility module for mirt training, containing a variety of useful
datastructures.

In this file:

functions:
    sigmoid:
        compute sigmoid element-wise on an ndarray
    get_exercises_ind:
        turn an array of exercise names into an array of indices within a
        matrix.
    conditional_probability_correct:
        predict the probabilities of getting questions correct given exercises,
        model parameters, and user abilities.
    conditional_energy_data:
        calculate the energy of correctness data given exercises,
        parameters, and abilities.
    sample_abilities_diffusion:
        sample the ability vector for this user from the posterior over user
        ability conditioned on the observed exercise performance.

class Parameters, which holds parameters for a MIRT model.

"""
import json
from multiprocessing import Pool
import numpy as np
import scipy
import sys
import warnings
import multiprocessing
import time

from train_util.regression_util import sigmoid


class Parameters(object):
    """
    Holds the parameters for a MIRT model.  Also used to hold the gradients
    for each parameter during training.
    """
    def __init__(self, num_abilities, num_exercises, vals=None,
                 exercise_ind_dict=None):
        """vals is a 1d array holding the flattened parameters """
        self.num_abilities = num_abilities
        self.num_exercises = num_exercises
        if vals is None:
            # the couplings to correct/incorrect (+1 for bias unit)
            self.W_correct = np.zeros((num_exercises, num_abilities + 1))
            # the couplings to time taken (+1 for bias unit)
            self.W_time = np.zeros((num_exercises, num_abilities + 1))
            # the standard deviation for the response time Gaussian
            self.sigma_time = np.zeros((num_exercises))
        else:
            # the couplings to correct/incorrect (+1 for bias unit)
            num_couplings = num_exercises * (num_abilities + 1)
            self.W_correct = vals[:num_couplings].copy().reshape(
                (-1, num_abilities + 1))
            # the couplings to time taken (+1 for bias unit)
            self.W_time = vals[num_couplings:2 * num_couplings].copy().reshape(
                (-1, num_abilities + 1))
            # the standard deviation for the response time Gaussian
            self.sigma_time = vals[2 * num_couplings:].reshape((-1))
            self.exercise_ind_dict = exercise_ind_dict

    def flat(self):
        """Returns a concatenation of the parameters for saving."""
        return np.concatenate((self.W_correct.ravel(), self.W_time.ravel(),
                               self.sigma_time.ravel()))


def get_exercises_ind(exercise_names, exercise_ind_dict):
    """Turn an array of exercise names into an array of indices within the
    couplings parameter matrix

    Args:
        exercise_names: A python list of exercise names (strings).
        exercise_ind_dict: A python dict mapping exercise names
            to their corresponding integer row index in a couplings
            matrix.

    Returns:
        A 1-d ndarray of indices, with shape = (len(exercise_names)).

    """
    # do something sensible if a string is passed in instead of an array-like
    # object -- just wrap it
    if isinstance(exercise_names, str) or isinstance(exercise_names, unicode):
        exercise_names = [exercise_names]
    inds = np.zeros(len(exercise_names), int)
    for i in range(len(exercise_names)):
        inds[i] = exercise_ind_dict[exercise_names[i]]
    return inds


def conditional_probability_correct(abilities, ex_parameters, exercises_ind):
    """Predict the probabilities of getting questions correct for a set of
    exercise indices, given model parameters in couplings and the
    abilities vector for the user.

    Args:
        abilities: An ndarray with shape = (a, 1), where a = the
            number of abilities in the model (not including the bias)
        couplings: An ndarray with shape (n, a + 1), where n = the
            number of exercises known by the model.
        execises_ind: An ndarray of exercise indices in the coupling matrix.
            The argument specifies which exercises the caller would like
            conditional probabilities for.
            Should be 1-d with shape = (# of exercises queried for)

    Returns:
        An ndarray of floats with shape = (exercises_ind.size)
    """
    # Pad the abilities vector with a 1 to act as a bias.
    # The shape of abilities will become (a+1, 1).
    abilities = np.append(abilities.copy(), np.ones((1, 1)), axis=0)
    difficulties = ex_parameters.W_correct[exercises_ind, :]
    Z = sigmoid(np.dot(difficulties, abilities))
    Z = np.reshape(Z, Z.size)  # flatten to 1-d ndarray
    return Z


def conditional_energy_data(
        abilities, theta, exercises_ind, correct, log_time_taken):
    """Calculate the energy of the observed responses "correct" to
    exercises "exercises", conditioned on the abilities vector for a single
    user, and with MIRT parameters given in "couplings"

    Args:
        abilities: An ndarray of shape (a, 1), where a is the dimensionality
            of abilities not including bias.
        theta: An object holding the mIRT model parameters.
        execises_ind: A 1-d ndarray of exercise indices in the coupling matrix.
            This argument specifices the source exercise for the observed
            data in the 'correct' argument.  Should be 1-d with shape = (q),
            where q = the # of problem observations conditioned on.
        correct: A 1-d ndarray of integers with the same shape as
            exercises_ind. The element values should equal 1 or 0 and
            represent an observed correct or incorrect answer, respectively.
            The elements of 'correct' correspond to the elements of
            'exercises_ind' in the same position.
        log_time_taken: A 1-d ndarray similar to correct, but holding the
            log of the response time (in seconds) to answer each problem.

    Returns:
        A 1-d ndarray of probabilities, with shape = (q)
    """
    # predicted probability correct
    c_pred = conditional_probability_correct(abilities, theta, exercises_ind)
    # probability of actually observed sequence of responses
    p_data = c_pred * correct + (1 - c_pred) * (1 - correct)

    # probability of observed time_taken
    # TODO(jascha) - This code could be faster if abilities was stored with
    # the bias term, rather than having to repeatedly copy and concatenate the
    # bias in an inner loop.
    abilities = np.append(abilities.copy(), np.ones((1, 1)), axis=0)
    W_time = theta.W_time[exercises_ind, :]
    sigma_time = theta.sigma_time[exercises_ind]
    pred_time_taken = np.dot(W_time, abilities)
    err = pred_time_taken.ravel() - log_time_taken
    E_time_taken = (err.ravel() ** 2 / (2. * sigma_time.ravel() ** 2) +
                    0.5 * np.log(sigma_time ** 2))

    E_observed = -np.log(p_data) + E_time_taken
    assert len(E_observed.shape) == 1

    return E_observed


def sample_abilities_diffusion_wrapper(theta, state, options, user_index):
    """Sample the ability vector for this user

    Sample from the posterior over user ability conditioned on the observed
    exercise performance. use Metropolis-Hastings with Gaussian proposal
    distribution.

    This is just a wrapper around sample_abilities_diffusion.
    """
    # TODO(jascha) make this a better sampler (eg, use the HMC sampler from
    # TMIRT)

    # make sure each student gets a different random sequence
    id = multiprocessing.current_process()._identity
    if len(id) > 0:
        np.random.seed([id[0], time.time() * 1e9])
    else:
        np.random.seed([time.time() * 1e9])

    abilities = state['abilities']
    correct = state['correct']
    log_time_taken = state['log_time_taken']
    exercises_ind = state['exercises_ind']

    num_steps = options.sampling_num_steps

    #abilities, Eabilities, _, _ = sample_abilities_diffusion(
    _, Eabilities, abilities, _ = sample_abilities_diffusion(
        theta, exercises_ind, correct, log_time_taken,
        abilities, num_steps)

    return abilities, Eabilities, user_index


def sample_abilities_diffusion(
        theta, exercises_ind, correct, log_time_taken, abilities_init=None,
        num_steps=1, sampling_epsilon=.5):
    """Sample the ability vector for this user from the posterior over user
    ability conditioned on the observed exercise performance. Use
    Metropolis-Hastings with Gaussian proposal distribution.

    Args:
        couplings: The parameters of the MIRT model.  An ndarray with
            shape (n, a + 1), where n = the number of exercises known by the
            model and a = is the dimensionality of abilities (not including
            bias).
        execises_ind: A 1-d ndarray of exercise indices in the coupling matrix.
            This argument specifices the source exercise for the observed
            data in the 'correct' argument.  Should be 1-d with shape = (q),
            where q = the # of problem observations conditioned on.
        correct: A 1-d ndarray of integers with the same shape as
            exercises_ind. The element values should equal 1 or 0 and
            represent an observed correct or incorrect answer, respectively.
            The elements of 'correct' correspond to the elements of
            'exercises_ind' in the same position.
        log_time_taken: A 1-d ndarray of floats with the same shape as
            exercises_ind. The element values are the log of the response
            time.  The elements correspond to the elements of 'exercises_ind'
            in the same position.
        abilities_init: None, or an ndarray of shape (a, 1) representing
            a desired initialization of the abilities in the sampling chain.
            If None, abilities are intitialized to noise.
        num_steps: The number of sampling iterations.
        sampling_epsilon: distance parameter for generating proposals.

    Returns: a four-tuple.  The positional values are:
        1: the final abilities sample in the chain
        2: the energy of the final sample in the chain
        3: The mean of the abilities vectors in the entire chain.
        4: The standard deviation of the abilities vectors in the entire chain.
    """
    # TODO -- this would run faster with something like an HMC sampler

    # initialize abilities using prior
    if abilities_init is None:
        abilities = np.random.randn(theta.num_abilities, 1)
    else:
        abilities = abilities_init

    # calculate the energy for the initialization state
    E_abilities = 0.5 * np.dot(abilities.T, abilities) + np.sum(
        conditional_energy_data(
            abilities, theta, exercises_ind, correct, log_time_taken))

    sample_chain = []
    for _ in range(num_steps):
        # generate the proposal state
        proposal = abilities + sampling_epsilon * np.random.randn(
            theta.num_abilities, 1)

        E_proposal = 0.5 * np.dot(proposal.T, proposal) + np.sum(
            conditional_energy_data(
                proposal, theta, exercises_ind, correct, log_time_taken))

        # probability of accepting proposal
        if E_abilities - E_proposal > 0.:
            # this is required to avoid overflow when E_abilities - E_proposal
            # is very large
            p_accept = 1.0
        else:
            p_accept = np.exp(E_abilities - E_proposal)

        if not np.isfinite(E_proposal):
            warnings.warn("Warning.  Non-finite proposal energy.")
            p_accept = 0.0

        if p_accept > np.random.rand():
            abilities = proposal
            E_abilities = E_proposal

        sample_chain.append(abilities[:, 0].tolist())

    sample_chain = np.asarray(sample_chain)

    # Compute the abilities posterior mean.
    mean_sample_abilities = np.mean(sample_chain, 0).reshape(
        theta.num_abilities, 1)
    stdev = np.std(sample_chain, 0).reshape(theta.num_abilities, 1)

    return abilities, E_abilities, mean_sample_abilities, stdev


def L_dL_singleuser(arg):
    """ calculate log likelihood and gradient wrt couplings of mIRT model
        for single user """
    theta, state, options = arg

    abilities = state['abilities'].copy()
    correct = state['correct']
    exercises_ind = state['exercises_ind']

    dL_flat_allzeros = Parameters(
        theta.num_abilities, len(exercises_ind)).flat()

    dL_flat_allzeros[:] = 0
    dL = Parameters(theta.num_abilities, theta.num_exercises,
                    vals=dL_flat_allzeros)
    # pad the abilities vector with a 1 to act as a bias
    abilities = np.append(abilities.copy(),
                          np.ones((1, abilities.shape[1])),
                          axis=0)
    # the abilities to exercise coupling parameters for this exercise
    W_correct = theta.W_correct[exercises_ind, :]

    # calculate the probability of getting a question in this exercise correct
    Y = np.dot(W_correct, abilities)
    Z = sigmoid(Y)  # predicted correctness value
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
    """Calculate log likelihood and gradient wrt couplings of mIRT model """

    L = 0.
    theta = Parameters(options.num_abilities, num_exercises,
                       vals=theta_flat.copy())

    num_users = float(len(user_states))

    # note that the nu gets divided back out below, so the regularization term
    # does not end up with a factor of nu.
    L += options.regularization * num_users * np.sum(theta_flat ** 2)
    dL_flat = 2. * options.regularization * num_users * theta_flat
    dL = Parameters(theta.num_abilities, theta.num_exercises,
                    vals=dL_flat)

    # also regularize the inverse of sigma, so it doesn't run to 0
    L += np.sum(options.regularization * num_users / theta.sigma_time ** 2)
    dL.sigma_time += (-2. * options.regularization * num_users
                      / theta.sigma_time ** 3)

    # TODO(jascha) this would be faster if user_states was divided into
    # minibatches instead of single students
    if pool is None:
        rslts = map(L_dL_singleuser, [(theta, state, options)
                    for state in user_states])
    else:
        rslts = pool.map(L_dL_singleuser,
                         [(theta, state, options) for state in user_states],
                         chunksize=100)
    for r in rslts:
        Lu, dLu, exercise_indu = r
        L += Lu
        print 'r'
        print r
        print 'dl'
        print dL.W_correct[exercise_indu, :]
        print 'dLu'
        print dLu.W_correct
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
    L /= np.log(2.) * num_users
    dL_flat /= np.log(2.) * num_users

    return L, dL_flat


class MirtModel(object):
    """A model that contains the parameters of a multidimensional item response
    theory model
    """
    def __init__(self, options, num_exercises, exercise_ind_dict,
                 user_states):
        self.theta = Parameters(options.num_abilities, num_exercises)
        self.theta.sigma_time[:] = 1.
        if options.resume_from_file:
            # HACK(jace): I need a cheap way
            # to output features from a previously trained model.  To use this
            # hacky version, pass --num_epochs 0 and you must pass the same
            # data file the model in resume_from_file was trained on.
            resume_from_model = np.load(options.resume_from_file)
            self.theta = resume_from_model['theta'][()]
            exercise_ind_dict = resume_from_model['exercise_ind_dict']
            sys.stderr.write("Loaded parameters from %s" % (
                options.resume_from_file))
        self.num_exercises = num_exercises
        self.pool = None
        if options.workers > 1:
            self.pool = Pool(options.workers)
        self.options = options
        self.exercise_ind_dict = exercise_ind_dict
        self.user_states = user_states

    def get_results(self):
        """Samples the ability vectors for the students in the data"""

        if self.pool is None:
            results = [sample_abilities_diffusion_wrapper(
                       self.theta, self.user_states[ind], self.options, ind)
                       for ind in range(len(self.user_states))]
        else:
            results = self.pool.map(
                sample_abilities_diffusion_wrapper,
                [(self.theta, self.user_states[ind], self.options, ind)
                 for ind in range(len(self.user_states))],
                chunksize=100)
        return results

    def run_em_step(self, epoch):
        """Run a single step of expectation maximization"""
        sys.stderr.write("epoch %d, " % epoch)
        # Expectation step
        # Compute (and print) the energies during learning as a diagnostic.
        # These should decrease.
        average_energy = 0.
        # TODO(jascha) this would be faster if user_states was divided into
        # minibatches instead of single students
        results = self.get_results()
        for result in results:
            abilities, El, ind = result
            self.user_states[ind]['abilities'] = abilities.copy()
            average_energy += El / float(len(self.user_states))

        sys.stderr.write("E joint log L + const %f, " % (
                         - average_energy / np.log(2.)))

        # debugging info -- accumulate mean and covariance of abilities vector
        mn_a = 0.
        cov_a = 0.
        for state in self.user_states:
            mn_a += state['abilities'][:, 0].T / float(len(self.user_states))
            cov_a += (state['abilities'][:, 0] ** 2).T / (
                float(len(self.user_states)))
        sys.stderr.write("<abilities> " + str(mn_a))
        sys.stderr.write(", <abilities^2>" + str(cov_a) + ", ")

        # Maximization step
        old_theta_flat = self.theta.flat()
        # Call the minimizer
        theta_flat, L, _ = scipy.optimize.fmin_l_bfgs_b(
            L_dL,
            self.theta.flat(),
            args=(
                self.user_states, self.num_exercises, self.options, self.pool),
            maxfun=self.options.max_pass_lbfgs, m=100)
        self.theta = Parameters(self.options.num_abilities, self.num_exercises,
                                vals=theta_flat)
        if self.options.correct_only:
            self.theta.sigma_time[:] = 1.
            self.theta.W_time[:, :] = 0.

        # Print debugging info on the progress of the training
        sys.stderr.write("M conditional log L %f, " % (-L))
        sys.stderr.write("reg penalty %f, " % (
            self.options.regularization * np.sum(theta_flat ** 2)))
        sys.stderr.write("||couplings|| %f, " % (
            np.sqrt(np.sum(self.theta.flat() ** 2))))
        sys.stderr.write("||dcouplings|| %f\n" % (
            np.sqrt(np.sum((theta_flat - old_theta_flat) ** 2))))

        # Maintain a consistent directional meaning of a
        # high/low ability estimate.  We always prefer higher ability to
        # mean better performance; therefore, we prefer positive couplings.
        # So, compute the sign of the average coupling for each dimension.
        coupling_sign = np.sign(np.mean(self.theta.W_correct[:, :-1], axis=0))
        coupling_sign = coupling_sign.reshape((1, -1))
        # Then, flip ability and coupling sign for dimensions w/ negative mean.
        self.theta.W_correct[:, :-1] *= coupling_sign
        self.theta.W_time[:, :-1] *= coupling_sign
        for user_state in self.user_states:
            user_state['abilities'] *= coupling_sign.T

        # save state as a .npz
        data_to_json(
            theta=self.theta,
            exercise_ind_dict=self.exercise_ind_dict,
            max_time_taken=self.options.max_time_taken,
            outfilename="%s_epoch=%d.json" % (self.options.output, epoch),
            )

        self.write_csv(epoch, self.exercise_ind_dict)

    def write_csv(self, epoch, exercise_ind_dict):
        """Save state as .csv - just for easy debugging inspection"""
        with open("%s_epoch=%d.csv" % (
                self.options.output, epoch), 'w+') as outfile:
            exercises = sorted(
                exercise_ind_dict.keys(),
                key=lambda nm: self.theta.W_correct[exercise_ind_dict[nm], -1])

            outfile.write('correct bias,')
            for coupling_index in range(self.options.num_abilities):
                outfile.write("correct coupling %d, " % coupling_index)
            outfile.write('time bias, ')
            for time_coupling_index in range(self.options.num_abilities):
                outfile.write("time coupling %d," % time_coupling_index)
            outfile.write('time variance, exercise name\n')

            for exercise in exercises:
                exercise_index = exercise_ind_dict[exercise]
                outfile.write(str(
                    self.theta.W_correct[exercise_index, -1]) + ',')
                for index in range(self.options.num_abilities):
                    outfile.write(str(
                        self.theta.W_correct[exercise_index, index]) + ',')
                outfile.write(
                    str(self.theta.W_time[exercise_index, -1]) + ',')
                for time_index in range(self.options.num_abilities):
                    outfile.write(str(
                        self.theta.W_time[exercise_index, time_index]) + ',')
                outfile.write(str(self.theta.sigma_time[exercise_index]) + ',')
                outfile.write(exercise + '\n')


def data_to_json(theta, exercise_ind_dict, max_time_taken, outfilename,
                 slug='Test', title='test parameters', description='parameters'
                 'for an adaptive test'):
    """Convert a set of mirt parameters into a json file and write it"""

    out_data = {
        "engine_class": "MIRTEngine",
        "slug": slug,
        "title": title,
        "description": description,
        # MIRT specific data
        "params": {
            "theta_flat": theta.flat().tolist(),
            "num_abilities": theta.num_abilities,
            "max_length": 15,
            "max_time_taken": max_time_taken,
            "exercise_ind_dict": exercise_ind_dict}
        }

    json_data = json.dumps(out_data, indent=4)

    with open(outfilename, 'w') as outfile:
        outfile.write(json_data)


def json_to_data(filename):
    """Load a json file back into memory as a numpy object"""
    with open(filename, 'r') as data_file:
        data = json.load(data_file)
        data['max_length'] = data['params']['max_length']
        data['max_time_taken'] = data['params']['max_time_taken']
        params = Parameters(data['params']['num_abilities'],
                            len(data['params']['exercise_ind_dict']),
                            np.array(data['params']['theta_flat']),
                            data['params']['exercise_ind_dict'])
        data['params'] = params
    return data
