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

import numpy as np
import warnings


class Parameters(object):
    """
    Holds the parameters for a MIRT model.  Also used to hold the gradients
    for each parameter during training.
    """
    def __init__(self, num_abilities, num_exercises, vals=None):
        """ vals is a 1d array holding the flattened parameters """
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
            nn = num_exercises * (num_abilities + 1)
            self.W_correct = vals[:nn].copy().reshape((-1, num_abilities + 1))
            # the couplings to time taken (+1 for bias unit)
            self.W_time = vals[nn:2 * nn].copy().reshape((-1,
                num_abilities + 1))
            # the standard deviation for the response time Gaussian
            self.sigma_time = vals[2 * nn:].reshape((-1))

    def flat(self):
        """Returns a concatenation of the parameters for saving."""
        return np.concatenate((self.W_correct.ravel(), self.W_time.ravel(),
                               self.sigma_time.ravel()))


def sigmoid(X):
    """Compute the sigmoid function element-wise on X.

    Args:
        X: An ndarray of any shape.

    Returns:
        An ndarray of the same shape as X where the elements are the
        sigmoid of the elements of X.
    """
    X[X > 100] = 100
    X[X < -100] = -100
    X = np.nan_to_num(X)
    den = 1. + np.exp(-1. * X)
    den = np.nan_to_num(den)
    den[den == 0] = 2
    d = 1. / den
    return d


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


def conditional_probability_correct(abilities, theta, exercises_ind):

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
    W_correct = theta.W_correct[exercises_ind, :]
    Z = sigmoid(np.dot(W_correct, abilities))
    Z = np.reshape(Z, Z.size)  # flatten to 1-d ndarray
    return Z


def conditional_energy_data(
        abilities, theta, exercises_ind, correct, log_time_taken):
    """Calculate the probability of the observed responses "correct" to
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

    TODO(eliana): Rename to condtional energy data and remove the E paramenter,
        treating it as always true?
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

    num_abilities = theta.num_abilities

    # initialize abilities using prior
    if abilities_init is None:
        abilities = np.random.randn(num_abilities, 1)
    else:
        abilities = abilities_init
    # calculate the energy for the initialization state
    E_abilities = 0.5 * np.dot(abilities.T, abilities) + np.sum(
        conditional_energy_data(abilities, theta,
            exercises_ind, correct, log_time_taken))

    assert np.isfinite(E_abilities)

    sample_chain = []
    for _ in range(num_steps):
        # generate the proposal state
        proposal = abilities + sampling_epsilon * np.random.randn(
            num_abilities, 1)

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
    mean_sample_abilities = np.mean(sample_chain, 0).reshape(num_abilities, 1)
    stdev = np.std(sample_chain, 0).reshape(num_abilities, 1)

    return abilities, E_abilities, mean_sample_abilities, stdev
