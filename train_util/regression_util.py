"""This file contains code useful in a variety of settings to do
standard numerical computations.
"""
import numpy as np
import scipy.optimize


def logistic_log_regression(features, correct):
    """Find weights on features given supervision
    """
    # randomly initialize the parameters
    num_features = features.shape[1]
    theta_init = np.random.randn(num_features, 1) / np.sqrt(num_features) / 100
    theta_init = theta_init.reshape((num_features))

    theta = scipy.optimize.optimize.fmin_bfgs(
        logL, theta_init, fprime=dlogLdtheta,
        args=(features, correct),
        maxiter=1000)

    return theta


def sigmoid(X):
    """Compute the sigmoid function of all values in ndarray X."""
    # Bound the values of X
    X[X > 100] = 100
    X[X < -100] = -100
    X = np.nan_to_num(X)

    denominator = 1.0 + np.exp(-1.0 * X)

    d = 1.0 / denominator
    return d


def sigmoid_inv(X):
    """Compute the inverse sigmoid function"""
    d = -np.log(1.0 / X - 1.0)
    d[d > 100] = 100
    d[d < -100] = -100
    return d


def logL(theta, X, Zt):
    """Compute the log likelihood for use in logistic regression.

    Arguments
      theta: the regrssion parameters
      X: the feature data matrix, observations-by-components.
      Zt: target labels (1 or 0)
    """
    theta = theta.reshape((theta.shape[0], 1))
    Zt = Zt.reshape((Zt.shape[0], 1))
    Y = np.dot(X, theta)
    Z = sigmoid(Y)
    pdata = Zt * Z + (1 - Zt) * (1 - Z)

    pdata[pdata < 1e-25] = 1e-25
    pdata[pdata > (1 - 1e-25)] = 1 - 1e-25

    L = np.sum(np.log(pdata)) / np.log(2) / X.shape[0]
    return -L


def dlogLdtheta(theta, X, Zt):
    """Compute the gradient for use in logistic regression.

    Arguments
      theta: the regression parameters
      X: the feature data matrix, observations-by-components.
      Zt: target labels (1 or 0)
    """
    theta = theta.reshape((theta.shape[0], 1))
    Zt = Zt.reshape((Zt.shape[0], 1))
    Zt = np.asarray(Zt)

    Y = np.dot(X, theta)
    Z = np.asarray(sigmoid(Y))

    pdata = Zt * Z + (1 - Zt) * (1 - Z)
    pdata[pdata < 1e-25] = 1e-25
    pdata[pdata > (1 - 1e-25)] = 1 - 1e-25

    dLdY = ((2 * Zt - 1) * Z * (1 - Z)) / pdata
    dL = np.dot(X.T, dLdY)
    dL /= (np.log(2) * X.shape[0])
    return -dL[:, 0]


def quantile(x, q):
    """Generate the q'th quantile of x
    Arguments:
        x: a numpy array
        q: the quantile of interest (a float)
    Returns:
        the element of x closest to the qth quantile.
    """
    if len(x.shape) != 1:
        return None
    x = x.tolist()
    x.sort()
    return x[int((len(x) - 1) * float(q))]


def quantiles(x, quantile_intervals):
    """Generate a series of quantiles
    Arguments:
        x:
            A numpy array of datapoints
        quantile_intervals:
            A list of quantiles to calculate
    Returns:
        a list in which members of the list are the quantiles of x
        corresponding to the quantiles in the list q.
    """
    return [quantile(x, q) for q in quantile_intervals]
