import numpy as np
from scipy.stats import norm, poisson, multinomial, multivariate_normal
from numpy.random import RandomState
from skbio.stats.composition import ilr_inv, ilr, closure


def chain_interactions(gradient, mu, sigma):
    """
    This generates an urn simulating a chain of interacting species.

    This commonly occurs in the context of a redox tower, where
    multiple species are distributed across a gradient.

    Parameters
    ----------
    gradient: array_like
       Vector of values associated with an underlying gradient.
    mu: array_like
       Vector of means.
    sigma: array_like
       Vector of standard deviations.
    rng: np.random.RandomState
       Numpy random state.

    Returns
    -------
    np.array
       A matrix of real-valued positive abundances where
       there are `n` rows and `m` columns where `n` corresponds
       to the number of samples along the `gradient` and `m`
       corresponds to the number of species in `mus`.
    """
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma[i])
          for i in range(len(mu))]
    return np.vstack(xs).T


def ols(Y, X):
    """ Ordinary Least Squares

    Performs linear regression between matrices Y and X.
    Specifically this solves the optimization problem

    min_B || Y - XB ||_2^2

    Parameters
    ----------
    X : np.array
        Matrix of dimensions n and p
    Y : np.array
        Matrix of dimensions n and D

    Returns
    -------
    pY : np.array
        Predicted Y matrix with dimensions n and D
    resid : np.array
        Matrix of residuals
    beta : np.array
        Learned parameter matrix of
    """
    n, p = X.shape
    inv = np.linalg.pinv(np.dot(X.T, X))
    cross = np.dot(inv, X.T)
    beta = np.dot(cross, Y)
    pY = np.dot(X, beta)
    resid = (Y - pY)
    return pY, resid, beta
