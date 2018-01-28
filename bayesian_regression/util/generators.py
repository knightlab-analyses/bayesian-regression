from bayesian_regression.util.sim import chain_interactions, ols
from bayesian_regression.util.balances import sparse_balance_basis

from skbio.stats.composition import _gram_schmidt_basis, ilr
from sklearn.utils import check_random_state
from scipy.stats import norm, invwishart
from scipy.sparse.linalg import eigsh
from biom import Table
import pandas as pd
import numpy as np


def band_table(num_samples, num_features, tree=None,
               low=2, high=10, sigma=2, alpha=6, seed=0):
    """ Generates a simulated table of counts.

    Each organism is modeled as a Gaussian distribution.  Then counts
    are simulated using a Poisson distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to simulate
    num_features : int
        Number of features to simulate
    tree : skbio.TreeNode
        Tree used as a scaffold for the ilr transform.
        If None, then the gram_schmidt_basis will be used.
    low : float
        Smallest gradient value.
    high : float
        Largest gradient value.
    sigma : float
        Variance of each species distribution
    alpha : int
        Global count bias.  This bias is added to every cell in the matrix.
    seed : int or np.random.RandomState
        Random seed

    Returns
    -------
    biom.Table
        Biom representation of the count table.
    pd.DataFrame
        DataFrame containing relevant metadata.
    beta : np.array
        Regression parameter estimates.
    theta : np.array
        Bias per sample.
    """
    state = np.random.RandomState(seed)

    # measured gradient values for each sample
    gradient = np.linspace(low, high, num_samples)
    # optima for features (i.e. optimal ph for species)
    mu = np.linspace(low, high, num_features)
    sigma = np.array([sigma] * num_features)
    # construct species distributions
    table = chain_interactions(gradient, mu, sigma)
    samp_ids = ['S%d' % i for i in range(num_samples)]
    feat_ids = ['F%d' % i for i in range(num_features)]

    # obtain basis required to convert from balances to proportions.
    if tree is None:
        basis = _gram_schmidt_basis(num_features)
    else:
        basis = sparse_balance_basis(tree)[0].todense()

    # construct balances from gaussian distribution.
    # this will be necessary when refitting parameters later.
    Y = ilr(table)
    X = gradient.reshape(-1, 1)
    X = np.hstack((np.ones(len(X)).reshape(-1, 1), X.reshape(-1, 1)))
    pY, resid, beta = ols(Y, X)

    # parameter estimates
    beta0 = np.ravel(beta[0, :]).reshape(-1, 1)
    beta1 = np.ravel(beta[1, :]).reshape(-1, 1)
    r = len(beta0)

    # Normal distribution to simulate linear regression
    M = np.eye(r)
    # Generate covariance matrix from inverse wishart
    sigma = invwishart.rvs(df=r+2, scale=M.dot(M.T), random_state=state)
    w, v = eigsh(sigma, k=2)
    # Low rank covariance matrix
    sim_L = (v @ np.diag(w)).T

    # sample
    y = X.dot(beta)
    Ys = np.vstack([state.multivariate_normal(y[i, :], sigma)
                    for i in range(y.shape[0])])
    Yp = Ys @ basis

    # calculate bias terms
    theta = -np.log(np.exp(Yp).sum(axis=1)) + alpha

    # multinomial sample the entries
    #table = np.vstack(multinomial(nd, Yp[i, :]) for i in range(y.shape[0]))

    # poisson sample the entries
    table = np.vstack(state.poisson(np.exp(Yp[i, :] + theta[i]))
                      for i in range(y.shape[0])).T

    table = Table(table, feat_ids, samp_ids)
    metadata = pd.DataFrame({'G': gradient}, index=samp_ids)
    return table, metadata, beta, theta

