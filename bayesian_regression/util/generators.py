from bayesian_regression.util.sim import chain_interactions, ols
from bayesian_regression.util.balances import sparse_balance_basis

from skbio.stats.composition import _gram_schmidt_basis, ilr, clr_inv
from sklearn.utils import check_random_state
from scipy.stats import norm, invwishart
from scipy.sparse.linalg import eigsh
from gneiss.util import match_tips
from biom import Table
import pandas as pd
import numpy as np


def phylogenetic_table(tree, num_samples, spread, alpha=5, link=None, seed=0):
    """ Generates a simulated table of counts accounting for
    Brownian evolution.

    This generates a table of species counts, where the
    species abundances are generated from Gaussian
    distribution.  The means of the Gaussians are
    specified by Brownian evolution.

    The counts are then simulated using a Poisson distribution.

    Parameters
    ----------
    tree : skbio.TreeNode
       Evolutionary tree generated from brownian evolution.
    num_samples : int
       Number of samples
    spread : float
       Width of species unimodal distribution.
    link : callable
       Link function for transforming the species means.
       If it isn't specified, the identity function will
       be used.

    Returns
    -------
    table : biom.Table
        Biom representation of the count table.
    metadata : pd.DataFrame
        DataFrame containing relevant metadata.
    beta : np.array
        Regression parameter estimates.
    theta : np.array
        Bias per sample.
    gamma : np.array
        Bias per feature

    Notes
    -----
    This assumes that there is a phenotype vector
    within every node in the tree
    """
    if link is None:
        link = lambda x: x
    state = check_random_state(seed)
    mu = np.array([link(n.phenotype[-1]) for n in tree.tips()])
    sigma = np.array([spread] * len(mu))
    gradient = np.linspace(min(mu), max(mu), num_samples)
    table = chain_interactions(gradient, mu, sigma)
    n_species = len(list(tree.tips()))
    table = table[:, np.argsort(mu)]
    return _subsample_table(table, tree, gradient, alpha, state)


def band_table(num_samples, num_features, tree=None,
               low=2, high=10, sigma=2, alpha=6, seed=0):
    """ Generates a simulated table of counts.

    Each organism is modeled as a Gaussian distribution.
    The means of the Gaussian are uniformly distributed
    between `low` and `high`.

    The counts are then simulated using a Poisson distribution.

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
    table : biom.Table
        Biom representation of the count table.
    metadata : pd.DataFrame
        DataFrame containing relevant metadata.
    beta : np.array
        Regression parameter estimates.
    theta : np.array
        Bias per sample.
    gamma : np.array
        Bias per feature
    """
    state = check_random_state(seed)

    # measured gradient values for each sample
    gradient = np.linspace(low, high, num_samples)
    # optima for features (i.e. optimal ph for species)
    mu = np.linspace(low, high, num_features)
    sigma = np.array([sigma] * num_features)
    # construct species distributions
    table = chain_interactions(gradient, mu, sigma)
    return _subsample_table(table, tree, gradient, alpha, state)


def _subsample_table(table, tree, gradient, alpha, seed=0):
    # TODO: Add in option for specifying sample biases
    # this will be especially important for simulation rarefaction.
    state = check_random_state(seed)
    # obtain basis required to convert from balances to proportions.
    num_samples, num_features = table.shape
    samp_ids = ['S%d' % i for i in range(num_samples)]
    feat_ids = ['F%d' % i for i in range(num_features)]
    table = pd.DataFrame(table, index=samp_ids, columns=feat_ids)
    if tree is None:
        basis = _gram_schmidt_basis(num_features)
    else:
        basis = sparse_balance_basis(tree)[0].todense()

    # construct balances from gaussian distribution.
    # this will be necessary when refitting parameters later.
    phi = clr_inv(np.array(basis))
    Y = ilr(table, basis=phi)
    X = gradient.reshape(-1, 1)
    X = np.hstack((np.ones(len(X)).reshape(-1, 1), X.reshape(-1, 1)))
    pY, resid, B = ols(Y, X)
    gamma = B[0]
    beta = B[1].reshape(1, -1)
    # parameter estimates
    r = beta.shape[1]
    # Normal distribution to simulate linear regression
    M = np.eye(r)
    # Generate covariance matrix from inverse wishart
    Sigma = invwishart.rvs(df=r+2, scale=M.dot(M.T), random_state=state)
    w, v = eigsh(Sigma, k=2)
    # Low rank covariance matrix
    sim_L = (v @ np.diag(w)).T

    # sample
    y = X.dot(B)
    Ys = np.vstack([state.multivariate_normal(y[i, :], Sigma)
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
    return table, metadata, beta, theta, gamma

