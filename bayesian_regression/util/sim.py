import pandas as pd
import numpy as np
from numpy.random import RandomState
from scipy.stats import (norm, poisson, multinomial,
                         multivariate_normal, invwishart)
from scipy.sparse.linalg import eigsh

from skbio.stats.composition import ilr_inv, ilr, closure
from skbio.stats.composition import _gram_schmidt_basis, ilr, clr_inv
from sklearn.utils import check_random_state
from bayesian_regression.util.balances import sparse_balance_basis

from gneiss.util import match_tips
from gneiss.cluster import rank_linkage
from biom import Table


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


def random_linkage(n, seed=0):
    """ Generates a tree with random topology.

    Parameters
    ----------
    n : int
        Number of nodes in the tree
    seed : int or np.random.RandomState
        Random seed

    Returns
    -------
    skbio.TreeNode

    Randomly generated tree.

    Examples
    --------
    >>> from gneiss.cluster import random_linkage
    >>> tree = random_linkage(10)
    Notes
    -----
    The nodes will be labeled from 0 to n.
    """
    state = check_random_state(seed)
    index = np.arange(n).astype(np.str)
    x = pd.Series(state.rand(n), index=index)
    t = rank_linkage(x)
    return t


def random_species_tree(time, rate, n_species, n_tries = 100, seed=0):
    """ Generate random tree with sensible branch lengths.

    Parameters
    ----------
    time : int
       Total amount of time simulated
    rate : float
       Speciation rate.  This used to parameterize the
       speciation process as a geometric random variable.
    n_species : int
       Number of species to ultimately simulate
    n_tries : int
       Number of rejection samples before giving up.
    seed : int or np.random.RandomState
        Random seed

    Returns
    -------
    tree : skbio.TreeNode
       A simulated tree.  If no valid trees could be found, this will return None.
       All of the nodes within the tree also have depth and branch length values.
    """
    def _generate_tree(time, rate, n_species, seed):
        state = check_random_state(seed)
        tree = random_linkage(n_species, state)
        for n in tree.preorder(include_self=True):
            # add on edge length.  We will force this to be discrete
            n.length = state.geometric(rate) + 1

        # find depth of the node from the root of the tree
        tree.depth = tree.length
        for n in tree.preorder(include_self=False):
            if n.is_tip():
                n.depth = time
                # correct for the last descendent
                n.length = time - n.parent.depth
            else:
                n.depth = n.parent.depth + n.length
        return tree

    # generate random tree and sanity check to make its not weird
    for _ in range(n_tries):
        tree = _generate_tree(time, rate, n_species, seed)
        valid = True
        for n in tree.postorder():
            if n.length < 0:
                valid=False
                break
        if valid:
            return tree

    if not valid:
        return None


def brownian_tree(tree, intercept, sigma, seed):
    """ Simulates Brownian evolution on a tree.

    Parameters
    ----------
    tree : skbio.TreeNode
        Phylogenetic tree to simulate brownian evolution
    intercept : float
        Initial phenotype value where the brownian motion should begin.
    sigma : float
        Brownian variance at each time point.
    seed : int or np.random.RandomState
        Random seed

    Returns
    -------
    skbio.TreeNode
        Phylogenetic tree with brownian motion simulated at each timepoint
        within each edge.  These values will be stored within the
        `phenotype` attribute within each edge.

    Notes
    -----
    This assumes that all of the edges within the tree have a `length`
    attribute.
    """
    state = check_random_state(seed)
    _tree = tree.copy()
    # simulate brownian evolution
    for n in _tree.preorder():
        n.phenotype = np.zeros(n.length)
        if n.is_root():
            n.phenotype[0] = intercept
        if not n.is_root():
            n.phenotype[0] = n.parent.phenotype[-1]
        for t in range(1, n.length):
            n.phenotype[t] = n.phenotype[t-1] + state.normal(0, sigma)
    return _tree
