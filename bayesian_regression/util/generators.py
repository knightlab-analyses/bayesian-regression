from bayesian_regression.util.sim import chain_interactions, ols
from bayesian_regression.util.balances import sparse_balance_basis

from skbio.stats.composition import _gram_schmidt_basis, ilr, clr_inv
from sklearn.utils import check_random_state
from scipy.stats import norm, invwishart
from scipy.sparse.linalg import eigsh
from biom import Table
from biom.util import biom_open
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import coo_matrix


def deposit(table, metadata, feature_metadata, it, rep, output_dir):
    """ Writes down tables, metadata and feature metadata into files.

    Parameters
    ----------
    table : biom.Table
        Biom table
    metadata : pd.DataFrame
        Dataframe of sample metadata
    feature_metadata : pd.DataFrame
        Dataframe of features metadata
    it : int
        iteration number
    rep : int
        repetition number
    output_dir : str
        output directory
    """
    choice = 'abcdefghijklmnopqrstuvwxyz'
    output_table = "%s/table.%d_%s.biom" % (
        output_dir, it, choice[rep])
    output_md = "%s/metadata.%d_%s.txt" % (
        output_dir, it, choice[rep])
    output_fmd = "%s/truth.%d_%s.txt" % (
        output_dir, it, choice[rep])
    with biom_open(output_table, 'w') as f:
        table.to_hdf5(f, generated_by='moi')
    metadata.to_csv(output_md, sep='\t', index_label='#SampleID')
    feature_metadata.to_csv(output_fmd, sep='\t')

def softplus(x):
    return np.log(1 + np.exp(x))

def random_poisson_model(num_samples, num_features,
                         tree=None,
                         low=2, high=10,
                         alpha_mean=0,
                         alpha_scale=5,
                         theta_mean=0,
                         theta_scale=5,
                         gamma_mean=0,
                         gamma_scale=5,
                         kappa_mean=0,
                         kappa_scale=5,
                         beta_mean=0,
                         beta_scale=5,
                         seed=0):
    """ Generates a table using a random poisson regression model.

    Parameters
    ----------
    num_samples : int
        Number of samples
    num_features : int
        Number of features
    basis : np.array
        Orthonormal contrast matrix.
    low : float
        Smallest gradient value.
    high : float
        Largest gradient value.
    alpha_mean : float
        Mean of alpha prior  (for global bias)
    alpha_scale: float
        Scale of alpha prior  (for global bias)
    theta_mean : float
        Mean of theta prior (for sample bias)
    theta_scale=5
        Scale of theta prior (for sample bias)
    gamma_mean : float
        Mean of gamma prior (for feature bias)
    gamma_scale=5
        Scale of gamma prior (for feature bias)
    kappa_mean : float
        Mean of kappa prior (for overdispersion)
    kappa_scale=5
        Scale of kappa prior (for overdispersion)
    beta_mean : float
        Mean of beta prior (for regression coefficients)
    beta_scale=5
        Scale of beta prior (for regression coefficients)

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
    kappa : np.array
        Dispersion rates of counts per sample.
    """
    # generate all of the coefficient using the random poisson model
    state = check_random_state(seed)
    alpha = state.normal(alpha_mean, alpha_scale)
    theta = state.normal(theta_mean, theta_scale, size=(num_samples, 1))
    beta = state.normal(beta_mean, beta_scale, size=num_features-1)
    gamma = state.normal(gamma_mean, gamma_scale, size=num_features-1)
    kappa = state.normal(kappa_mean, kappa_scale, size=num_features)

    if tree is None:
        basis = coo_matrix(_gram_schmidt_basis(num_features), dtype=np.float32)
    else:
        basis = sparse_balance_basis(tree)[0]

    G = np.linspace(low, high, num_samples)
    N, D = num_samples, num_features
    G_data = np.vstack((np.ones(N), G)).T
    B = np.vstack((gamma, beta))
    V = G_data @ B @ basis + theta + alpha
    #kprime = kappa @ basis

    mu = np.vstack((state.normal(V[:, i] + alpha, softplus(kappa[i]))
                    for i in range(num_features))).T
    table = np.vstack(
        state.poisson(np.exp(mu[i, :]))
        for i in range(mu.shape[0])
    ).T

    samp_ids = ['S%d' % i for i in range(num_samples)]
    feat_ids = ['F%d' % i for i in range(num_features)]
    balance_ids = ['L%d' % i for i in range(num_features-1)]

    table = Table(table, feat_ids, samp_ids)
    metadata = pd.DataFrame({'G': G.ravel()}, index=samp_ids)
    beta = pd.DataFrame({'beta': beta.ravel()}, index=balance_ids)
    gamma = pd.DataFrame({'gamma': gamma.ravel()}, index=balance_ids)
    kappa = pd.DataFrame({'kappa': kappa.ravel()}, index=feat_ids)
    theta = pd.DataFrame({'theta': theta.ravel()}, index=samp_ids)
    return table, metadata, basis, alpha, beta, theta, gamma, kappa


def band_table(num_samples, num_features, tree=None,
               mu=5, sigma=2, low=2, high=10,
               spread=2, feature_bias=1, alpha=2,
               dispersion_shape=5, dispersion_rate=0.1,
               seed=0):
    """ Generates a simulated band table of counts.

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
    mu : float
        Random effects distribution of the means of the species.
    sigma : float
        Variance of the random effects distribution.
    low : float
        Smallest gradient value.
    high : float
        Largest gradient value.
    spread : float
        Variance of each species distribution
    feature_bias : float
        Parameter for exponential distribution for feature biases.
    alpha : int
        Global count bias.  This bias is added to every cell in the matrix.
    dispersion_shape : float
        Shape parameter of gamma prior for dispersion parameter.
    dispersion_rate : float
        Rate parameter of gamma prior for dispersion parameter.
    seed : int or np.random.RandomState
        Random seed

    Returns
    -------
    table : biom.Table
        Biom representation of the count table.
    metadata : pd.DataFrame
        DataFrame containing relevant sample metadata.
    feature metadata : pd.DataFrame
        DataFrame containing relevant feature metadata.
    beta : np.array
        Regression parameter estimates.
    theta : np.array
        Bias per sample.
    gamma : np.array
        Bias per feature
    dispersion : np.array
        Dispersion rates of counts per sample.
    """
    state = check_random_state(seed)

    # measured gradient values for each sample
    gradient = np.linspace(low, high, num_samples)
    # optima for features (i.e. optimal ph for species)
    mus = state.normal(mu, sigma, num_features)
    mus = np.sort(mus)
    spread = np.array([spread] * num_features)
    # construct species distributions
    gamma = np.exp(state.normal(0, feature_bias,
                                size=num_features))
    table = chain_interactions(gradient, mus, spread, gamma)
    ans = _subsample_table(table, tree, gradient,
                           alpha=alpha, feature_bias=feature_bias,
                           dispersion_shape=dispersion_shape,
                           dispersion_rate=dispersion_rate,
                           seed=state)
    table, metadata, beta, theta, gamma, dispersion = ans

    feature_metadata = pd.DataFrame(
        {
            'mu': mus,
        }, index=table.ids(axis='observation')
    )

    return table, metadata, feature_metadata, beta, theta, gamma, dispersion


def block_table(num_samples, num_features,
                mu_num=6, mu_null=5, mu_denom=4, sigma=0.5,
                pi1=0.25, pi2=0.25, low=4, high=6,
                spread=1, feature_bias=1, alpha=6,
                dispersion_shape=5, dispersion_rate=0.1,
                seed=0):
    """ Generates a simulated block table of counts.

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
    mu_num : float
        Random effects distribution of the means of the numerator species.
    mu_null : float
        Random effects distribution of the means of the null species.
        These species don't have a preference for either environment
    mu_denom : float
        Random effects distribution of the means of the denominator species.
    sigma : float
        Standard deviation of both random effects distributions.
    pi1 : float
        Proportion of features in the numerator distribution.
    pi2 : float
        Proportion of features in the denominator distribution.
    low : float
        Smallest gradient value.
    high : float
        Largest gradient value.
    spread : float
        Variance of each species distribution
    feature_bias : float
        Parameter for exponential distribution for feature biases.
    alpha : int
        Global count bias.  This bias is added to every cell in the matrix.
    dispersion_shape : float
        Shape parameter of gamma prior for dispersion parameter.
    dispersion_rate : float
        Rate parameter of gamma prior for dispersion parameter.
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
    dispersion : np.array
        Dispersion rates of counts per sample.
    """
    state = check_random_state(seed)
    # measured gradient values for each sample
    gradient = np.linspace(low, high, num_samples)
    # optima for features (i.e. optimal ph for species)
    #mu = np.linspace(low, high, num_features)
    mu_num_hat = state.normal(
        mu_num, sigma, size=int(round(pi1 * num_features)))
    mu_denom_hat = state.normal(
        mu_denom, sigma, size=int(round((pi2) * num_features)))
    mu_null_hat = state.normal(
        mu_null, sigma, size=int(round((1-pi1-pi2) * num_features)))

    mu = np.hstack((mu_num_hat, mu_null_hat, mu_denom_hat))
    spread = np.array([spread] * num_features)
    # construct species distributions
    gamma = state.normal(0, feature_bias,
                         size=num_features)
    table = chain_interactions(gradient, mu, spread, gamma)
    ans = _subsample_table(
        table, None, gradient=gradient,
        feature_bias=feature_bias, alpha=alpha,
        dispersion_shape=dispersion_shape,
        dispersion_rate=dispersion_rate,
        seed=state)

    table, metadata, beta, theta, gamma, dispersion = ans
    mid = (mu_num + mu_denom) / 2
    metadata['G'] = np.round(sigmoid(gradient - mid))
    feature_metadata = pd.DataFrame(
        {
            'mu': mu,
            'class': ([1] * len(mu_num_hat) +
                      [0] * len(mu_null_hat) +
                      [-1] * len(mu_denom_hat))
        }, index=table.ids(axis='observation')
    )
    return table, metadata, feature_metadata, beta, theta, gamma, dispersion


def _subsample_table(table, tree, gradient, alpha, feature_bias,
                     dispersion_shape, dispersion_rate, seed=0):
    state = check_random_state(seed)
    # obtain basis required to convert from balances to proportions.
    num_samples, num_features = table.shape
    dispersion = state.gamma(dispersion_shape, dispersion_rate,
                             size=num_samples)
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
    gamma = B[0] # + dgamma
    beta = B[1].reshape(1, -1)
    B = np.vstack((gamma, beta))

    # sample
    y = X @ B
    Yp = y @ basis
    theta = -np.log(np.exp(Yp).sum(axis=1))
    # multinomial sample the entries
    #table = np.vstack(multinomial(nd, Yp[i, :]) for i in range(y.shape[0]))

    # poisson sample the entries with overdispersion
    mu = np.vstack(
        np.exp(Yp[i, :] + theta[i] + alpha)
        for i in range(y.shape[0])
    )
    table = np.vstack(
        state.poisson(mu[i, :])
        for i in range(y.shape[0])
    ).T

    # negative binomial sampling
    # table = np.vstack(
    #     state.poisson(
    #         state.gamma(dispersion[i], dispersion[i] * mu[i, :]))
    #     for i in range(y.shape[0])
    # ).T

    table = Table(table, feat_ids, samp_ids)
    metadata = pd.DataFrame({'G': gradient}, index=samp_ids)
    return table, metadata, beta, theta, gamma, dispersion
