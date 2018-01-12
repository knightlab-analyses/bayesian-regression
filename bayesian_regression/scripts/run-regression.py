import click
import numpy as np
import pandas as pd
from biom import load_table
from skbio.stats.composition import (clr, centralize,
                                     multiplicative_replacement,
                                     _gram_schmidt_basis)

from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import tempfile
from subprocess import Popen
import io
from patsy import dmatrix

import tensorflow as tf
import edward as ed
from edward.models import Normal, Multinomial, Exponential, InverseGamma
from edward.models import Bernoulli, Normal, Empirical, PointMass
import pickle

@click.group()
def run_regression():
    pass

@run_regression.command()
def ols_pseudo_cmd(table_file, metadata_file, formula, output_file):
    pass

@run_regression.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--formula',
              help='Formula specifying regression equation')
@click.option('--output-file',
              help='Saved tensorflow model.')
def bayes_mult_cmd(table_file, metadata_file, formula, output_file):

    #metadata = _type_cast_to_float(metadata.copy())
    metadata = pd.read_table(metadata_file, index_col=0)
    G_data = dmatrix(formula, metadata, return_type='dataframe')
    table = load_table(table_file)

    # basic filtering parameters
    soil_filter = lambda val, id_, md: id_ in metadata.index
    read_filter = lambda val, id_, md: np.sum(val) > 10
    #sparse_filter = lambda val, id_, md: np.mean(val > 0) > 0.1
    sample_filter = lambda val, id_, md: np.sum(val) > 1000

    table = table.filter(soil_filter, axis='sample')
    table = table.filter(sample_filter, axis='sample')
    table = table.filter(read_filter, axis='observation')
    #table = table.filter(sparse_filter, axis='observation')
    print(table.shape)
    y_data = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                          index=table.ids(axis='sample'),
                          columns=table.ids(axis='observation'))

    y_data, G_data = y_data.align(G_data, axis=0, join='inner')

    psi = _gram_schmidt_basis(y_data.shape[1])
    G_data = G_data.values
    y_data = y_data.values
    N, D = y_data.shape
    p = G_data.shape[1] # number of covariates
    r = G_data.shape[1] # rank of covariance matrix

    psi = tf.convert_to_tensor(psi, dtype=tf.float32)
    n = tf.convert_to_tensor(y_data.sum(axis=1), dtype=tf.float32)

    # hack to get multinomial working
    def _sample_n(self, n=1, seed=None):
        # define Python function which returns samples as a Numpy array
        def np_sample(p, n):
            return multinomial.rvs(p=p, n=n, random_state=seed).astype(np.float32)

        # wrap python function as tensorflow op
        val = tf.py_func(np_sample, [self.probs, n], [tf.float32])[0]
        # set shape from unknown shape
        batch_event_shape = self.batch_shape.concatenate(self.event_shape)
        shape = tf.concat(
            [tf.expand_dims(n, 0), tf.convert_to_tensor(batch_event_shape)], 0)
        val = tf.reshape(val, shape)
        return val
    Multinomial._sample_n = _sample_n


    # dummy variable for gradient
    G = tf.placeholder(tf.float32, [N, p])

    b = Exponential(rate=1.0)
    B = Normal(loc=tf.zeros([p, D-1]), 
               scale=tf.ones([p, D-1]) )

    # Factorization of covariance matrix
    # http://edwardlib.org/tutorials/klqp
    l = Exponential(rate=1.0)
    L = Normal(loc=tf.zeros([p, D-1]), 
               scale=tf.ones([p, D-1]) )
    z = Normal(loc=tf.zeros([N, p]), 
               scale=tf.ones([N, p]))

    # Cholesky trick to get multivariate normal
    v = tf.matmul(G, B) + tf.matmul(z, L)

    # get clr transformed values
    eta = tf.matmul(v, psi)

    Y = Multinomial(total_count=n, logits=eta)


    T = 100000  # the number of mixin samples from MCMC sampling

    qb = PointMass(params=tf.Variable(tf.random_normal([])))
    qB = PointMass(params=tf.Variable(tf.random_normal([p, D-1])))
    qz = Empirical(params=tf.Variable(tf.random_normal([T, N, p])))
    ql = PointMass(params=tf.Variable(tf.random_normal([])))
    qL = PointMass(params=tf.Variable(tf.random_normal([p, D-1])))

    # Imputation
    inference_z = ed.SGLD(
        {z: qz}, 
        data={G: G_data, Y: y_data, B: qB, L: qL}
    )

    # Maximization
    inference_BL = ed.MAP(
        {B: qB, L: qL, b: qb, l: ql}, 
        data={G: G_data, Y: y_data, z: qz}
    )

    inference_z.initialize(step_size=1e-10)
    inference_BL.initialize(n_iter=1000)


    sess = ed.get_session()
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()
    for i in range(inference_BL.n_iter):
        inference_z.update()  # e-step
        # will need to compute the expectation of z

        info_dict = inference_BL.update() # m-step
        inference_BL.print_progress(info_dict)

    save_path = saver.save(sess, output_file)
    print("Model saved in file: %s" % save_path)
    pickle.dump({'qB': sess.run(qB.mean()),
                 'qL': sess.run(qL.mean()),
                 'qz': sess.run(qz.mean())},
                open(output_file + '.params.pickle', 'wb')
    )


if __name__ == "__main__":
    run_regression()
