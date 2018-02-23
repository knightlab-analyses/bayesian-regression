import os
import click
import numpy as np
import pandas as pd
from biom import load_table
from skbio.stats.composition import ancom
from skbio.stats.composition import (clr, centralize,
                                     multiplicative_replacement)
from sklearn.cross_decomposition import PLSRegression
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import tempfile
from subprocess import Popen
import io
from gneiss.regression import ols
from skbio.stats.composition import ilr, ilr_inv, clr, _gram_schmidt_basis

from bayesian_regression.util.model import sparse_matmul
from bayesian_regression.util.balances import ilr_to_clr
from bayesian_regression.util.inference import get_batch
from edward.models import Normal, Poisson, PointMass
from scipy.sparse import coo_matrix
import tensorflow as tf
import edward as ed
import subprocess


@click.group()
def run():
    pass

@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def ancom_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    res, _ = ancom(table+1, grouping=metadata[category])
    res = res.rename(
        columns={'W': 'rank',
                 'Reject null hypothesis': 'significant'}
    )
    res.to_csv(output_file, sep='\t')


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def t_test_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cats = metadata[category]
    cs = np.unique(cats)
    def func(x):
        return ttest_ind(*[x[cats == k] for k in cs])
    m, p = np.apply_along_axis(func, axis=0,
                               arr=table.values)
    res = pd.DataFrame(
        {
            'rank': m,
            'significant': p
        }, index=table.columns
    )
    res.to_csv(output_file, sep='\t')


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def mann_whitney_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cats = metadata[category]
    cs = np.unique(cats)
    def func(x):
        try: # catches the scenario where all values are the same.
            return mannwhitneyu(*[x[cats == k] for k in cs])
        except:
            return 0, 1

    m, p = np.apply_along_axis(func, axis=0,
                               arr=table.values)
    res = pd.DataFrame(
        {
            'rank': m,
            'significant': p
        }, index=table.columns
    )
    res.to_csv(output_file, sep='\t')


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def random_forest_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    rf = RandomForestClassifier()
    rf.fit(X=table, y=metadata[category])

    res = pd.DataFrame(
        {
            'rank': rf.feature_importances_,
            'significant': rf.feature_importances_ > 0
        }, index=table.columns
    )
    res.to_csv(output_file, sep='\t')


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def pseudo_balances_cmd(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    balances = pd.DataFrame(ilr(table+1),
                            index=table.index)

    model = ols(category, balances, metadata)
    model.fit()
    ranks = clr(ilr_inv(model._beta.iloc[1]))
    # arbituary threshold for now, since there aren't statistical tests
    # available yet.
    sig = np.abs(ranks) > 1e-1
    print(table.shape)
    res = pd.DataFrame(
        {
            'rank': ranks,
            'significant': sig
        }, index=table.columns
    )
    res.to_csv(output_file, sep='\t')


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def deseq2_cmd(table_file, metadata_file, category, output_file):
    install_setup = ('module load gcc_4.9.1;'
                     'module load R_3.3.0')
    src = os.path.dirname(__file__)
    cmd = ('%s; Rscript %s/DESeq2_nbinom.r '
           '-i %s -m %s -c %s -o %s') % (install_setup, src, table_file,
                                         metadata_file, category, output_file)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()


@run.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def bayesian_balances_cmd(table_file, metadata_file, category, output_file):
    beta1 = 0.9
    beta2 = 0.99
    iterations = 5000
    learning_rate = 1e-1
    batch_size = 1000

    table = load_table(table_file)

    metadata = pd.read_table(metadata_file, index_col=0)
    metadata.columns = [x.replace('-', '_') for x in metadata.columns]
    metadata = metadata.loc[table.ids(axis='sample')]

    # basic filtering parameters
    sample_filter = lambda val, id_, md: (id_ in metadata.index) and np.sum(val) > 1000
    metadata = metadata.loc[table.ids(axis='sample')]
    metadata = metadata.reindex(index=table.ids(axis='sample'))

    G_data = metadata[category].values.reshape(-1, 1)

    sort_f = lambda x: list(metadata.index)
    table = table.sort(sort_f=sort_f, axis='sample') 

    biom_data = table.matrix_data.tocoo().T
    y_data = biom_data.data
    # Because biom got it backwards :/
    y_row = biom_data.row
    y_col = biom_data.col
    N, D = biom_data.shape

    p = 1     # number of covariates

    basis = coo_matrix(_gram_schmidt_basis(D), dtype=np.float32).T

    # Model
    # dummy variables for mini-batch size
    batch_row = tf.placeholder(tf.int32, shape=[batch_size], name='batch_i')
    batch_col = tf.placeholder(tf.int32, shape=[batch_size], name='batch_j')

    # global bias
    alpha = Normal(loc=tf.zeros([]),
                   scale=tf.ones([]),
                   name='alpha')
    # sample bias   
    theta = Normal(loc=tf.zeros([N, 1]),
                   scale=tf.ones([N, 1]),
                   name='theta')
    # species bias
    gamma = Normal(loc=tf.zeros([1, D-1]),
                   scale=tf.ones([1, D-1]) , 
                   name='gamma')

    # dummy variable for gradient                                                                   
    G = tf.placeholder(tf.float32, [N, p], name='G')
    # add bias terms for samples
    Gprime = tf.concat([theta, tf.ones([N, 1]), G], axis=1)

    # Specify regression coefficents
    B = Normal(loc=tf.zeros([p, D-1]),
               scale=tf.ones([p, D-1]), 
               name='B')

    # add bias terms for features
    Bprime = tf.concat([tf.ones([1, D-1]), gamma, B], axis=0)

    # Convert basis to SparseTensor
    psi = tf.SparseTensor(
        indices=np.mat([basis.row, basis.col]).transpose(),
        values=basis.data,
        dense_shape=basis.shape)

    # clr transform coefficients first                                                               
    V = ilr_to_clr(Bprime, psi)
    # retrieve entries selected by index
    eta = sparse_matmul(
        Gprime, V, 
        row_index=batch_row, col_index=batch_col
    )
    # obtain counts                                          
    Y = Poisson( rate=tf.exp(eta + alpha), name='Y' ) 
    
    # Inference
    qalpha = PointMass(
        params=tf.Variable(tf.random_normal([])) ,
        name='qalpha')

    qgamma = PointMass(
        params=tf.Variable(tf.random_normal([1, D-1])) ,
        name='qgamma')

    qtheta = PointMass(
        params=tf.Variable(tf.random_normal([N, 1])),
        name='qtheta')

    qB = PointMass(
        params=tf.Variable(tf.random_normal([p, D-1])) ,
        name='qB')

    # a placeholder for the microbial counts
    # since we will be manually feeding it into the inference via minibatch SGD
    Y_ph = tf.placeholder(tf.float32, shape=[batch_size], name='Y_placeholder')

    inference = ed.MAP({
        theta: qtheta,
        alpha: qalpha,
        gamma: qgamma,
        B: qB},
        data={G: G_data, Y: Y_ph}
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
                                       beta1=beta1,    
                                       beta2=beta2)
    # adds checks for nans
    #tf.add_check_numerics_ops()
    sess = ed.get_session()
    inference.initialize(n_iter=iterations,
                         optimizer=optimizer,
                         n_print=100)

    # initialize all tensorflow variables
    tf.global_variables_initializer().run()

    losses = np.array([0.] * inference.n_iter)
    errors = np.array([0.] * inference.n_iter)
    data = table.matrix_data.tocoo().T

    for i in range(inference.n_iter):
        # get batches
        idx_row, idx_col, idx_data = get_batch(M=batch_size, Y=biom_data)

        info_dict = inference.update(
            feed_dict={batch_row: idx_row.astype(np.int32), 
                       batch_col: idx_col.astype(np.int32), 
                       Y_ph: idx_data})
        inference.print_progress(info_dict)

    est_B = sess.run(qB.mean())
    ranks = np.ravel(est_B @ basis.T)

    res = pd.DataFrame(
        {
            'rank': ranks,
        }, index=table.ids(axis='observation')
    )
    res.to_csv(output_file, sep='\t')


if __name__ == "__main__":
    run()
