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
from skbio.stats.composition import ilr, ilr_inv, clr


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
def metagenomeseq_cmd(table_file, metadata_file, category, output_file):
    cmd = (
        'source activate qiime;'
        'differential_abundance.py -i %s '
        '-o %s -m %s '
        '-a metagenomeSeq_fitZIG '
        '-c %s -x %s -y %s;'
        'source activate differential_benchmarks'
    )
    metadata = pd.read_table(metadata_file, index_col=0)
    tmp = '%s.tmp' % table_file
    values = list(metadata[category].value_counts())
    cmd = cmd % (table_file, tmp, metadata_file, category, values[0], values[1])
    proc = Popen(cmd, shell=True)
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
def deseq2_cmd(table_file, metadata_file, category, output_file):
    pass


if __name__ == "__main__":
    run()
