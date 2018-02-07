import os
import shutil
import subprocess
import numpy as np
from snakemake import snakemake
from bayesian_regression.util.generators import band_table, block_table, deposit
import yaml

# snakemake config
config_file = 'config.yaml'
workflow_type = 'local'
local_cores = 1
jobs = 1
force = False
snakefile = '../Snakefile'
dry_run = True
output_dir = 'benchmark_results/'
quiet=True
keep_logger=True

# simulation parameters
num_samples = 50
num_features = 200
#mu_num = 8            # mean of the numerator taxa
mu_null = 0            # mean of the common taxa
#mu_denom = 2          # mean of the denominator taxa
max_diff = 3           # largest separation between the normals
min_diff = 0.5         # smallest separation between the normals
sigma = 1              # variance of the random effects distribution
pi1 = 0.25             # percentage of the species
pi2 = 0.25             # percentage of the species
low = -3               # lower value for spectrum
high = 3               # higher value for the spectrum
spread = 3             # variance of unimodal species distribution
feature_bias = 1       # species bias
alpha = 6              # global sampling depth
seed = None            # random seed

# benchmark parameters
top_N = int((pi1 + pi2) * num_features)     # top hits to evaluate
intervals = 3
benchmark = 'effect_size'
category = 'G'
reps = 2
tools = ['t_test', 'mann_whitney']

# generate config file
data = {'category': category,
        'benchmark': benchmark,
        'intervals': intervals,
        'output_dir': output_dir,
        'reps': reps,
        'tools': tools,
        'mu_null': mu_null,
        'top_N': top_N}
with open(config_file, 'w') as yfile:
    yaml.dump(data, yfile, default_flow_style=False)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

# generate simulations
for i, ef in enumerate(np.linspace(max_diff, min_diff, intervals)):
    for r in range(reps):
        mu_num = mu_null + ef
        mu_denom = mu_null - ef
        res = block_table(num_samples, num_features,
                          mu_num, mu_null, mu_denom, sigma, pi1, pi2,
                          low, high, spread, feature_bias,
                          alpha, seed=seed)
        table, metadata, feature_metadata, beta, theta, gamma = res
        metadata['effect_size'] = ef
        deposit(table, metadata, feature_metadata, i, r, output_dir)


cmd = ' '.join([
    'snakemake ',
    '--snakefile %s ' % snakefile,
    '--local-cores %s ' % local_cores,
    '--jobs %s ' % jobs,
    '--configfile %s ' % config_file,
])
proc = subprocess.Popen(cmd, shell=True)
proc.wait()
