0;95;0cimport os
import shutil
import subprocess
import numpy as np
from bayesian_regression.util.generators import band_table, block_table, deposit
import yaml

# snakemake config
config_file = 'params.yaml'
workflow_type = 'torque'
local_cores = 4
cores = 8
jobs = 30
force = True
snakefile = '../../Snakefile'
dry_run = False
output_dir = 'factorial_results_small/'
quiet=False
keep_logger=True
cluster_config = 'cluster.json'
latency_wait = 120
profile = os.path.dirname(__file__)
restart_times = 1

# simulation parameters
regenerate_simulations = False
num_samples = 50
num_features = 300
#mu_num = 8            # mean of the numerator taxa
mu_null = 0            # mean of the common taxa
#mu_denom = 2          # mean of the denominator taxa
max_diff = 4           # largest separation between the normals
min_diff = 0.5         # smallest separation between the normals
min_alpha = 3          # smallest sequencing depth
max_alpha = 9          # largest sequencing depth
min_bias = 0.1         # smallest feature bias variance
max_bias = 3           # largest feature bias variance
min_null = 0.9         # smallest proportion of null species
max_null = 0.1         # largest proportion of null species
min_ratio = 1          # smallest differential species ratio
max_ratio = 5          # largest differential species ratio
sigma = 0.5            # variance of the random effects distribution
pi1 = 0.1              # percentage of the species
pi2 = 0.3              # percentage of the species
low = -4               # lower value for spectrum
high = 4               # higher value for the spectrum
spread = 2             # variance of unimodal species distribution
feature_bias = 1       # species bias
alpha = 6              # global sampling depth
seed = None            # random seed

# benchmark parameters
top_N = 10     # top hits to evaluate
intervals = 3
benchmark = 'factorial'
category = 'G'
reps = 3
tools = ['t_test', 'mann_whitney', 'deseq2', 'random_forest', 'ancom', 'pseudo_balances', 'bayesian_balances']

sample_ids = []
if regenerate_simulations:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    
    # iterate across different sequencing depths
    for i1, al in enumerate(np.linspace(max_alpha, min_alpha, intervals)):
        # iterate across different feature biases
        for i2, bi in enumerate(np.linspace(max_bias, min_bias, intervals)):
            # iterate across different null species proportions
            for i3, nu in enumerate(np.linspace(max_null, min_null, intervals)):    
                # iterate across different species 1 proportions
                for i4, ra in enumerate(np.linspace(max_ratio, min_ratio, intervals)):        
                    # iterate across different effect sizes
                    for i5, ef in enumerate(np.linspace(max_diff, min_diff, intervals)):
                        pi1 = (1 - nu) * (ra / (ra + 1))
                        pi2 = (1 - nu) * (1 / (ra + 1))
                        feature_bias = bi
                        alpha = al
                        sample_id = i1 + i2 * 10 + i3 * 100 + i4 * 1000 + i5 * 10000

                        for r in range(reps):
                            mu_num = mu_null + ef
                            mu_denom = mu_null - ef
                            res = block_table(num_samples, num_features,
                                              mu_num, mu_null, mu_denom, sigma, pi1, pi2,
                                              low, high, spread, feature_bias,
                                              alpha, seed=seed)
                            table, metadata, feature_metadata, beta, theta, gamma = res
                            metadata['effect_size'] = ef
                            metadata['pi1'] = pi1
                            metadata['pi2'] = pi2
                            metadata['feature_bias'] = feature_bias
                            metadata['alpha'] = alpha
                            deposit(table, metadata, feature_metadata, 
                                    sample_id, r, output_dir)

                        sample_ids.append(sample_id)

    # generate config file
    data = {'category': category,
            'benchmark': benchmark,
            'intervals': intervals,
            'output_dir': output_dir,
            'samples': sample_ids,
            'reps': reps,
            'tools': tools,
            'mu_null': mu_null,
            'top_N': top_N}
    with open(config_file, 'w') as yfile:
        yaml.dump(data, yfile, default_flow_style=False)

if workflow_type == 'local':
    cmd = ' '.join([
        'snakemake --verbose --nolock',
        '--snakefile %s ' % snakefile,
        '--local-cores %s ' % local_cores,
        '--jobs %s ' % jobs,
        '--configfile %s ' % config_file,
        '--latency-wait %d' % latency_wait
    ])

elif workflow_type == 'torque':
    eo = '-e {cluster.error} -o {cluster.output} '

    cluster_setup = '\" qsub %s\
                     -l nodes=1:ppn={cluster.n} \
                     -l mem={cluster.mem}gb \
                     -l walltime={cluster.time}\" ' % eo

    cmd = ' '.join(['snakemake --verbose --nolock',                    
                    '--snakefile %s ' % snakefile,
                    '--local-cores %s ' % local_cores,
                    '--cores %s ' % cores,
                    '--jobs %s ' % jobs,
                    '--restart-times %d' % restart_times,
                    '--keep-going',
                    '--cluster-config %s ' % cluster_config,
                    '--cluster %s '  % cluster_setup,
                    '--configfile %s ' % config_file,
                    '--latency-wait %d' % latency_wait
                ])    

elif workflow_type == "profile":
    cmd = ' '.join(['snakemake --nolock',
                    '--snakefile %s ' % snakefile,
                    '--cluster-config %s ' % cluster_config,
                    '--profile %s '  % profile,
                    '--configfile %s ' % config_file
                    ]
                   )

else:
    ValueError('Incorrect workflow specified:', workflow_type)

print(cmd)
proc = subprocess.Popen(cmd, shell=True)
proc.wait()
