import os
import tempfile
import numpy as np
from itertools import product


category = config['category']
benchmark = config['benchmark']
output_dir = config['output_dir']
intervals = config['intervals']
top_N = config['top_N']
mu_null = config['mu_null']
reps = config['reps']
TOOLS = config['tools']

choice = 'abcdefghijklmnopqrstuvwxyz'
REPLICATES = list(choice[:reps])
SAMPLES = np.arange(intervals).astype(np.str)
SAMPLES = list(map(lambda x: '%s_%s' % x, product(SAMPLES, REPLICATES)))


rule all:
    input:
        output_dir+"confusion_matrix.summary"


rule run_tool:
    input:
        table = output_dir + "table.{sample}.biom",
        metadata = output_dir + "metadata.{sample}.txt"
    output:
        output_dir + "{tool}.{sample}.results"
    run:
        shell("""
        run.py {wildcards.tool}_cmd \
            --table-file {input.table} \
            --metadata-file {input.metadata} \
            --category {category} \
            --output-file {output}
        """)


rule summarize:
    input:
        tables = expand(output_dir + "table.{sample}.biom", sample=SAMPLES),
        results = expand(output_dir + "{tool}.{sample}.results",
                         tool=TOOLS, sample=SAMPLES),
        truths = expand(output_dir + "truth.{sample}.txt", sample=SAMPLES)
    output:
        output_dir + "{tool}.summary"
    run:
        from bayesian_regression.src.evaluate import top_absolute_results
        top_absolute_results(input.tables, input.results,
                             input.truths, output[0], top_N, mu_null)

rule aggregate_summaries:
    input:
        summaries = expand(output_dir + "{tool}.summary", tool=TOOLS),
        metadata = expand(output_dir + "metadata.{sample}.txt", sample=SAMPLES),
        tables = expand(output_dir + "table.{sample}.biom", sample=SAMPLES),
    output:
        output_dir + "confusion_matrix.summary"
    run:
        from bayesian_regression.src.evaluate import aggregate_summaries
        aggregate_summaries(input.summaries, input.tables, input.metadata,
                            benchmark, output[0])


