import matplotlib.pyplot as plt


def plot_brownian(tree, phenotype_attr='phenotype', depth_attr='depth',
                  length_attr='length',
                  ax=None, **kwargs):
    """ Plots Brownian motion on a tree

    Parameters
    ----------
    tree : skbio.TreeNode
        Phylogenetic tree to simulate brownian evolution
    phenotype_attr : str
        Values where phenotype information was stored.  This was the
        attribute that was simulated by brownian motion
    depth_attr : str
        Attribute indicating the depth of the node from the root.
    length_attr : str
        Attribute indicating the branch length of the node.
    ax : matplot.pyplot.Axes
        Matplotlib axes object to plot the brownian motion.
    **kwargs : dict
        Attributes to be passed in for plotting.

    Returns
    -------
    matplot.pyplot.Axes
        Rendered axes object.

    Notes
    -----
    This assumes that the `phenotype`, `depth` and `length` has been
    encoded within every node in the `tree`.
    """
    # plot brownian motion
    if ax is None:
        fig, ax = plt.subplots()
    linewidth = 0.5
    for n in tree.preorder():
        values = getattr(n, phenotype_attr)
        depth = getattr(n, depth_attr)
        length = getattr(n, length_attr)

        if not n.is_tip():
            x = np.arange(n.depth - n.length, n.depth + 1)
            y = np.hstack((values, values[-1]))
        else:
            x = np.arange(depth - n.length, depth)
            y = values
        ax.plot(x, y, **kwargs)
