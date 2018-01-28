import tensorflow as tf
import numpy as np
from gneiss.util import NUMERATOR, DENOMINATOR
from scipy.sparse import coo_matrix


def ilr_to_clr(X, psi):
    """ Sparse ilr to clr transform.

    This is also known as the log_softmax transform with an extra
    rotation transform specified by the orthonormal matrix `psi`.

    Parameters
    ----------
    X : tf.Tensor
       Tensor of logits with dimensions N x D-1 to be transformed
    psi : tf.SparseTensor
       SparseTensor of dimensions D x D-1 used to transform `X`
       into a N x D dimesional tensor of logits.

    Returns
    -------
    tf.Tensor
       A N x D dimensional tensor of logits.  This also refered to as
       clr coordinates.
    """
    return tf.transpose(
        tf.sparse_tensor_dense_matmul(psi, tf.transpose(X))
    )


def sparse_balance_basis(tree):
    """ Calculates sparse representation of an ilr basis from a tree.

    This computes an orthonormal basis specified from a bifurcating tree.

    Parameters
    ----------
    tree : skbio.TreeNode
        Input bifurcating tree.  Must be strictly bifurcating
        (i.e. every internal node needs to have exactly 2 children).
        This is used to specify the ilr basis.

    Returns
    -------
    scipy.sparse.coo_matrix
       The ilr basis required to perform the ilr_inv transform.
       This is also known as the sequential binary partition.
       Note that this matrix is represented in clr coordinates.
    nodes : list, str
        List of tree nodes indicating the ordering in the basis.

    Raises
    ------
    ValueError
        The tree doesn't contain two branches.

    """
    # this is inspired by @wasade in
    # https://github.com/biocore/gneiss/pull/8
    t = tree.copy()
    D = len(list(tree.tips()))
    # calculate number of tips under each node
    for n in t.postorder(include_self=True):
        if n.is_tip():
            n._tip_count = 1
        else:
           try:
               left, right = n.children[NUMERATOR], n.children[DENOMINATOR],
           except:
               raise ValueError("Not a strictly bifurcating tree.")
           n._tip_count = left._tip_count + right._tip_count

    # calculate k, r, s, t coordinate for each node
    left, right = t.children[NUMERATOR], t.children[DENOMINATOR],
    t._k, t._r, t._s, t._t = 0, left._tip_count, right._tip_count, 0
    for n in t.preorder(include_self=False):
        if n.is_tip():
            n._k, n._r, n._s, n._t = 0, 0, 0, 0

        elif n == n.parent.children[NUMERATOR]:
            n._k = n.parent._k
            n._r = n.children[NUMERATOR]._tip_count
            n._s = n.children[DENOMINATOR]._tip_count
            n._t = n.parent._s + n.parent._t
        elif n == n.parent.children[DENOMINATOR]:
            n._k = n.parent._r + n.parent._k
            n._r = n.children[NUMERATOR]._tip_count
            n._s = n.children[DENOMINATOR]._tip_count
            n._t = n.parent._t
        else:
            raise ValueError("Tree topology is not correct.")

    # navigate through tree to build the basis in a sparse matrix form
    value = []
    row, col = [], []
    nodes = []
    i = 0

    for n in t.levelorder(include_self=True):

        if n.is_tip():
            continue

        for j in range(n._k, n._k + n._r):
            row.append(i)
            col.append(D-1-j)
            A = np.sqrt(n._s / (n._r * (n._s + n._r)))

            value.append(A)

        for j in range(n._k + n._r, n._k + n._r + n._s):
            row.append(i)
            col.append(D-1-j)
            B = -np.sqrt(n._r / (n._s * (n._s + n._r)))

            value.append(B)
        i += 1
        nodes.append(n.name)

    basis = coo_matrix((value, (row, col)), shape=(D-1, D))

    return basis, nodes


def match_tips(table, tree):
    """ Returns the contingency table and tree with matched tips.

    Sorts the columns of the contingency table to match the tips in
    the tree.  The ordering of the tips is in post-traversal order.
    If the tree is multi-furcating, then the tree is reduced to a
    bifurcating tree by randomly inserting internal nodes.
    The intersection of samples in the contingency table and the
    tree will returned.

    Parameters
    ----------
    table : biom.Table
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.

    Returns
    -------
    biom.Table :
        Subset of the original contingency table with the common features.
    skbio.TreeNode :
        Sub-tree with the common features.
    """
    tips = [x.name for x in tree.tips()]
    common_tips = set(tips) & set(table.ids(axis='observation'))


    _tree = tree.shear(names=list(common_tips))

    def filter_uncommon(val, id_, md):
        return id_ in common_tips
    _table = table.filter(filter_uncommon, axis='observation', inplace=False)

    _tree.bifurcate()
    _tree.prune()
    sort_f = lambda x: [n.name for n in _tree.tips()]
    _table = _table.sort(sort_f=sort_f, axis='observation')
    return _table, _tree
