import tensorflow as tf
import numpy as np


def sparse_matmul(A, B, row_index, col_index):
    """ Sparse matrix multiplication.

    This will try to evaluate the following product

    A[i] @ B[j]

    where i, j are the row and column indices specified in `indices`.

    Parameters
    ----------
    A : tf.Tensor
       Left 2D tensor
    B : tf.Tensor
       Right 2D tensor
    row_idx : tf.Tensor
       Row indexes to access in A
    col_idx : tf.Tensor
       Column indexes to access in B


    Returns
    -------
    tf.Tensor
       Result stored in a sparse tensor format, where the values
       are derived from A[i] @ B[j] where i, j are the row and column
       indices specified in `indices`.
    """
    A_flat = tf.gather(A, row_index, axis=0)
    B_flat = tf.transpose(tf.gather(B, col_index, axis=1))
    values = tf.reduce_sum(tf.multiply(A_flat, B_flat), axis=1)
    return values


