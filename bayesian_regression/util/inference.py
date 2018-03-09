import numpy as np


def get_batch(M, Y, p=0.5):
    """ Get's batch data

    Parameters
    ----------
    M : int
        batch size
    Y : scipy.sparse.coo_matrix
        Scipy sparse matrix in COO-format.

    Returns
    -------
    batch_row : np.array
        Selected rows
    batch_col : np.array
        Selected columns
    batch_data : np.array
        Selected data
    """
    y_data = Y.data
    y_row = Y.row
    y_col = Y.col
    # get positive sample
    positive_idx = np.random.choice(len(y_data), round(M * p))
    positive_row = y_row[positive_idx]
    positive_col = y_col[positive_idx]
    positive_data = y_data[positive_idx]

    # store all of the positive (i, j) coords
    idx = np.vstack((y_row, y_col)).T
    idx = set(map(tuple, idx.tolist()))

    # get negative sample
    N, D = Y.shape
    negative_row = np.zeros(round(M * (1-p)))
    negative_col = np.zeros(round(M * (1-p)))
    negative_data = np.zeros(round(M * (1-p)))
    for k in range(int(M * (1-p))):
        i, j = np.random.randint(N), np.random.randint(D)
        while (i, j) in idx:
            i, j = np.random.randint(N), np.random.randint(D)
        negative_row[k] = i
        negative_col[k] = j
    batch_row = np.hstack((positive_row, negative_row))
    batch_col = np.hstack((positive_col, negative_col))
    batch_data = np.hstack((positive_data, negative_data))
    return batch_row, batch_col, batch_data
