import unittest
import numpy as np
import numpy.testing as npt
from bayesian_regression.util.balances import sparse_balance_basis
from skbio import TreeNode
from skbio.util import get_data_path
from scipy.sparse import coo_matrix


def assert_coo_allclose(res, exp, rtol=1e-7, atol=1e-7):
    res_data = np.vstack((res.row, res.col, res.data)).T
    exp_data = np.vstack((exp.row, exp.col, exp.data)).T

    # sort by row and col
    res_data = res_data[res_data[:, 1].argsort()]
    res_data = res_data[res_data[:, 0].argsort()]
    exp_data = exp_data[exp_data[:, 1].argsort()]
    exp_data = exp_data[exp_data[:, 0].argsort()]

    npt.assert_allclose(res_data, exp_data, rtol=rtol, atol=atol)


class TestSparseBalances(unittest.TestCase):

    def test_sparse_balance_basis_base_case(self):
        tree = u"(a,b)r;"
        t = TreeNode.read([tree])
        exp_basis = coo_matrix(
            np.array(
                [[np.sqrt(1. / 2),
                  -np.sqrt(1. / 2)]]
            )[:, ::-1]
        )
        exp_keys = [t.name]
        res_basis, res_keys = sparse_balance_basis(t)

        assert_coo_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_sparse_balance_basis_unbalanced(self):
        tree = u"((a,b)c, d)r;"
        t = TreeNode.read([tree])
        exp_basis = coo_matrix(
            np.array(
                [[np.sqrt(2. / 3), -np.sqrt(1. / 6), -np.sqrt(1. / 6)],
                 [0, np.sqrt(1. / 2), -np.sqrt(1. / 2)]]
            )[:, ::-1]
        )
        exp_keys = [t.name, t[0].name]
        res_basis, res_keys = sparse_balance_basis(t)

        assert_coo_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)


if __name__ == "__main__":
    unittest.main()
