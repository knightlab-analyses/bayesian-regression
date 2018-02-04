import unittest
from bayesian_regression.util.generators import band_table
from biom import Table
import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np
import pandas as pd


class TestGenerator(unittest.TestCase):
    def setUp(self):
        pass

    def test_band_table(self):
        res_table, res_md, res_beta, res_theta, res_gamma = band_table(5, 6)
        mat = np.array(
            [[161.0, 88.0, 26.0, 4.0, 0.0],
             [185.0, 144.0, 40.0, 4.0, 4.0],
             [28.0, 39.0, 156.0, 45.0, 12.0],
             [7.0, 64.0, 50.0, 81.0, 56.0],
             [0.0, 29.0, 83.0, 217.0, 194.0],
             [0.0, 0.0, 19.0, 54.0, 127.0]]
        )

        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']

        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [2., 4., 6., 8., 10.]},
                              index=samp_ids)
        exp_beta = np.array(
            [[-0.28284271, -0.48989795, -0.69282032, -0.89442719, -1.09544512]]
        )

        exp_theta = np.array(
            [2.23148138, 3.64417845, 3.9674706, 3.32461839, 2.31151262]
        )

        exp_gamma = np.array(
            [0.79195959, 1.89427207, 3.41791359, 5.36656315, 7.74114548]
        )

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        npt.assert_allclose(exp_beta, res_beta)
        npt.assert_allclose(exp_theta, res_theta)


if __name__ == "__main__":
    unittest.main()
