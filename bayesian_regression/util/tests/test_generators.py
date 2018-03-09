import unittest
from bayesian_regression.util.generators import band_table, block_table

from biom import Table
import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


class TestGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def test_band_table(self):
        res = band_table(5, 6, alpha=4)
        res_table, res_md, res_fmd, res_beta, res_theta, res_gamma, res_d = res
        mat = np.array([
            [8.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 0.0, 0.0],
            [0.0, 10.0, 2.0, 0.0, 6.0],
            [0.0, 0.0, 3.0, 0.0, 8.0],
            [0.0, 1.0, 0.0, 6.0, 7.0],
            [0.0, 0.0, 9.0, 1.0, 0.0]]
        )

        exp_beta = np.array([[-0.48699685, -0.51737237, -0.70588861,
                              -0.59306809, -0.6546415]])
        exp_theta = np.array([-4.79269502, -3.10904263, -1.86554414,
                              -2.00271511, -2.76255814])
        exp_gamma = np.array([[3.1040167, 2.59893833, 4.4744781,
                               4.35601705, 4.91757565]])
        exp_d = np.array([0.85659586, 0.57930342, 0.60328846,
                          0.42852916, 0.84240309])

        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']
        exp_fmd = pd.DataFrame({'mu': [3.045444, 5.800314, 6.957476,
                                       8.528105, 8.735116, 9.481786]},
                               index=feat_ids)
        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [2., 4., 6., 8., 10.]},
                              index=samp_ids)

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        pdt.assert_frame_equal(exp_fmd, res_fmd)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-6)
        npt.assert_allclose(exp_theta, res_theta, atol=1e-6)
        npt.assert_allclose(exp_gamma, res_gamma, atol=1e-6)
        npt.assert_allclose(exp_d, res_d, atol=1e-6)


    def test_block_table(self):
        res = block_table(10, 8, sigma=1,
                          mu_num=7, mu_null=5, mu_denom=3,
                          spread=1, low=3, high=7,
                          alpha=7, feature_bias=1,
                          dispersion_shape=10., dispersion_rate=0.05
        )
        res_table, res_md, res_fmd, res_beta, res_theta, res_gamma, res_d = res

        mat = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 59.0],
                [0.0, 0.0, 5.0, 0.0, 14.0, 0.0, 0.0, 13.0, 579.0, 212.0],
                [0.0, 0.0, 1.0, 0.0, 7.0, 74.0, 91.0, 27.0, 75.0, 6.0],
                [9.0, 0.0, 242.0, 21.0, 388.0, 12.0, 1.0, 43.0, 0.0, 1.0],
                [0.0, 2.0, 8.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [27.0, 79.0, 141.0, 1.0, 7.0, 6.0, 2.0, 2.0, 0.0, 0.0],
                [10.0, 0.0, 34.0, 1.0, 39.0, 0.0, 0.0, 609.0, 168.0, 2.0]
            ]
        )

        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4',
                    'S5', 'S6', 'S7', 'S8', 'S9']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

        exp_fmd = pd.DataFrame(
            {'class': np.array([1, 1, 0, 0, 0, 0, -1, -1]),
             'mu': np.array([8.76405235, 7.40015721, 6.86755799,
                             4.02272212, 5.95008842, 4.84864279,
                             3.97873798, 5.2408932])},
            index=feat_ids)

        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [0., 0., 0., 0., 0.,
                                     1., 1., 1., 1., 1.]},
                              index=samp_ids)

        exp_beta = np.array([[0.9644195, 0.9916733, 3.16491905,
                              0.72764693, 1.59959892, 2.15728319,
                              0.68762395]])
        exp_theta = np.array([-5.82269392, -5.27523152, -4.37335198,
                              -3.63409942, -3.09887319, -2.60261712,
                              -2.56496994, -2.78891745, -2.79177075,
                              -3.83987724])
        exp_gamma = np.array([[ -7.8977583, -7.19184908, -18.64689094,
                                -4.50511347,  -9.53372303, -12.1336633,
                                -4.72704718]])

        exp_disp = np.array([0.53708289, 0.45565115, 0.73737355,
                             0.34883391, 0.54432881, 0.27874347,
                             0.40048065, 0.43338424, 0.49073395,
                             0.57684584])

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        pdt.assert_frame_equal(exp_fmd, res_fmd)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-6)
        npt.assert_allclose(exp_theta, res_theta, atol=1e-6)
        npt.assert_allclose(exp_gamma, res_gamma, atol=1e-6)
        npt.assert_allclose(exp_disp, res_d, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
