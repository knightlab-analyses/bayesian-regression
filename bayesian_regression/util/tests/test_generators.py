import unittest
from bayesian_regression.util.generators import band_table, phylogenetic_table
from bayesian_regression.util.sim import random_species_tree, brownian_tree

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

    def test_phylogenetic_table(self):
        n_features = 10
        n_samples = 6
        time = 50
        state = check_random_state(3)
        spread = 1
        rate = 0.2
        intercept = 0
        sigma = 0.1
        tree = random_species_tree(time, rate, n_features, seed=state)
        tree = brownian_tree(tree, intercept, sigma, seed=state)
        res = phylogenetic_table(
            tree, n_samples, spread, alpha=5, seed=state)
        res_table, res_metadata, res_beta, res_theta, res_gamma = res

        data = np.array([
            [30.0, 20.0, 13.0, 17.0, 11.0, 4.0],
            [7.0, 3.0, 16.0, 20.0, 19.0, 11.0],
            [39.0, 31.0, 20.0, 22.0, 25.0, 8.0],
            [6.0, 9.0, 15.0, 5.0, 3.0, 4.0],
            [20.0, 48.0, 18.0, 9.0, 3.0, 10.0],
            [25.0, 3.0, 9.0, 29.0, 21.0, 31.0],
            [3.0, 5.0, 14.0, 6.0, 8.0, 16.0],
            [20.0, 29.0, 23.0, 19.0, 12.0, 6.0],
            [4.0, 1.0, 21.0, 23.0, 44.0, 53.0],
            [4.0, 10.0, 7.0, 7.0, 10.0, 20.0]
        ])

        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4',
                    'F5', 'F6', 'F7', 'F8', 'F9']
        samp_ids = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
        exp_table = Table(data, feat_ids, samp_ids)

        exp_metadata = pd.DataFrame({'G': [-0.766315,-0.332438, 0.101438,
                                           0.535314, 0.969190, 1.403066]},
                                    index=samp_ids)
        exp_beta = np.array([[1.32238554, 0.45549609, 0.95300833,
                              0.36203256, 0.17803183, 0.13406303,
                              0.33116103, 0.20000772, 0.43526815]])

        exp_theta = np.array([[2.39074196],
                              [2.32902944],
                              [2.62669704],
                              [2.50259685],
                              [2.33829864],
                              [2.32058252]])
        exp_gamma = np.array([-0.2823803, 0.15355296, -0.59341646,
                              0.18475205, 0.00976021, -0.02889122,
                              -0.33423157, -0.04734037, -0.47674288])

        self.assertEqual(res_table, exp_table)
        pdt.assert_frame_equal(res_metadata, exp_metadata)
        npt.assert_allclose(res_beta, exp_beta)
        npt.assert_allclose(res_theta, exp_theta)
        npt.assert_allclose(res_gamma, exp_gamma, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
