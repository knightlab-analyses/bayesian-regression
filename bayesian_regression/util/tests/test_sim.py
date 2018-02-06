import numpy as np
import numpy.testing as npt
import unittest
from bayesian_regression.util.sim import (
    chain_interactions, ols, random_linkage,
    random_species_tree, brownian_tree
)
from sklearn.utils import check_random_state


class TestSim(unittest.TestCase):

    def test_chain_interactions(self):

        gradient = np.linspace(0, 10, 5)
        mu = np.array([4, 6])
        sigma = np.array([1, 1])

        exp = np.array([[1.33830226e-04, 6.07588285e-09],
                        [1.29517596e-01, 8.72682695e-04],
                        [2.41970725e-01, 2.41970725e-01],
                        [8.72682695e-04, 1.29517596e-01],
                        [6.07588285e-09, 1.33830226e-04]])
        res = chain_interactions(gradient, mu, sigma)
        npt.assert_allclose(exp, res)

    def test_ols(self):
        y = np.array([[1, 2, 3, 4, 5],
                      [2, 4, 6, 8, 10]]).T
        x = np.array([[1, 1, 1, 1, 1],
                      [1, 2, 3, 4, 5]]).T
        exp_beta = np.array([[0, 0],
                             [1, 2]])
        py, _, res_beta = ols(y, x)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-7, rtol=1e-4)
        npt.assert_allclose(y, py)

    def test_random_linkage(self):
        t = random_linkage(10, seed=0)
        exp = (
            '((7:0.035944879859474754,8:0.035944879859474754)'
            'y1:0.15902486846990777,((9:0.023589743237510452,'
            '(4:0.006966205961893901,6:0.006966205961893901)'
            'y5:0.01662353727561655)y3:0.07471735610142118,'
            '(1:0.06480041117839454,((0:0.0019651604652139443,'
            '3:0.0019651604652139443)y7:0.03677504008830565,'
            '(2:0.02156536849750612,5:0.02156536849750612)'
            'y8:0.017174832056013473)y6:0.026060210624874944)'
            'y4:0.03350668816053709)y2:0.09666264899045089)y0;\n'
        )
        self.assertEqual(str(t), exp)

    def test_random_species_tree(self):
        n_species = 6
        time = 15
        rate = 0.9
        tree = random_species_tree(time, rate, n_species, seed=1)
        exp = '(1:13,((0:9,3:9)y2:2,(2:9,(4:7,5:7)y4:2)y3:2)y1:2)y0:2;\n'
        self.assertEqual(str(tree), exp)

    def test_random_species_tree2(self):
        n_species = 7
        time = 20
        rate = 0.3
        tree = random_species_tree(time, rate, n_species, seed=3)
        exp = '((2:16,6:16)y1:2,((4:11,5:11)y3:4,(1:11,(0:7,3:7)y5:4)y4:4)y2:3)y0:2;\n'
        self.assertEqual(str(tree), exp)

    def test_random_species_tree_invalid(self):
        n_species = 7
        time = 20
        rate = 0.01
        tree = random_species_tree(time, rate, n_species, seed=3)
        self.assertEqual(tree, None)

    def test_brownian_tree(self):
        n_species = 1000
        time = 200
        rate = 0.2
        state = check_random_state(3)
        sigma = 0.2
        intercept = 0
        tree = random_species_tree(time, rate, n_species, seed=state)
        res_tree = brownian_tree(tree, intercept, sigma, seed=state)
        # make sure that the last observed phenotype for all of the tips are
        # normally distributed around the intercept.
        ps = np.array([t.phenotype[-1] for t in res_tree.tips()])
        print(np.mean(ps))
        npt.assert_allclose(np.mean(ps), 0, atol=2e-1, rtol=2e-1)


if __name__ == "__main__":
    unittest.main()
