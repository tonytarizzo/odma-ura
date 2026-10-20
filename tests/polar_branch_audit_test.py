"""Algebraic and source-law checks for broad approximation screening; no performance assertions."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import gammaln, softmax

from tests.polar_approximation_audit import least_squares_gauge, posterior_tails
from tests.polar_branch_audit import checked_marginals, configuration_beam, connected_tails, feature_setup
from tests.polar_branch_audit import feature_three_tails, frequency_numerators, region_tails, select_regions
from tests.polar_branch_audit import merged_recursion
from tests.polar_posterior_test import count_states, exact_posterior


class BranchAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phi = np.random.default_rng(351).normal(size=(3, 5))
        cls.phi /= np.linalg.norm(cls.phi, axis=0)
        cls.y, cls.sigma2 = np.array([.5, -.3, .2]), 1.7

    def test_configuration_beam_exact_limit_and_conditional_retained_law(self):
        for source, moment in itertools.product(("iid", "distinct"), (False, True)):
            states, p = exact_posterior(self.phi, self.y, 3, self.sigma2, source)
            exact = posterior_tails(states, p, 3)
            tails, retained, work = configuration_beam(self.phi, self.y, 3, self.sigma2, source, 100, moment)
            np.testing.assert_allclose(tails, exact, atol=2e-14)
            self.assertEqual(len(retained), len(states))
            narrow, retained, work = configuration_beam(self.phi, self.y, 3, self.sigma2, source, 4, moment)
            mask = np.any(np.all(states[:, None, :] == retained[None, :, :], axis=2), axis=1)
            conditional = posterior_tails(states[mask], p[mask] / p[mask].sum(), 3)
            np.testing.assert_allclose(narrow, conditional, atol=2e-14)
            self.assertLessEqual(work["retained_configurations"], 4)

    def test_region_convolution_matches_declared_positive_model_and_exact_limit(self):
        for source, k in itertools.product(("iid", "distinct"), (3, 4)):
            gram, linear = self.phi.T @ self.phi / self.sigma2, self.phi.T @ self.y / self.sigma2
            gauge = least_squares_gauge(gram)
            gram, linear = gram - gauge[:, None] - gauge[None, :], linear - k * gauge
            for groups in ([[0, 1], [2, 3, 4]], [[0, 1, 2, 3, 4]]):
                tails, _ = region_tails(gram, linear, groups, k, source)
                states = count_states(5, k, source)
                retained = np.zeros_like(gram)
                for group in groups:
                    retained[np.ix_(group, group)] = gram[np.ix_(group, group)]
                score = states @ linear - np.einsum("bi,ij,bj->b", states, retained, states) / 2
                if source == "iid":
                    score -= gammaln(states + 1).sum(1)
                expected = posterior_tails(states, softmax(score), k)
                np.testing.assert_allclose(tails, expected, atol=3e-14)
                if len(groups) == 1:
                    states, p = exact_posterior(self.phi, self.y, k, self.sigma2, source)
                    np.testing.assert_allclose(tails, posterior_tails(states, p, k), atol=3e-14)

    def test_marked_merging_exact_limit_and_accumulated_error_bound(self):
        for source in ("iid", "distinct"):
            states, p = exact_posterior(self.phi, self.y, 3, self.sigma2, source)
            exact = posterior_tails(states, p, 3)
            for spacing in (0., .5, 2., 4.):
                tails, work = merged_recursion(self.phi, self.y, 3, self.sigma2, source, spacing)
                self.assertAlmostEqual(tails.sum(), 3., places=12)
                self.assertLessEqual(abs(tails - exact).max(), work["algebraic_tv_bound"] + 2e-12)
                if not spacing:
                    np.testing.assert_allclose(tails, exact, atol=3e-14)
                if source == "distinct":
                    np.testing.assert_array_equal(tails[:, 1:], 0.)

    def test_region_selector_preserves_partition_and_size_budget(self):
        weight = abs(self.phi.T @ self.phi)
        np.fill_diagonal(weight, 0.)
        for size in (1, 2, 3, 5):
            groups, work = select_regions(weight, size)
            self.assertEqual(sorted(sum(groups, [])), list(range(5)))
            self.assertLessEqual(max(map(len, groups)), size)
            self.assertGreater(work, 0)

    def test_connected_complete_order_matches_physical_law_and_checks_exclusions(self):
        for source, k, gauged in itertools.product(("iid", "distinct"), (3, 4), (False, True)):
            gram, linear = self.phi.T @ self.phi / self.sigma2, self.phi.T @ self.y / self.sigma2
            if gauged:
                gauge = least_squares_gauge(gram)
                gram, linear = gram - gauge[:, None] - gauge[None, :], linear - k * gauge
            tails, meta = connected_tails(gram, linear, k, source, k)
            self.assertIsNotNone(tails, meta)
            states, p = exact_posterior(self.phi, self.y, k, self.sigma2, source)
            np.testing.assert_allclose(tails, posterior_tails(states, p, k), atol=2e-12)
        tails, meta = connected_tails(gram, linear, k, "distinct", 1)
        self.assertIsNone(tails)
        self.assertEqual(meta["status"], "distinct_source_violation")

    def test_low_rank_k3_contraction_matches_positive_kernel_enumeration(self):
        for source, dimension, degree in itertools.product(("iid", "distinct"), (1, 3), (2, 4)):
            q, features, signs, meta = feature_setup(self.phi, self.y, self.sigma2, dimension, degree)
            kernel = (features * signs) @ features.T
            self.assertGreater(kernel.min(), 0.)
            tails, status = feature_three_tails(q, features, signs, source)
            self.assertIsNotNone(tails, status)
            states = count_states(5, 3, source)
            log_kernel = np.log(kernel)
            score = states @ np.log(q) + (np.einsum("bi,ij,bj->b", states, log_kernel, states)
                                         - states @ log_kernel.diagonal()) / 2
            score -= gammaln(states + 1).sum(1)
            np.testing.assert_allclose(tails, posterior_tails(states, softmax(score), 3), atol=3e-12)
            self.assertEqual(meta["features"], math.comb(meta["dimension"] + degree, degree))
            if dimension == 1 and degree == 2:
                accurate, _ = feature_three_tails(q, features, signs, source, precision=60)
                np.testing.assert_allclose(accurate, tails, atol=2e-14)

    def test_fourier_marked_coefficients_match_exact_posterior_under_gaussian_quadrature(self):
        phi, y, variance = np.array([[-1., -.3, .4, 1.]]), np.array([.2]), .9
        nodes, weights = np.polynomial.hermite.hermgauss(100)
        omega = (nodes * math.sqrt(2 / variance))[:, None]
        for source in ("iid", "distinct"):
            numerator = np.einsum("f,fij->ij", weights / math.sqrt(math.pi), frequency_numerators(phi, y, 3, source, omega))
            tails, status = checked_marginals(numerator, 3, source)
            self.assertIsNotNone(tails, status)
            states, p = exact_posterior(phi, y, 3, variance, source)
            np.testing.assert_allclose(tails, posterior_tails(states, p, 3), atol=2e-13)

    def test_invalid_signed_outputs_are_not_repaired(self):
        for numerator, source in ((np.array([[-1., 0., 0.]]), "iid"),
                                  (np.array([[.9, -.1, .2]]), "iid"),
                                  (np.array([[.4, .2, .4]]), "distinct")):
            tails, status = checked_marginals(numerator, 2, source)
            self.assertIsNone(tails)
            self.assertNotEqual(status, "valid_marginals")


if __name__ == "__main__":
    unittest.main()
