"""Finite certificates for the complete triangle posterior and its exact compression limits."""

import itertools
import math
import unittest
from collections import Counter

import numpy as np
from scipy.special import softmax

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from tests import polar_posterior_test as reference
from tests.polar_posterior_test import exact_posterior
from tests.polar_three_user import PositivePolarPair, edge_deletion_diagnostic, three_user_counts, triangle_quotient


class ThreeUserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.PolarPosteriorTests.setUpClass()
        cls.fixture = reference.PolarPosteriorTests
        cls.generator = cls.fixture.base.polar.encode(np.eye(2, dtype=np.uint8)).T

    def test_positive_pair_rows_and_contractions_including_disjoint_patterns(self):
        f = self.fixture
        disjoint = np.arange(16).reshape(2, 8)
        energy = np.random.default_rng(822).dirichlet(np.ones(8), size=2)
        signs = 1. - 2. * f.base.polar.encode(message_bits(np.arange(4), 2))
        phi = np.zeros((16, 8))
        for p in range(2):
            phi[np.ix_(disjoint[p], np.arange(4 * p, 4 * (p + 1)))] = np.sqrt(energy[p, :, None]) * signs.T
        for pos, e, wave in ((f.base.positions, f.energy, f.phi), (disjoint, energy, phi)):
            for sampling in ("iid", "distinct"):
                operator = PositivePolarPair(pos, e, self.generator, .3, sampling)
                pair = np.exp(-(wave.T @ wave) / .3)
                if sampling == "distinct":
                    np.fill_diagonal(pair, 0)
                actual = np.array([operator.row(w) for w in range(wave.shape[1])])
                np.testing.assert_allclose(actual, pair, rtol=1e-13, atol=1e-14)
                weights = np.random.default_rng(22).dirichlet(np.ones(wave.shape[1]))
                np.testing.assert_allclose(operator(weights), pair @ weights, rtol=1e-13, atol=1e-14)

    def test_positive_representation_keeps_a_diagonal_lost_by_one_plus_f(self):
        f = self.fixture
        operator = PositivePolarPair(f.base.positions, f.energy, self.generator, .02, "iid")
        self.assertEqual(1 + np.expm1(-50.), 0.)
        self.assertGreater(operator.row(0)[0], 0.)
        np.testing.assert_allclose(operator.row(0)[0], math.exp(-50), rtol=1e-13, atol=0)
        basis = np.eye(16)[0]
        np.testing.assert_allclose(operator(basis)[0], math.exp(-50), rtol=1e-13, atol=0)

    def test_complete_count_laws_trace_and_expected_counts(self):
        f = self.fixture
        for sampling in ("iid", "distinct"):
            for sigma2 in (.05, .3, 1.):
                operator = PositivePolarPair(f.base.positions, f.energy, self.generator, sigma2, sampling)
                nu = softmax(f.phi.T @ f.y / sigma2)
                actual, z = three_user_counts(nu, operator.row, operator)
                states, posterior = exact_posterior(f.phi, f.y, 3, sigma2, sampling)
                expected = np.array([posterior @ (states == j) for j in range(4)]).T
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-14)
                pair = np.array([operator.row(w) for w in range(16)])
                a = np.sqrt(nu[:, None] * nu[None, :]) * pair
                cube = a @ a @ a
                np.testing.assert_allclose(z, np.trace(cube), rtol=1e-12)
                np.testing.assert_allclose(actual @ np.arange(4), 3 * cube.diagonal() / z, atol=2e-14)

    def test_degenerate_count_laws_and_distinct_exclusion(self):
        q, _ = three_user_counts(np.ones(1), lambda w: np.array([.2]), lambda x: .2 * x)
        np.testing.assert_allclose(q, [[0, 0, 0, 1]], atol=1e-15)
        pair = np.array([[0., .4, 2.], [.4, 0., 1.3], [2., 1.3, 0.]])
        q, _ = three_user_counts(np.array([.1, .2, .7]), lambda w: pair[w], lambda x: pair @ x)
        np.testing.assert_allclose(q, np.tile([0, 1, 0, 0], (3, 1)), atol=1e-15)

    def test_pattern_quotients_preserve_every_count_marker_and_pattern_multiplicity(self):
        f = self.fixture
        nu = softmax(f.phi.T @ f.y / f.sigma2)
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(f.phi, f.y, 3, f.sigma2, sampling)
            for marked in range(16):
                polynomial = np.zeros(4)
                for patterns in itertools.combinations_with_replacement(range(4), 3):
                    multiplicity = 6 // math.prod(math.factorial(c) for c in Counter(patterns).values())
                    local, _ = triangle_quotient(f.base.positions, f.energy, self.generator, nu, patterns,
                                                 f.sigma2, sampling, marked)
                    polynomial += multiplicity * local
                expected = [posterior @ (states[:, marked] == j) for j in range(4)]
                np.testing.assert_allclose(polynomial / polynomial.sum(), expected, atol=2e-14)

    def test_selected_payload_code_joint_quotient_against_all_assignments(self):
        base = ODMAPolarBaseline(12, 256, 3201, prefix_bits=6, code_length=32, crc_bits=8)
        positions = base.positions[:3]
        generator = base.polar.encode(np.eye(6, dtype=np.uint8)).T
        energy = np.random.default_rng(753).dirichlet(np.ones(32), size=3)
        nu = np.random.default_rng(811).dirichlet(np.ones(192))
        signs = 1. - 2. * base.polar.encode(message_bits(np.arange(64), 6))
        wave = np.zeros((192, 256))
        for p in range(3):
            wave[np.ix_(np.arange(p * 64, (p + 1) * 64), positions[p])] = signs * np.sqrt(energy[p])
        pair = np.exp(-(wave @ wave.T) / .3)
        payloads = np.array(list(itertools.product(range(64), repeat=3)))
        for patterns in ((0, 1, 2), (0, 0, 1), (0, 0, 0)):
            words = payloads + 64 * np.array(patterns)
            counts = (words == 3).sum(1)
            for sampling in ("iid", "distinct"):
                weights = nu[words].prod(1) * pair[words[:, 0], words[:, 1]]
                weights *= pair[words[:, 0], words[:, 2]] * pair[words[:, 1], words[:, 2]]
                if sampling == "distinct":
                    weights *= (words[:, 0] != words[:, 1]) & (words[:, 0] != words[:, 2]) & (words[:, 1] != words[:, 2])
                expected = np.bincount(counts, weights=weights, minlength=4)
                actual, _ = triangle_quotient(positions, energy, generator, nu, patterns, .3, sampling, 3)
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-16)

    def test_edge_covariance_and_the_four_user_cycle_boundary(self):
        f = self.fixture
        for sampling in ("iid", "distinct"):
            result = edge_deletion_diagnostic(f.phi, f.y, .3, sampling)
            self.assertGreater(result["max_tail_change"], 0)
            self.assertGreaterEqual(result["posterior_pupe_regret"], -1e-14)
        # Four orthogonal messages with distinct sampling: a cycle admits nonadjacent repeated words.
        pair = np.ones((4, 4)) - np.eye(4)
        a = pair / 4
        cycle = np.trace(np.linalg.matrix_power(a, 4))
        full = math.factorial(4) / 4 ** 4
        self.assertEqual(cycle, 84 / 256)
        self.assertEqual(full, 24 / 256)
        self.assertNotEqual(cycle, full)


if __name__ == "__main__":
    unittest.main()
