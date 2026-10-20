"""Checks for exact reusable bounds, weighted trees, and overlap-parity contractions."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import gammaln, softmax

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from tests import polar_posterior_test as reference
from tests.polar_exact_computation import certificate_search, tail_action
from tests.polar_exact_reuse import CachedCompletion, PolarPairOperator, negative_prior_matrix, weighted_tree_sums
from tests.polar_posterior_test import count_states, exact_posterior
from tests.polar_alternative_inference_test import connected_edge_sets


def labelled_trees(size):
    for sequence in itertools.product(range(size), repeat=size - 2):
        degree = np.ones(size, dtype=int) + np.bincount(sequence, minlength=size)
        edges = []
        for vertex in sequence:
            leaf = np.flatnonzero(degree == 1)[0]
            edges.append((leaf, vertex))
            degree[leaf] -= 1
            degree[vertex] -= 1
        edges.append(tuple(np.flatnonzero(degree == 1)))
        yield edges


class ExactReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.PolarPosteriorTests.setUpClass()
        cls.fixture = reference.PolarPosteriorTests

    def test_exact_prior_negative_events_at_every_tiny_prefix(self):
        m, k = 5, 3
        for sampling in ("iid", "distinct"):
            states = count_states(m, k, sampling)
            beta = np.ones(len(states)) if sampling == "distinct" else np.exp(-gammaln(states + 1).sum(1))
            for depth in range(m + 1):
                for prefix in set(tuple(a[:depth]) for a in states):
                    mask = np.all(states[:, :depth] == prefix, axis=1)
                    events = (states[mask, :, None] >= np.arange(1, k + 1)).reshape(mask.sum(), -1)
                    conditional = beta[mask] / beta[mask].sum()
                    expected = np.einsum("a,ai,aj->ij", conditional, ~events, events)
                    actual = negative_prior_matrix(prefix, k - sum(prefix), m, k, sampling)
                    np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_cached_directions_bound_every_legal_completion(self):
        f, k = self.fixture, 3
        phi = f.phi[:, :6]
        for sampling in ("iid", "distinct"):
            cache = CachedCompletion(phi, f.y, k, sampling)
            states = count_states(6, k, sampling)
            for depth in range(6):
                for prefix in set(tuple(a[:depth]) for a in states):
                    h = k - sum(prefix)
                    local = states[np.all(states[:, :depth] == prefix, axis=1)]
                    residual = f.y - phi[:, :depth] @ prefix
                    lower, calls = cache(depth, residual, h)
                    minimum = np.square(f.y - local @ phi.T).sum(1).min()
                    self.assertLessEqual(lower, minimum + 1e-12)
                    self.assertEqual(calls, 0)
                    for i, u in enumerate(cache.directions):
                        expected = ((local[:, depth:] @ phi[:, depth:].T) @ u).max()
                        self.assertAlmostEqual(cache.support[i, depth, h], expected, places=12)
            self.assertLessEqual(cache.setup_scores, 8 * phi.shape[1])

    def test_rooted_recurrence_matches_all_labelled_trees(self):
        nu = np.array([.2, .3, .5])
        interaction = np.array([[.1, .7, .3], [.7, .4, .2], [.3, .2, .6]])
        sums = weighted_tree_sums(lambda x: interaction @ x, nu, 5)
        alpha = (interaction @ nu).max()
        for size in range(2, 6):
            assignments = np.array(list(itertools.product(range(3), repeat=size)))
            weights = nu[assignments].prod(1)
            exact = sum(weights @ np.prod([interaction[assignments[:, i], assignments[:, j]] for i, j in edges], axis=0)
                        for edges in labelled_trees(size))
            self.assertAlmostEqual(sums[size], exact, places=12)
            self.assertLessEqual(sums[size], size ** (size - 2) * alpha ** (size - 1) + 1e-12)
        constant = weighted_tree_sums(lambda x: np.ones(3) * (.4 * x.sum()), nu, 6)
        for size in range(2, 7):
            self.assertAlmostEqual(constant[size], size ** (size - 2) * .4 ** (size - 1), places=12)

    def test_tree_majorant_bounds_exact_connected_graph_sums(self):
        nu = np.array([.2, .3, .5])
        interaction = np.array([[.1, .7, .3], [.7, .4, .2], [.3, .2, .6]])
        trees = weighted_tree_sums(lambda x: interaction @ x, nu, 4)
        for size in (2, 3, 4):
            assignments = np.array(list(itertools.product(range(3), repeat=size)))
            weights = nu[assignments].prod(1)
            exact = sum(weights @ np.prod([interaction[assignments[:, i], assignments[:, j]] for i, j in edges], axis=0)
                        for edges in connected_edge_sets(size))
            upper = trees[size] * (1 + interaction.max()) ** ((size - 1) * (size - 2) // 2)
            self.assertLessEqual(exact, upper + 1e-12)

    def test_overlap_parities_and_xor_convolution_match_full_pair_matrix(self):
        f = self.fixture
        base = ODMAPolarBaseline(12, 256, 3201, prefix_bits=6, code_length=32, crc_bits=8)
        generator = base.polar.encode(np.eye(6, dtype=np.uint8)).T
        positions = base.positions[:4]
        energy = np.random.default_rng(971).dirichlet(np.ones(32), size=4)
        signs = 1. - 2. * base.polar.encode(message_bits(np.arange(64), 6))
        phi = np.zeros((256, 256))
        for p in range(4):
            phi[np.ix_(positions[p], np.arange(64 * p, 64 * (p + 1)))] = np.sqrt(energy[p, :, None]) * signs.T
        small_generator = f.base.polar.encode(np.eye(2, dtype=np.uint8)).T
        for pos, energies, gen, matrix in ((f.base.positions, f.energy, small_generator, f.phi),
                                          (positions, energy, generator, phi)):
            for sampling in ("iid", "distinct"):
                operator = PolarPairOperator(pos, energies, gen, .3, sampling)
                dense = abs(np.expm1(-(matrix.T @ matrix) / .3))
                if sampling == "distinct":
                    np.fill_diagonal(dense, 1)
                rng = np.random.default_rng(123)
                for x in (rng.normal(size=matrix.shape[1]), np.ones(matrix.shape[1])):
                    np.testing.assert_allclose(operator(x), dense @ x, rtol=1e-12, atol=2e-12)
                signed = PolarPairOperator(pos, energies, gen, .3, sampling, absolute=False)
                pair = np.expm1(-(matrix.T @ matrix) / .3)
                if sampling == "distinct":
                    np.fill_diagonal(pair, -1)
                np.testing.assert_allclose(signed(x), pair @ x, rtol=1e-12, atol=2e-12)
                nu = rng.dirichlet(np.ones(matrix.shape[1]))
                np.testing.assert_allclose(weighted_tree_sums(operator, nu, 5),
                                           weighted_tree_sums(lambda x: dense @ x, nu, 5), rtol=1e-12, atol=1e-12)

    def test_two_user_full_posterior_from_structured_pair_contraction(self):
        f = self.fixture
        generator = f.base.polar.encode(np.eye(2, dtype=np.uint8)).T
        for sampling in ("iid", "distinct"):
            operator = PolarPairOperator(f.base.positions, f.energy, generator, f.sigma2, sampling, absolute=False)
            nu = softmax(f.phi.T @ f.y / f.sigma2)
            pair_nu = nu.sum() + operator(nu)
            normalizer = nu @ pair_nu
            q2 = nu ** 2 * (math.exp(-1 / f.sigma2) if sampling == "iid" else 0) / normalizer
            q1 = 2 * nu * pair_nu / normalizer - 2 * q2
            actual = np.stack((1 - q1 - q2, q1, q2), axis=1)
            states, posterior = exact_posterior(f.phi, f.y, 2, f.sigma2, sampling)
            expected = np.array([posterior @ (states == j) for j in range(3)]).T
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_reused_and_prior_weighted_certificates_match_bayes(self):
        f, k = self.fixture, 3
        for sampling in ("iid", "distinct"):
            for sigma2 in (.05, .3, 1.):
                rng = np.random.default_rng(9100)
                truth = rng.choice(16, k, replace=sampling == "iid")
                y = f.phi[:, truth].sum(1) + math.sqrt(sigma2) * rng.normal(size=12)
                phi = f.phi[:, np.argsort(-(f.phi.T @ y), kind="stable")]
                states, posterior = exact_posterior(phi, y, k, sigma2, sampling)
                tails = np.array([posterior @ (states >= j) for j in range(1, k + 1)]).T
                action = tail_action(tails, k)
                expected = (tails * (np.arange(k) < action[:, None])).sum()
                for mode in ("cached", "dual"):
                    cache = CachedCompletion(phi, y, k, sampling) if mode == "cached" else None
                    action, meta = certificate_search(phi, y, k, sigma2, sampling, mode, "difference", bounder=cache,
                                                      negative_probability=negative_prior_matrix)
                    actual = (tails * (np.arange(k) < action[:, None])).sum()
                    self.assertAlmostEqual(actual, expected, places=12)
                    if cache is not None:
                        self.assertLessEqual(meta["score_evaluations"], 128)


if __name__ == "__main__":
    unittest.main()
