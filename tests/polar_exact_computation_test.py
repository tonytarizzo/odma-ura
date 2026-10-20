"""Certificates for exact elimination, mixed-pattern state lower bounds, and decision-directed search."""

import math
import unittest

import numpy as np
from scipy.special import gammaln, softmax

from tests.polar_exact_computation import (
    audit_banks, certificate_search, completion_bound, event_ranges, frontier_decode, tail_action,
)
from tests import polar_posterior_test as reference
from tests.polar_alternative_inference_test import assemble_connected, connected_polynomials
from tests.polar_posterior_test import count_states, exact_posterior


def future_geometry(phi, h, sampling):
    n, m = phi.shape
    if m == 0:
        return np.zeros((n, n)), np.zeros(n)
    offset = h * phi.mean(1)
    if h == 0 or (sampling == "distinct" and h == m):
        return np.zeros((n, n)), offset
    u, values, _ = np.linalg.svd(phi - phi.mean(1, keepdims=True), full_matrices=False)
    q = u[:, values > 1e-12]
    return q @ q.T, offset


class ExactComputationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.PolarPosteriorTests.setUpClass()
        cls.fixture = reference.PolarPosteriorTests

    def test_nested_future_spaces_consume_each_gaussian_direction_once(self):
        phi = np.array([[1, 0, .6, -.6, 0, 0], [0, 1, 0, 0, .8, -.8], [0, 0, .8, .8, .6, .6]])
        y, k = np.array([.3, -.4, .7]), 3
        for sampling in ("iid", "distinct"):
            for counts in count_states(6, k, sampling):
                projector, offset = future_geometry(phi, k, sampling)
                theta, remaining = np.zeros(3), k
                distance = np.square((np.eye(3) - projector) @ (y - offset)).sum()
                for stage in range(3):
                    local = counts[2 * stage:2 * stage + 2]
                    signal = phi[:, 2 * stage:2 * stage + 2] @ local
                    remaining -= local.sum()
                    new_projector, new_offset = future_geometry(phi[:, 2 * stage + 2:], remaining, sampling)
                    np.testing.assert_allclose(projector @ new_projector, new_projector, atol=2e-14)
                    np.testing.assert_allclose((np.eye(3) - projector) @ (signal + new_offset - offset), 0, atol=2e-14)
                    innovation = (projector - new_projector) @ (y - theta - signal - new_offset)
                    distance += innovation @ innovation
                    theta = new_projector @ (theta + signal)
                    projector, offset = new_projector, new_offset
                self.assertAlmostEqual(distance, np.square(y - phi @ counts).sum(), places=12)

    def test_linear_cut_rank_and_observation_scaling(self):
        past = np.array([[0., 0.], [1., 0.], [2., 0.]])
        future = np.array([[0., 0.], [0., 1.], [0., 2.]])
        for y in (np.array([.3, -.2]), np.array([-.4, .7])):
            kernel = np.exp(-np.square(y - past[:, None, :] - future[None, :, :]).sum(2) / 2)
            self.assertEqual(np.linalg.matrix_rank(kernel), 1)
        future = past.copy()
        matrices = []
        for y in (np.array([.3, -.2]), np.array([-.4, .7])):
            kernel = np.exp(-np.square(y - past[:, None, :] - future[None, :, :]).sum(2) / 2)
            self.assertEqual(np.linalg.matrix_rank(kernel), 3)
            matrices.append(kernel)
        ratio = matrices[1] / matrices[0]
        self.assertEqual(np.linalg.matrix_rank(ratio), 1)
        u, values, vh = np.linalg.svd(matrices[0])
        weights, completion_weights = np.array([.1, .3, .6]), np.array([.2, .5, .3])
        compressed = weights @ (u * values)
        self.assertAlmostEqual(compressed @ vh @ completion_weights, weights @ matrices[0] @ completion_weights)

    def test_frontier_forward_backward_all_count_queries_and_orders(self):
        f = self.fixture
        groups = [list(range(4 * p, 4 * p + 4)) for p in range(4)]
        for sampling in ("iid", "distinct"):
            for k in (1, 3):
                states, posterior = exact_posterior(f.phi, f.y, k, f.sigma2, sampling)
                expected = np.array([posterior @ (states == j) for j in range(k + 1)]).T
                beta = np.ones(len(states)) if sampling == "distinct" else np.exp(-gammaln(states + 1).sum(1))
                evidence = beta @ np.exp(-np.square(f.y - states @ f.phi.T).sum(1) / (2 * f.sigma2))
                for order in ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2)):
                    action, probabilities, z, meta = frontier_decode(f.phi, groups, f.y, k, f.sigma2, sampling, order)
                    np.testing.assert_allclose(probabilities, expected, atol=2e-14)
                    self.assertAlmostEqual(z / evidence, 1, places=12)
                    self.assertAlmostEqual(meta["backward_evidence"] / evidence, 1, places=12)
                    self.assertEqual(meta["states"][-1], 1)
                    tails = np.array([posterior @ (states >= j) for j in range(1, k + 1)]).T
                    gain = (tails * (np.arange(k) < action[:, None])).sum()
                    self.assertAlmostEqual(gain, np.sort(tails.ravel())[-k:].sum(), places=13)

    def test_frontier_merging_preserves_ambiguous_identities(self):
        phi = np.array([[1., 1., 0., 0.], [0., 0., 1., -1.], [0., 0., 0., 0.]])
        y, groups = np.array([1.2, -.3, .7]), [[0, 1], [2, 3]]
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(phi, y, 2, .4, sampling)
            _, probabilities, _, meta = frontier_decode(phi, groups, y, 2, .4, sampling, (0, 1))
            expected = np.array([posterior @ (states == j) for j in range(3)]).T
            np.testing.assert_allclose(probabilities, expected, atol=1e-14)
            self.assertEqual(meta["frontier_widths"], [0, 0])
            self.assertEqual(meta["states"], [1, 3, 1])

    def test_mixed_pattern_state_lower_bound_on_all_selected_banks(self):
        rows = audit_banks()
        self.assertEqual(len(rows), 6)
        for row in rows:
            cut = next(c for c in row["cuts"] if c["processed_patterns"] == 8)
            self.assertEqual(cut["future_axes"], 256)
            self.assertEqual(cut["private_ranks"], [6] * 8)
            self.assertEqual(cut["log2_reachable_lower_bound"], 48)
            self.assertGreaterEqual(row["natural_peak_frontier"], 240)
            self.assertLessEqual(row["greedy_peak_frontier"], row["natural_peak_frontier"])

    def test_completion_bounds_cover_every_finite_completion(self):
        phi = self.fixture.phi[:, :7]
        residual = np.array([.4, -.2, .1, .7, -.5, .6, -.3, .2, -.1, .3, .1, -.6])
        for sampling in ("iid", "distinct"):
            for h in (1, 2, 3):
                states = count_states(7, h, sampling)
                exact = np.square(residual - states @ phi.T).sum(1).min()
                previous = 0.
                for mode in ("ball", "box", "dual"):
                    lower, calls = completion_bound(phi, residual, h, sampling, mode)
                    self.assertGreaterEqual(lower + 1e-13, previous)
                    self.assertLessEqual(lower, exact + 1e-13)
                    self.assertLessEqual(calls, 8)
                    previous = lower

    def test_count_event_feasibility_and_joint_difference_masks(self):
        m, k = 4, 3
        event_words = np.repeat(np.arange(m), k)
        for sampling in ("iid", "distinct"):
            states = count_states(m, k, sampling)
            for depth in range(m + 1):
                for prefix in sorted(set(tuple(a[:depth]) for a in states)):
                    local = states[np.all(states[:, :depth] == prefix, axis=1)]
                    events = (local[:, :, None] >= np.arange(1, k + 1)).reshape(len(local), -1)
                    guaranteed, possible = event_ranges(prefix, k - sum(prefix), m, k, sampling)
                    np.testing.assert_array_equal(guaranteed, events.all(0))
                    np.testing.assert_array_equal(possible, events.any(0))
                    mask = (~guaranteed[:, None]) & possible[None, :]
                    actual = ((~events[:, :, None]) & events[:, None, :]).any(0)
                    different = event_words[:, None] != event_words[None, :]
                    np.testing.assert_array_equal(mask[different], actual[different])
            # Unknown mass contributes exactly zero when both indicators are guaranteed true.
            guaranteed, possible = event_ranges((1, 1), 1, m, k, sampling)
            self.assertTrue(possible[k])
            self.assertFalse((~guaranteed[0]) & possible[k])

    def test_connected_remainder_bounds_a_contrast_once(self):
        phi, y, k = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.]]), np.array([.7, -.2]), 3
        for sampling in ("iid", "distinct"):
            for sigma2 in (.7, 5.):
                nu = softmax(phi.T @ y / sigma2)
                pair = np.exp(-phi.T @ phi / sigma2)
                if sampling == "distinct":
                    np.fill_diagonal(pair, 0)
                absolute = abs(pair - 1)
                delta = 3 * (absolute @ nu).max() ** 2 * (1 + absolute.max())
                exact, partial = [], []
                for w in range(4):
                    blocks = connected_polynomials(phi, y, k, sigma2, sampling, w)
                    exact.append(np.flip(np.cumsum(np.flip(assemble_connected(blocks, k)[1:]))))
                    partial.append(np.flip(np.cumsum(np.flip(assemble_connected(blocks, k, 2)[1:]))))
                errors = (np.asarray(exact) - np.asarray(partial)).ravel()
                self.assertLessEqual(np.max(abs(errors[:, None] - errors[None, :])), delta + 1e-12)

    def test_all_search_variants_return_the_exhaustive_bayes_action(self):
        f, k = self.fixture, 3
        for sampling in ("iid", "distinct"):
            for sigma2 in (.05, .3, 1.):
                rng = np.random.default_rng(9100)
                truth = rng.choice(16, k, replace=sampling == "iid")
                y = f.phi[:, truth].sum(1) + math.sqrt(sigma2) * rng.normal(size=12)
                order = np.argsort(-(f.phi.T @ y), kind="stable")
                phi = f.phi[:, order]
                states, posterior = exact_posterior(phi, y, k, sigma2, sampling)
                tails = np.array([posterior @ (states >= j) for j in range(1, k + 1)]).T
                best = tail_action(tails, k)
                optimal = (tails * (np.arange(k) < best[:, None])).sum()
                for mode in ("ball", "box", "dual"):
                    for certificate in ("interval", "difference"):
                        action, meta = certificate_search(phi, y, k, sigma2, sampling, mode, certificate)
                        self.assertLessEqual(meta["leaves"], len(states))
                        gain = (tails * (np.arange(k) < action[:, None])).sum()
                        self.assertAlmostEqual(gain, optimal, places=12)


if __name__ == "__main__":
    unittest.main()
