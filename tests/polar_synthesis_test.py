"""Algebra checks for combinations of Report 7 results; no approximation performance experiment."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import gammaln, softmax

from tests import polar_posterior_test as reference
from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import count_states, exact_posterior


def tv(p, q):
    return abs(p - q).sum() / 2


def kl(p, q):
    return p @ np.log(p / q)


class SynthesisTests(unittest.TestCase):
    def test_exact_recentering_preserves_posterior_and_pair_cross_ratios(self):
        reference.PolarPosteriorTests.setUpClass()
        f = reference.PolarPosteriorTests
        centre = np.random.default_rng(42).normal(size=12) / 5
        shifted = f.phi - centre[:, None]
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(f.phi, f.y, 3, f.sigma2, sampling)
            gram = shifted.T @ shifted
            pairs = (np.einsum("ai,ij,aj->a", states, gram, states) - states @ gram.diagonal()) / 2
            unary = f.phi.T @ (f.y - 2 * centre)
            log_prior = -gammaln(states + 1).sum(1)
            actual = softmax((states @ unary - pairs) / f.sigma2 + log_prior)
            np.testing.assert_allclose(actual, posterior, atol=2e-14)
        original, recentered = -(f.phi.T @ f.phi) / f.sigma2, -(shifted.T @ shifted) / f.sigma2
        scaling = np.exp(centre @ f.phi / f.sigma2)
        factored = math.exp(-centre @ centre / f.sigma2) * scaling[:, None] * np.exp(original) * scaling[None, :]
        np.testing.assert_allclose(factored, np.exp(recentered), rtol=1e-13, atol=1e-14)
        for u, v, a, b in itertools.product(range(4), repeat=4):
            first = original[u, v] + original[a, b] - original[u, b] - original[a, v]
            second = recentered[u, v] + recentered[a, b] - recentered[u, b] - recentered[a, v]
            expected = -(f.phi[:, u] - f.phi[:, a]) @ (f.phi[:, v] - f.phi[:, b]) / f.sigma2
            self.assertAlmostEqual(first, second, places=12)
            self.assertAlmostEqual(first, expected, places=12)
        eps = .05
        wave = np.vstack((np.full(4, math.sqrt(1 - eps ** 2)), eps * np.array([[1, 0, -1, 0], [0, 1, 0, -1]])))
        mean = wave.mean(1)
        raw = abs(np.expm1(-(wave.T @ wave) / .3)).max()
        centred = abs(np.expm1(-((wave - mean[:, None]).T @ (wave - mean[:, None])) / .3)).max()
        self.assertGreater(raw, .96)
        self.assertLess(centred, .009)

    def test_future_width_is_a_support_function_and_log_cross_ratio_range(self):
        rng = np.random.default_rng(541)
        phi = rng.normal(size=(3, 5))
        delta, s, y, sigma2 = rng.normal(size=(3, 3)), rng.normal(size=3), rng.normal(size=3), .7
        for sampling in ("iid", "distinct"):
            for h in (0, 1, 3, 5):
                future = count_states(5, h, sampling) @ phi.T
                for d in delta:
                    scores = d @ phi
                    width = h * np.ptp(scores) if sampling == "iid" else np.sort(scores)[-h:].sum() - np.sort(scores)[:h].sum() if h else 0
                    self.assertAlmostEqual(width, np.ptp(future @ d), places=12)
                    ratios = (-np.square(y - s - d - future).sum(1) + np.square(y - s - future).sum(1)) / (2 * sigma2)
                    self.assertAlmostEqual(np.ptp(ratios), width / sigma2, places=12)

    def test_one_cut_compression_preserves_markers_and_bounds_posterior_and_risk(self):
        rng = np.random.default_rng(851)
        left, right = rng.normal(size=(3, 4)), rng.normal(size=(3, 3))
        y, sigma2 = rng.normal(size=3), 8.
        for sampling in ("iid", "distinct"):
            past, future_counts = count_states(4, 1, sampling), count_states(3, 2, sampling)
            signals, future = past @ left.T, future_counts @ right.T
            groups = np.arange(4) // 2
            reps = signals[[0, 2]][groups]
            delta = signals - reps
            projections = delta @ future.T
            midpoint = (projections.max(1) + projections.min(1)) / 2
            widths = np.ptp(projections, axis=1)
            constant = ((delta * (y - reps)).sum(1) - np.square(delta).sum(1) / 2) / sigma2
            log_prior = -gammaln(past + 1).sum(1)[:, None] - gammaln(future_counts + 1).sum(1)[None, :]
            log_full = log_prior - np.square(y - signals[:, None, :] - future).sum(2) / (2 * sigma2)
            log_approx = log_prior + constant[:, None] - midpoint[:, None] / sigma2
            log_approx -= np.square(y - reps[:, None, :] - future).sum(2) / (2 * sigma2)
            np.testing.assert_allclose(log_full - log_approx, (midpoint[:, None] - projections) / sigma2, atol=2e-14)
            p, q = softmax(log_full.ravel()), softmax(log_approx.ravel())
            tau = math.tanh(widths.max() / (4 * sigma2))
            self.assertLessEqual(tv(p, q), tau + 1e-14)
            states = np.array([np.concatenate((a, b)) for a in past for b in future_counts])
            tails = np.array([p @ (states >= j) for j in range(1, 4)]).T
            approximate_tails = np.array([q @ (states >= j) for j in range(1, 4)]).T
            action, optimal = tail_action(approximate_tails, 3), tail_action(tails, 3)
            regret = p @ (np.minimum(states, optimal).sum(1) - np.minimum(states, action).sum(1)) / 3
            self.assertLessEqual(regret, 2 * tau + 1e-14)
            for marked in range(4):
                # Group marked weights, not just their scalar sums; future messages use disjoint identities.
                grouped = np.zeros((2, 2))
                for i, a in enumerate(past):
                    grouped[groups[i], a[marked]] += math.exp(constant[i] - midpoint[i] / sigma2)
                future_weight = np.exp(-gammaln(future_counts + 1).sum(1))
                kernel = np.exp(-np.square(y - signals[[0, 2], None, :] - future).sum(2) / (2 * sigma2))
                polynomial = grouped.T @ (kernel @ future_weight)
                expected = np.array([q @ (states[:, marked] == j) for j in (0, 1)])
                np.testing.assert_allclose(polynomial / polynomial.sum(), expected, atol=2e-14)
            mask = states[:, 0] == 0
            epsilon = p[~mask].sum()
            restricted = q * mask / q[mask].sum()
            conditional = p * mask / p[mask].sum()
            self.assertLessEqual(tv(p, restricted), epsilon + (1 - epsilon) * tv(conditional, restricted) + 1e-14)
        future = np.array([[0., -1.], [0., 1.]])
        delta = np.array([100., 0.])
        self.assertEqual(np.ptp(future @ delta), 0.)

    def test_sharp_log_ratio_bound_and_decision_margin(self):
        for eps in (.01, .3, 2.):
            q = softmax([eps, 0.])
            p = q * np.exp([-eps, eps])
            p /= p.sum()
            self.assertAlmostEqual(tv(p, q), math.tanh(eps / 2), places=14)
        q = np.array([.8, .15, .05])
        p = q * np.exp([-.1, .1, .04])
        p /= p.sum()
        bound = math.tanh(.2 / 4)
        self.assertGreater(q[0] - q[1], 2 * bound)
        self.assertEqual(p.argmax(), q.argmax())

    def test_composed_corrections_require_the_changed_reference(self):
        states = np.array(list(itertools.product((-1., 1.), repeat=2)))
        q = np.full(4, .25)
        h = states.prod(1)
        r1, r2 = 1 + .5 * states[:, 0], 1 + .5 * states[:, 1]
        self.assertEqual((q * r1) @ h, 0.)
        self.assertEqual((q * r2) @ h, 0.)
        final = q * r1 * r2
        final /= final.sum()
        self.assertEqual(final @ h, .25)
        current, change = q.copy(), 0.
        for ratio in (r1, r2):
            mean = current @ ratio
            change += ((current * ratio) @ h - (current @ h) * mean) / mean
            current = current * ratio / mean
        self.assertAlmostEqual(change, final @ h - q @ h, places=14)

    def test_alias_conditionals_and_distribution_projection(self):
        phi = np.array([[1., 1., 0.], [0., 0., 1.]])
        for sampling in ("iid", "distinct"):
            states, p = exact_posterior(phi, np.array([1.3, .4]), 2, .7, sampling)
            _, other = exact_posterior(phi, np.array([-.5, 1.8]), 2, .2, sampling)
            _, classes = np.unique(states @ phi.T, axis=0, return_inverse=True)
            prior = np.exp(-gammaln(states + 1).sum(1))
            q = softmax(np.random.default_rng(314).normal(size=len(states)))
            corrected = np.zeros(len(states))
            for cell in np.unique(classes):
                mask = classes == cell
                conditional = prior[mask] / prior[mask].sum()
                np.testing.assert_allclose(p[mask] / p[mask].sum(), conditional, atol=1e-14)
                np.testing.assert_allclose(other[mask] / other[mask].sum(), conditional, atol=1e-14)
                corrected[mask] = q[mask].sum() * conditional
            self.assertLessEqual(kl(p, corrected), kl(p, q) + 1e-14)
            self.assertLessEqual(kl(corrected, p), kl(q, p) + 1e-14)
            self.assertLessEqual(tv(p, corrected), tv(p, q) + 1e-14)

    def test_slot_symmetrization_preserves_count_queries(self):
        words = np.array(list(itertools.product(range(3), repeat=3)))
        counts = np.eye(3, dtype=int)[words].sum(1)
        _, orbit = np.unique(counts, axis=0, return_inverse=True)
        q = softmax(np.random.default_rng(48).normal(size=len(words)))
        symmetric = np.array([q[orbit == cell].mean() for cell in orbit])
        for j in range(1, 4):
            np.testing.assert_allclose(q @ (counts >= j), symmetric @ (counts >= j), atol=1e-14)
        target = softmax(-np.square(counts - np.array([2, 1, 0])).sum(1))
        self.assertLessEqual(kl(target, symmetric), kl(target, q) + 1e-14)
        self.assertLessEqual(kl(symmetric, target), kl(q, target) + 1e-14)


if __name__ == "__main__":
    unittest.main()
