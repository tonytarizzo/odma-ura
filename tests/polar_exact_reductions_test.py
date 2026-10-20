"""Exact state reductions, bounds and sum aliases for Report 7; no receiver performance claims."""

import itertools
import math
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, softmax

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode
from tests.polar_alternative_inference_test import assemble_connected, connected_edge_sets, connected_polynomials
from tests.polar_posterior_test import count_states, exact_posterior


def support_bound(phi, residual, h, dual, sampling):
    scores = dual @ phi
    support = h * scores.max() if sampling == "iid" else np.sort(scores)[-h:].sum() if h else 0.0
    return 2 * dual @ residual - dual @ dual - 2 * support


def partition_majorant(blocks, k, cutoff):
    values = [1.0]
    for total in range(1, k + 1):
        values.append(sum(math.comb(total - 1, m - 1) * blocks[m] * values[total - m]
                          for m in range(1, min(total, cutoff) + 1)))
    return values[k]


class ExactReductionTests(unittest.TestCase):
    def test_sufficient_observation_and_character_moments(self):
        a = math.sqrt(.84)
        phi = np.array([[a, -a, 0, 0], [0, 0, a, -a], [.4, .4, .4, .4]])
        y, k, sigma2 = np.array([.7, -.2, 2.1]), 2, .6
        mean = phi.mean(1)
        covariance = phi @ phi.T / 4 - np.outer(mean, mean)
        np.testing.assert_allclose(covariance, np.diag([.42, .42, 0]), atol=1e-15)
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(phi, y, k, sigma2, sampling)
            _, projected = exact_posterior(phi[:2], y[:2], k, sigma2, sampling)
            np.testing.assert_allclose(posterior, projected, atol=1e-14)
            full = np.exp(-np.square(y - states @ phi.T).sum(1) / (2 * sigma2))
            reduced = np.exp(-np.square(y[:2] - states @ phi[:2].T).sum(1) / (2 * sigma2))
            np.testing.assert_allclose(full / reduced, np.exp(-(y[2] - k * mean[2]) ** 2 / (2 * sigma2)))
        code = PolarCode(6, 32, 8)
        generator = code.encode(np.eye(6, dtype=np.uint8)).T
        groups, inverse = np.unique(generator, axis=0, return_inverse=True)
        coefficients = np.eye(len(groups))[inverse] / math.sqrt(32)
        signs = 1 - 2 * code.encode(message_bits(np.arange(64), 6)).astype(float)
        np.testing.assert_allclose(coefficients @ coefficients.T, signs.T @ signs / (64 * 32), atol=1e-15)
        np.testing.assert_allclose(coefficients[:, ~groups.any(1)].sum(1), signs.mean(0) / math.sqrt(32))
        self.assertEqual(np.linalg.matrix_rank((signs - signs.mean(0)).T), 28)
        dimensions = [int(np.any(np.unique(generator[:, r:], axis=0), axis=1).sum()) for r in range(7)]
        self.assertEqual(dimensions, [28, 23, 15, 7, 3, 1, 0])
        for sampling in ("iid", "distinct"):
            states = count_states(16, 4, sampling)
            local_sums = states @ signs[:16]
            self.assertEqual(len(np.unique(local_sums, axis=0)), len(states))
            # Restricted rows contain the complete four-bit Walsh basis, including the constant character.
            _, indices = np.unique(generator[:, 2:], axis=0, return_index=True)
            characters = signs[:16, indices].T
            np.testing.assert_array_equal((local_sums[:, indices] @ characters) / 16, states)

    def test_continuation_projection_preserves_marked_evidence(self):
        left = np.array([[0, 0, 0], [1, 0, -1], [0, 1, 0]])
        right = np.array([[.8, -.8], [0, 0], [.6, .6]])
        y, sigma2, q, h = np.array([.4, .3, -.1]), .7, 2, 1
        offset = h * right.mean(1)
        for sampling in ("iid", "distinct"):
            a, b = count_states(3, q, sampling), count_states(2, h, sampling)
            sums, future = a @ left.T, b @ right.T
            weights = np.ones(len(a)) if sampling == "distinct" else np.exp(-gammaln(a + 1).sum(1))
            direct, merged = np.zeros(q + 1), np.zeros(q + 1)
            # All these partial signals have the same projection onto the future x-axis variation.
            keys = {}
            for counts, signal, weight in zip(a, sums, weights):
                orthogonal_weight = weight * np.exp(-np.square((y - offset - signal)[1:]).sum() / (2 * sigma2))
                key = signal[0]
                keys.setdefault(key, np.zeros(q + 1))[counts[0]] += orthogonal_weight
                direct[counts[0]] += weight * np.exp(-np.square(y - signal - future).sum(1) / (2 * sigma2)).sum()
            self.assertEqual(len(keys), 1)
            for key, coefficients in keys.items():
                merged += coefficients * np.exp(-np.square(y[0] - key - future[:, 0]) / (2 * sigma2)).sum()
            np.testing.assert_allclose(merged, direct, atol=1e-14)
            # Different projections cannot be merged up to a scalar when both continuations remain possible.
            alternative = np.array([.2, .1, .3])
            ratios = np.exp((-np.square(y - alternative - future).sum(1)
                             + np.square(y - sums[0] - future).sum(1)) / (2 * sigma2))
            self.assertGreater(np.ptp(ratios), .01)

    def test_full_selected_banks_have_full_affine_observation_span(self):
        generator = PolarCode(6, 32, 8).encode(np.eye(6, dtype=np.uint8)).T
        groups, inverse = np.unique(generator, axis=0, return_inverse=True)
        np.testing.assert_array_equal(np.flatnonzero(~generator.any(1)), [0])
        for bits in (12, 14):
            for seed in (3201, 3202, 3203):
                base = ODMAPolarBaseline(bits, 256, seed, prefix_bits=bits - 6, code_length=32, crc_bits=8)
                supports = [set(p[inverse == j]) for p in base.positions for j, g in enumerate(groups) if g.any()]
                known = set()
                while True:
                    extra = set.union(set(), *(s - known for s in supports if len(s - known) == 1))
                    if not extra:
                        break
                    known |= extra
                # Nonzero character averages give variation vectors. Subtract known axes to isolate new ones.
                self.assertTrue(any(p[0] in known for p in base.positions))
                # One known pattern-mean axis anchors differences of means from every other pattern.
                known.update(base.positions[:, 0].tolist())
                self.assertEqual(known, set(range(256)))

    def test_completion_dual_bounds_and_convex_distance(self):
        rng = np.random.default_rng(12013)
        phi = rng.normal(size=(3, 5))
        phi /= np.linalg.norm(phi, axis=0)
        residual, h = np.array([1.1, -.9, .4]), 2
        for sampling in ("iid", "distinct"):
            completions = count_states(5, h, sampling) @ phi.T
            exact_distance = np.square(completions - residual).sum(1).min()
            for dual in rng.normal(size=(20, 3)):
                self.assertLessEqual(support_bound(phi, residual, h, dual, sampling), exact_distance + 1e-13)
            result = minimize(lambda x: np.square(residual - phi @ x).sum(), np.full(5, h / 5),
                              jac=lambda x: -2 * phi.T @ (residual - phi @ x), method="SLSQP",
                              bounds=[(0, 1 if sampling == "distinct" else h)] * 5,
                              constraints={"type": "eq", "fun": lambda x: x.sum() - h,
                                           "jac": lambda x: np.ones(5)}, options={"ftol": 1e-13, "maxiter": 200})
            self.assertTrue(result.success)
            dual_value = support_bound(phi, residual, h, residual - phi @ result.x, sampling)
            self.assertAlmostEqual(result.fun, dual_value, places=7)
            self.assertLessEqual(result.fun, exact_distance + 1e-12)
        phi, residual = np.eye(2), -np.ones(2)
        for sampling in ("iid", "distinct"):
            self.assertEqual(support_bound(phi, residual, 2, -2 * np.ones(2), sampling), 8)
        self.assertEqual(max(0, np.linalg.norm(residual) - 2) ** 2, 0)
        # An underestimated support reverses safety: keeping only e1 misses the best dual score e2.
        dual, residual = np.array([0., 3.]), np.array([0., 2.])
        unsafe = support_bound(phi[:, :1], residual, 2, dual, "iid")
        self.assertGreater(unsafe, 0)  # The full alphabet has a zero-distance completion 2*e2.

    def test_one_dimensional_separator_can_have_full_kernel_rank(self):
        for size in range(2, 7):
            x = np.arange(size)
            nodes = np.exp(-.3 * np.arange(size))
            kernel = np.exp(-.3 * np.outer(x, x))
            vandermonde_determinant = math.prod(nodes[j] - nodes[i] for i in range(size) for j in range(i + 1, size))
            self.assertAlmostEqual(np.linalg.det(kernel) / vandermonde_determinant, 1, places=7)
            self.assertEqual(np.linalg.matrix_rank(kernel), size)

    def test_weighted_connected_majorant_and_count_tail_intervals(self):
        phi = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        y, k, marked = np.array([.7, -.2]), 4, 0
        for sampling in ("iid", "distinct"):
            for sigma2 in (5., .7):
                nu = softmax(phi.T @ y / sigma2)
                pair = np.exp(-phi.T @ phi / sigma2)
                if sampling == "distinct":
                    np.fill_diagonal(pair, 0)
                absolute = abs(pair - 1)
                epsilon, alpha = absolute.max(), (absolute @ nu).max()
                bounds = [0., 1.] + [m ** (m - 2) * alpha ** (m - 1)
                                    * (1 + epsilon) ** ((m - 1) * (m - 2) // 2) for m in range(2, k + 1)]
                for size in range(2, k + 1):
                    assignments = np.array(list(itertools.product(range(4), repeat=size)))
                    weights = np.prod(nu[assignments], axis=1)
                    value = 0.
                    for edges in connected_edge_sets(size):
                        graph = np.ones(len(assignments))
                        for i, j in edges:
                            graph *= absolute[assignments[:, i], assignments[:, j]]
                        value += weights @ graph
                    self.assertLessEqual(value, bounds[size] + 1e-10)
                connected = connected_polynomials(phi, y, k, sigma2, sampling, marked)
                full = assemble_connected(connected, k)
                for cutoff in (1, 2, 3):
                    partial = assemble_connected(connected, k, cutoff)
                    delta = partition_majorant(bounds, k, k) - partition_majorant(bounds, k, cutoff)
                    self.assertLessEqual(np.abs(full - partial).sum(), delta + 1e-9)
                    z_low, z_high = max(0., partial.sum() - delta), partial.sum() + delta
                    self.assertLessEqual(z_low, full.sum() + 1e-12)
                    self.assertGreaterEqual(z_high, full.sum() - 1e-12)
                    for j in range(1, k + 1):
                        tail = full[j:].sum() / full.sum()
                        numerator_low = max(0., partial[j:].sum() - delta)
                        numerator_high = max(0., partial[j:].sum() + delta)
                        lower = numerator_low / z_high
                        upper = min(1., numerator_high / z_low) if z_low > 0 else 1.
                        self.assertLessEqual(lower, tail + 1e-12)
                        self.assertGreaterEqual(upper, tail - 1e-12)
                if sampling == "iid" and sigma2 == 5.:
                    delta = partition_majorant(bounds, k, k) - partition_majorant(bounds, k, 2)
                    tails = []
                    for message in range(4):
                        blocks = [np.pad(c, (0, k - 2))
                                  for c in connected_polynomials(phi, y, 2, sigma2, sampling, message)]
                        partial = assemble_connected(blocks, k, 2)
                        tails.append(np.flip(np.cumsum(np.flip(partial[1:]))))
                    tails = np.asarray(tails)
                    self.assertGreater((tails[:, 0] - delta).min(), (tails[:, 1] + delta).max())
                    states, posterior = exact_posterior(phi, y, k, sigma2, sampling)
                    risks = [1 - posterior @ np.minimum(states, action).sum(1) / k for action in states]
                    self.assertAlmostEqual(1 - posterior @ np.minimum(states, 1).sum(1) / k, min(risks))

    def test_first_same_pattern_alias_and_noiseless_bayes_risk(self):
        code = PolarCode(6, 32, 8)
        signs = 1 - 2 * code.encode(message_bits(np.arange(64), 6)).astype(np.int16)
        triples = np.array(list(itertools.combinations_with_replacement(range(64), 3)))
        self.assertEqual(len(np.unique(signs[triples].sum(1), axis=0)), 45760)
        words = np.array(list(itertools.combinations_with_replacement(range(64), 4)), dtype=np.uint8)
        _, inverse, counts = np.unique(signs[words].sum(1).astype(np.int8), axis=0, return_inverse=True, return_counts=True)
        self.assertEqual(dict(zip(*np.unique(counts, return_counts=True))), {1: 765648, 2: 392, 4: 12})
        np.testing.assert_array_equal(signs[[12, 15, 57, 58]].sum(0), signs[[20, 23, 33, 34]].sum(0))
        generator = code.encode(np.eye(6, dtype=np.uint8)).T
        subspace = message_bits(np.array([0, 3, 53, 54]), 6)
        difference = message_bits(np.array([24]), 6)[0]
        restricted = generator[((generator @ subspace.T) % 2 == 0).all(1)]
        self.assertTrue(np.all(restricted @ difference % 2 == 0))
        risk_mass = 0.
        for group in np.flatnonzero(counts > 1):
            local = words[inverse == group]
            self.assertTrue(all(len(set(w)) == 4 for w in local))
            states = np.array([np.bincount(w, minlength=64) for w in local])
            self.assertTrue(np.all(states.sum(0) <= 1))
            inclusion = states.mean(0)
            risk_mass += len(local) * (1 - np.sort(inclusion)[-4:].sum() / 4)
        self.assertEqual(risk_mass, 428)
        self.assertAlmostEqual(math.factorial(4) * risk_mass / 64 ** 4, .0006122589111328125)
        self.assertAlmostEqual(risk_mass / math.comb(64, 4), .0006736168819722495)


if __name__ == "__main__":
    unittest.main()
