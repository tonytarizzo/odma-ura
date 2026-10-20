"""Conditional approximation guarantees and a finite count-forest solver; no receiver performance claim."""

import itertools
import math
import unittest

import numpy as np
import torch
from scipy.special import gammaln, softmax

from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import count_states


def count_forest(phi, y, k, sigma2, edges, sampling, gauge=None):
    """Exact surrogate count marginals on a supplied forest, keeping total K and the chosen source law."""
    m = phi.shape[1]
    gram, linear = phi.T @ phi / sigma2, phi.T @ y / sigma2
    if gauge is not None:
        linear = linear - k * gauge
        gram = gram - gauge[:, None] - gauge[None, :]
    counts = np.arange(k + 1)
    log_unary = linear[:, None] * counts - gram.diagonal()[:, None] * counts ** 2 / 2
    if sampling == "iid":
        log_unary -= gammaln(counts + 1)
    theta = torch.tensor(log_unary, dtype=torch.float64, requires_grad=True)
    cap = 1 if sampling == "distinct" else k
    neighbours = [[] for _ in range(m)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    visited = set()

    def convolve(left, right):
        return torch.stack([sum(left[..., q] * right[..., h - q] for q in range(h + 1)) for h in range(k + 1)], -1)

    def inside(word, parent):
        if word in visited:
            raise ValueError("The supplied graph is not a forest")
        visited.add(word)
        rows = [torch.cat((torch.zeros(a), theta[word, a].exp().reshape(1), torch.zeros(k - a))) for a in range(cap + 1)]
        table = torch.stack(rows)
        for child in neighbours[word]:
            if child == parent:
                continue
            subtree = inside(child, word)
            levels = torch.arange(cap + 1, dtype=torch.float64)
            pair = torch.exp(-gram[word, child] * levels[:, None] * levels[None, :])
            table = convolve(table, pair @ subtree)
        return table

    total = torch.tensor([1.] + [0.] * k, dtype=torch.float64)
    for word in range(m):
        if word not in visited:
            total = convolve(total, inside(word, -1).sum(0))
    evidence = total[k]
    if not torch.isfinite(evidence) or evidence <= 0:
        raise FloatingPointError("Count-forest evidence needs rescaling or a feasible source domain")
    probabilities, = torch.autograd.grad(evidence.log(), theta)
    return probabilities.detach().numpy(), float(evidence.detach())


def enumerated_surrogate(phi, y, k, sigma2, edges, sampling, gauge=None):
    states = count_states(phi.shape[1], k, sampling)
    gram = phi.T @ phi / sigma2
    linear = phi.T @ y / sigma2
    if gauge is not None:
        linear = linear - k * gauge
        gram = gram - gauge[:, None] - gauge[None, :]
    score = states @ linear - (states ** 2) @ gram.diagonal() / 2
    if sampling == "iid":
        score -= gammaln(states + 1).sum(1)
    for a, b in edges:
        score -= gram[a, b] * states[:, a] * states[:, b]
    return states, softmax(score), np.exp(score).sum()


class AssumptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phi = np.random.default_rng(154).normal(size=(8, 5))
        cls.phi /= np.linalg.norm(cls.phi, axis=0)
        cls.y, cls.sigma2 = np.linspace(-.3, .4, 8), 1.7

    def test_count_forest_all_marginals_preserve_cardinality_and_source(self):
        for sampling in ("iid", "distinct"):
            for k in (2, 3):
                for edges in ([], [(0, 1), (1, 2), (1, 3), (3, 4)], [(0, 2), (1, 4)]):
                    probabilities, evidence = count_forest(self.phi, self.y, k, self.sigma2, edges, sampling)
                    states, posterior, expected_z = enumerated_surrogate(self.phi, self.y, k, self.sigma2, edges, sampling)
                    expected = np.array([posterior @ (states == j) for j in range(k + 1)]).T
                    np.testing.assert_allclose(probabilities, expected, rtol=2e-12, atol=1e-13)
                    self.assertAlmostEqual(evidence, expected_z, places=11)
                    self.assertAlmostEqual((probabilities @ np.arange(k + 1)).sum(), k, places=12)
                    if sampling == "distinct":
                        np.testing.assert_array_equal(probabilities[:, 2:], 0)

    def test_omitted_count_interactions_bound_posterior_and_decision_error(self):
        edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
        complete = list(itertools.combinations(range(5), 2))
        missing = [edge for edge in complete if edge not in edges]
        gram = self.phi.T @ self.phi / self.sigma2
        k = 3
        for sampling in ("iid", "distinct"):
            states, p, _ = enumerated_surrogate(self.phi, self.y, k, self.sigma2, complete, sampling)
            _, q, _ = enumerated_surrogate(self.phi, self.y, k, self.sigma2, edges, sampling)
            correction = -sum(gram[a, b] * states[:, a] * states[:, b] for a, b in missing)
            epsilon = math.comb(k, 2) * max(abs(gram[a, b]) for a, b in missing)
            self.assertLessEqual(abs(correction).max(), epsilon + 1e-14)
            tv = abs(p - q).sum() / 2
            self.assertLessEqual(tv, math.tanh(np.ptp(correction) / 4) + 1e-14)
            self.assertLessEqual(tv, math.tanh(epsilon / 2) + 1e-14)
            probabilities, _ = count_forest(self.phi, self.y, k, self.sigma2, edges, sampling)
            tails = np.flip(np.cumsum(np.flip(probabilities[:, 1:], axis=1), axis=1), axis=1)
            exact = np.array([p @ (states >= j) for j in range(1, k + 1)]).T
            proposed, best = tail_action(tails, k), tail_action(exact, k)
            regret = p @ (np.minimum(states, best).sum(1) - np.minimum(states, proposed).sum(1)) / k
            self.assertLessEqual(regret, 2 * tv + 1e-14)

    def test_inside_mass_is_not_posterior_mass(self):
        inside, continuation = np.array([.99, .01]), np.array([.001, 1.])
        posterior = inside * continuation
        posterior /= posterior.sum()
        self.assertGreater(posterior[1], .90)
        self.assertLess(inside[1], .02)

    def test_small_kernel_expansion_has_constructible_rank_and_relative_bound(self):
        rng = np.random.default_rng(55)
        left, right = rng.uniform(-.1, .1, (7, 2)), rng.uniform(-.1, .1, (9, 2))
        degree, dimension = 2, 2
        indices = []
        for order in range(degree + 1):
            for combination in itertools.combinations_with_replacement(range(dimension), order):
                indices.append(np.bincount(combination, minlength=dimension))
        self.assertEqual(len(indices), math.comb(dimension + degree, degree))
        a = np.stack([(-1.) ** alpha.sum() * np.prod(left ** alpha, axis=1)
                      / math.prod(math.factorial(int(x)) for x in alpha) for alpha in indices], axis=1)
        b = np.stack([np.prod(right ** alpha, axis=1) for alpha in indices], axis=1)
        kernel, approximate = np.exp(-left @ right.T), a @ b.T
        rho = np.linalg.norm(left, axis=1).max() * np.linalg.norm(right, axis=1).max()
        delta = math.exp(2 * rho) * rho ** (degree + 1) / math.factorial(degree + 1)
        self.assertLess(delta, 1)
        self.assertTrue(np.all(approximate > 0))
        self.assertLessEqual(np.max(abs(approximate - kernel) / kernel), delta + 1e-14)

    def test_diagonal_gaussian_metric_error_and_pattern_offset(self):
        d = np.linspace(.5, 1.5, 8)
        direction = np.arange(1., 9.)
        direction /= np.linalg.norm(direction)
        e = .12 * np.outer(direction, direction)
        covariance = np.sqrt(d[:, None] * d[None, :]) * (np.eye(8) + e)
        residuals = self.y[:, None] - self.phi
        diagonal_energy = (residuals ** 2 / d[:, None]).sum(0)
        colored_energy = (residuals * np.linalg.solve(covariance, residuals)).sum(0)
        epsilon = np.linalg.norm(e, 2)
        self.assertTrue(np.all(abs(colored_energy - diagonal_energy) <= epsilon / (1 - epsilon) * diagonal_energy + 1e-13))
        direct = -diagonal_energy / 2
        expanded = (self.phi * (self.y / d)[:, None]).sum(0) - (self.phi ** 2 / d[:, None]).sum(0) / 2
        np.testing.assert_allclose(direct - expanded, -np.square(self.y).dot(1 / d) / 2, atol=1e-13)

    def test_moment_closure_source_covariance_and_a_low_noise_failure(self):
        m = self.phi.shape[1]
        centred = self.phi - self.phi.mean(1, keepdims=True)
        covariance = centred @ centred.T / m
        for h in (0, 2, 3, 5):
            sums = count_states(m, h, "distinct") @ self.phi.T
            deviations = sums - sums.mean(0)
            actual = deviations.T @ deviations / len(sums)
            np.testing.assert_allclose(actual, h * (m - h) / (m - 1) * covariance, atol=1e-13)
        sums, prior, sigma2 = np.array([-2., 0., 2.]), np.array([.25, .5, .25]), .01
        self.assertEqual(prior @ sums ** 3, 0)
        evidence = lambda r: prior @ np.exp(-(r - sums) ** 2 / (2 * sigma2))
        exact_ratio = evidence(1.) / evidence(0.)
        gaussian_ratio = math.exp(-1 / (2 * (2 + sigma2)))
        self.assertGreater(gaussian_ratio / exact_ratio, 1e10)

    def test_action_multiplicity_caps_and_mean_readout_have_excess_copy_bounds(self):
        m, k = 5, 3
        states = count_states(m, k, "iid")
        rng = np.random.default_rng(281)
        for _ in range(8):
            posterior = rng.dirichlet(np.full(len(states), .2))
            gain = posterior @ np.minimum(states[:, None, :], states[None, :, :]).sum(2)
            optimal = gain.max()
            for cap in (1, 2, 3):
                restricted = gain[states.max(1) <= cap].max()
                excess = posterior @ np.maximum(states - cap, 0).sum(1)
                self.assertLessEqual(optimal - restricted, excess + 1e-13)
            means = posterior @ states
            chosen = np.argsort(-means, kind="stable")[:k]
            achieved = (posterior @ (states > 0))[chosen].sum()
            excess = posterior @ np.maximum(states - 1, 0).sum(1)
            self.assertLessEqual(optimal - achieved, excess + 1e-13)
        deterministic = np.r_[2, np.ones(18), 0]
        self.assertEqual(deterministic.sum(), 20)
        self.assertEqual(np.maximum(deterministic - 1, 0).sum() / 20, .05)
        self.assertTrue(np.any(deterministic > 1))

    def test_empirical_tail_decision_bound_with_exact_reference_samples(self):
        k = 3
        states, p, _ = enumerated_surrogate(self.phi, self.y, k, self.sigma2, list(itertools.combinations(range(5), 2)), "iid")
        exact = np.array([p @ (states >= j) for j in range(1, k + 1)]).T
        best = tail_action(exact, k)
        samples = states[np.random.default_rng(28).choice(len(states), 60, p=p)]
        estimated = np.array([(samples >= j).mean(0) for j in range(1, k + 1)]).T
        selected = tail_action(estimated, k)
        regret = p @ (np.minimum(states, best).sum(1) - np.minimum(states, selected).sum(1)) / k
        self.assertLessEqual(regret, 2 * abs(exact - estimated).max() + 1e-13)


if __name__ == "__main__":
    unittest.main()
