"""Exact Fourier, elimination, and connected-group certificates; tiny systems only."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import softmax

from tests import polar_posterior_test as reference
from tests.polar_posterior_test import exact_posterior


def fourier_count_query(signatures, embedding, y, k, sigma2, sampling, marked):
    dimension, side = signatures.shape[1], 2 * k + 1
    grid = np.array(list(itertools.product(range(side), repeat=dimension)))
    phase = np.exp(-2j * np.pi * grid @ signatures.T / side)
    transform = np.zeros((len(grid), k + 1, k + 1), dtype=complex)
    transform[:, 0, 0] = 1
    for w in range(len(signatures)):
        updated = np.zeros_like(transform)
        for j in range(2 if sampling == "distinct" else k + 1):
            degree = j if w == marked else 0
            coefficient = phase[:, w] ** j / math.factorial(j)
            updated[:, j:, degree:] += coefficient[:, None, None] * transform[:, :k + 1 - j, :k + 1 - degree]
        transform = updated
    inverse = np.exp(2j * np.pi * grid @ grid.T / side) / len(grid)
    atoms = inverse @ transform[:, k, :]
    centered = np.where(grid <= k, grid, grid - side)
    signal = centered @ embedding.T
    gaussian = np.exp(-np.square(signal - y).sum(1) / (2 * sigma2))
    numerator = gaussian @ atoms
    return numerator, atoms


def eliminate_factors(factors, order, sizes):
    factors = [(tuple(scope), np.asarray(value, dtype=complex)) for scope, value in factors]
    largest = 1
    for variable in order:
        relevant = [factor for factor in factors if variable in factor[0]]
        factors = [factor for factor in factors if variable not in factor[0]]
        scope = tuple(sorted(set().union(*(set(s) for s, _ in relevant))))
        combined = np.ones([sizes[v] for v in scope], dtype=complex)
        for local, value in relevant:
            shape = [sizes[v] if v in local else 1 for v in scope]
            combined *= value.reshape(shape)
        largest = max(largest, combined.size)
        result = combined.sum(axis=scope.index(variable))
        factors.append((tuple(v for v in scope if v != variable), result))
    return math.prod(value.item() for _, value in factors), largest


def elimination_count_query(phi, y, k, sigma2, sampling, marked, order):
    m = phi.shape[1]
    unary = softmax(phi.T @ y / sigma2)
    pair = np.exp(-(phi.T @ phi) / sigma2)
    if sampling == "distinct":
        np.fill_diagonal(pair, 0)
    transformed = []
    for ell in range(k + 1):
        phase = np.exp(-2j * np.pi * ell / (k + 1))
        marked_unary = unary.astype(complex)
        marked_unary[marked] *= phase
        factors = [((i,), marked_unary) for i in range(k)]
        factors.extend(((i, j), pair) for i, j in itertools.combinations(range(k), 2))
        result, _ = eliminate_factors(factors, order, [m] * k)
        transformed.append(result)
    return np.fft.ifft(transformed)


def connected_edge_sets(m):
    edges = list(itertools.combinations(range(m), 2))
    for mask in range(1 << len(edges)):
        chosen = [edge for bit, edge in enumerate(edges) if mask & (1 << bit)]
        seen = {0}
        while True:
            expanded = seen | {v for i, j in chosen for u, v in ((i, j), (j, i)) if u in seen}
            if expanded == seen:
                break
            seen = expanded
        if len(seen) == m:
            yield chosen


def connected_polynomials(phi, y, k, sigma2, sampling, marked):
    q = softmax(phi.T @ y / sigma2)
    pair = np.exp(-(phi.T @ phi) / sigma2)
    if sampling == "distinct":
        np.fill_diagonal(pair, 0)
    interaction = pair - 1
    connected = [np.zeros(k + 1)]
    for size in range(1, k + 1):
        assignments = np.array(list(itertools.product(range(len(q)), repeat=size)))
        prior_weight = np.prod(q[assignments], axis=1)
        degree = (assignments == marked).sum(1)
        connected_weight = np.zeros(len(assignments))
        for edges in connected_edge_sets(size):
            term = np.ones(len(assignments))
            for i, j in edges:
                term *= interaction[assignments[:, i], assignments[:, j]]
            connected_weight += term
        connected.append(np.bincount(degree, weights=prior_weight * connected_weight, minlength=k + 1))
    return connected


def assemble_connected(connected, k, maximum_block=None):
    maximum_block = k if maximum_block is None else maximum_block
    partition = [np.r_[1.0, np.zeros(k)]]
    for size in range(1, k + 1):
        value = np.zeros(k + 1)
        for block in range(1, min(size, maximum_block) + 1):
            value += math.comb(size - 1, block - 1) * np.convolve(connected[block], partition[size - block])[:k + 1]
        partition.append(value)
    return partition[k]


class AlternativeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.PolarPosteriorTests.setUpClass()
        cls.fixture = reference.PolarPosteriorTests

    def test_finite_fourier_count_queries_and_no_aliasing(self):
        signatures = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        phi, y, k, sigma2 = signatures.T, np.array([.3, -.7]), 3, .8
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(phi, y, k, sigma2, sampling)
            for marked in range(4):
                numerator, atoms = fourier_count_query(signatures, np.eye(2), y, k, sigma2, sampling, marked)
                expected = np.array([posterior @ (states[:, marked] == j) for j in range(k + 1)])
                np.testing.assert_allclose(numerator.imag, 0, atol=1e-13)
                np.testing.assert_allclose(numerator.real / numerator.real.sum(), expected, atol=1e-13)
                self.assertGreaterEqual(atoms.real.min(), -1e-13)
                total_mass = 4 ** k / math.factorial(k) if sampling == "iid" else math.comb(4, k)
                self.assertAlmostEqual(atoms.real.sum(), total_mass, places=12)
        # All admissible integer coordinates have different residues on the chosen grid.
        self.assertEqual(len(set(i % (2 * k + 1) for i in range(-k, k + 1))), 2 * k + 1)
        side = 2 * k + 1
        grid = np.array(list(itertools.product(range(side), repeat=2)))
        centered = np.where(grid <= k, grid, grid - side)
        phase = np.exp(-2j * np.pi * grid @ signatures.T / side)
        characteristic = phase.mean(1)
        self.assertLessEqual(np.abs(characteristic).max(), 1 + 1e-14)
        measure_transform = phase.sum(1) ** k / math.factorial(k)
        gaussian = np.exp(-np.square(centered - y).sum(1) / (2 * sigma2))
        gaussian_dual = np.exp(2j * np.pi * grid @ grid.T / side) @ gaussian
        terms = measure_transform * gaussian_dual / len(grid)
        evidence, _ = fourier_count_query(signatures, np.eye(2), y, k, sigma2, "iid", 0)
        self.assertAlmostEqual(terms.sum().real, evidence.sum().real, places=12)
        keep = np.square(centered).sum(1) <= 1
        bound = np.abs(terms[~keep]).sum()
        self.assertLessEqual(abs(terms.sum() - terms[keep].sum()), bound + 1e-13)

    def test_exact_elimination_count_queries_and_orders(self):
        f = self.fixture
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(f.phi, f.y, 3, f.sigma2, sampling)
            for marked in (2, 6, 13):
                expected = np.array([posterior @ (states[:, marked] == j) for j in range(4)])
                for order in ((0, 1, 2), (2, 0, 1)):
                    numerator = elimination_count_query(f.phi, f.y, 3, f.sigma2, sampling, marked, order)
                    np.testing.assert_allclose(numerator / numerator.sum(), expected, atol=2e-13)
        # A chain makes the cost of a bad elimination order visible without changing the exact answer.
        rng = np.random.default_rng(81)
        factors = [((i,), rng.random(3) + .2) for i in range(4)]
        factors += [((i, i + 1), rng.random((3, 3)) + .2) for i in range(3)]
        left, left_size = eliminate_factors(factors, (0, 1, 2, 3), [3] * 4)
        middle, middle_size = eliminate_factors(factors, (1, 2, 0, 3), [3] * 4)
        self.assertAlmostEqual(left.real, middle.real, places=12)
        self.assertEqual((left_size, middle_size), (9, 27))

    def test_connected_group_expansion_matches_full_count_law(self):
        f = self.fixture
        phi, k, sigma2 = f.phi[:, :5], 4, 1.7
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(phi, f.y, k, sigma2, sampling)
            for marked in (0, 2):
                connected = connected_polynomials(phi, f.y, k, sigma2, sampling, marked)
                numerator = assemble_connected(connected, k)
                expected = np.array([posterior @ (states[:, marked] == j) for j in range(k + 1)])
                np.testing.assert_allclose(numerator / numerator.sum(), expected, atol=3e-12)
                elimination = elimination_count_query(phi, f.y, k, sigma2, sampling, marked, (0, 1, 2, 3))
                np.testing.assert_allclose(numerator, elimination.real, atol=3e-14)
                # Discarding larger connected blocks is a change of model, not an exact pairwise closure.
                truncated = assemble_connected(connected, k, maximum_block=2)
                self.assertGreater(np.max(np.abs(truncated - numerator)), 1e-5)

    def test_low_order_connected_truncation_needs_an_error_bound(self):
        f = self.fixture
        phi, k, marked = f.phi[:, :5], 4, 0
        for sigma2 in (5.0, .5):
            connected = connected_polynomials(phi, f.y, k, sigma2, "iid", marked)
            full = assemble_connected(connected, k)
            truncated = assemble_connected(connected, k, maximum_block=3)
            epsilon = np.max(np.abs(np.exp(-(phi.T @ phi) / sigma2) - 1))
            edges = math.comb(k, 2)
            omitted_bound = sum(math.comb(edges, j) * epsilon ** j for j in range(3, edges + 1))
            self.assertLessEqual(np.abs(full - truncated).sum(), omitted_bound + 1e-12)
            states, posterior = exact_posterior(phi, f.y, k, sigma2, "iid")
            expected = np.array([posterior @ (states[:, marked] == j) for j in range(k + 1)])
            np.testing.assert_allclose(full / full.sum(), expected, atol=2e-12)
            if sigma2 == 5.0:
                tails = lambda a: np.flip(np.cumsum(np.flip(a / a.sum())))[1:]
                self.assertLess(np.max(np.abs(tails(full) - tails(truncated))), .0005)
            else:
                self.assertGreater(truncated.sum(), 0)
                self.assertLess(truncated.min(), -.02)


if __name__ == "__main__":
    unittest.main()
