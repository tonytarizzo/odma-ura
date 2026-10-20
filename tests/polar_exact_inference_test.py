"""Finite exact-decoder certificates for Report 7; intentionally exponential, not a benchmark receiver."""

import itertools
import math
import unittest
from collections import defaultdict

import numpy as np

from tests import polar_posterior_test as reference
from tests.polar_posterior_test import count_states, exact_posterior


def convolve_banks(left, right, k):
    """Positive discrete measures, keyed by exact integer signal signatures and indexed by cardinality."""
    result = [defaultdict(float) for _ in range(k + 1)]
    for total in range(k + 1):
        for q in range(total + 1):
            for s, a in left[q].items():
                for t, b in right[total - q].items():
                    result[total][tuple(x + y for x, y in zip(s, t))] += a * b
    return result


def exact_tree_decode(signatures, embedding, y, k, sigma2, sampling):
    """Full inside/outside sum, Gaussian weighting, count tails, and minimum-multiset-PUPE output."""
    m, dimension = signatures.shape
    zero = (0,) * dimension

    def leaf_bank(w):
        bank = [dict() for _ in range(k + 1)]
        for q in range(min(k, 1) + 1 if sampling == "distinct" else k + 1):
            bank[q][tuple(q * signatures[w])] = 1 / math.factorial(q)
        return bank

    def build(words):
        if len(words) == 1:
            return {"words": words, "inside": leaf_bank(words[0])}
        mid = len(words) // 2
        left, right = build(words[:mid]), build(words[mid:])
        return {"words": words, "left": left, "right": right,
                "inside": convolve_banks(left["inside"], right["inside"], k)}

    def weight(signature):
        residual = y - embedding @ np.asarray(signature)
        return math.exp(-residual @ residual / (2 * sigma2))

    root = build(list(range(m)))
    evidence = sum(a * weight(s) for s, a in root["inside"][k].items())
    probabilities = np.zeros((m, k + 1))

    def outside_pass(node, outside):
        if len(node["words"]) == 1:
            w = node["words"][0]
            for q in range(k + 1):
                for s, a in node["inside"][q].items():
                    for t, b in outside[k - q].items():
                        probabilities[w, q] += a * b * weight(tuple(x + z for x, z in zip(s, t)))
            probabilities[w] /= evidence
            return
        for child, sibling in (("left", "right"), ("right", "left")):
            outside_pass(node[child], convolve_banks(outside, node[sibling]["inside"], k))

    outside_pass(root, [{zero: 1.0}] + [dict() for _ in range(k)])
    tails = np.flip(np.cumsum(np.flip(probabilities[:, 1:], axis=1), axis=1), axis=1)
    decoded = np.zeros(m, dtype=int)
    for _ in range(k):
        available = [tails[w, decoded[w]] if decoded[w] < k else -np.inf for w in range(m)]
        decoded[np.argmax(available)] += 1
    return decoded, probabilities, evidence, len(root["inside"][k])


def ordered_evidence(phi, r, k, sigma2):
    if k == 0:
        return math.exp(-r @ r / (2 * sigma2))
    return sum(ordered_evidence(phi, r - column, k - 1, sigma2) for column in phi.T)


class ExactInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.PolarPosteriorTests.setUpClass()
        cls.fixture = reference.PolarPosteriorTests

    def test_recursive_evidence_count_probabilities_and_factorial_moments(self):
        f = self.fixture
        states, posterior = exact_posterior(f.phi, f.y, 3, f.sigma2, "iid")
        z = ordered_evidence(f.phi, f.y, 3, f.sigma2)
        for w in (0, 2, 6, 13):
            q = []
            other = np.delete(f.phi, w, axis=1)
            for j in range(4):
                residual = f.y - j * f.phi[:, w]
                value = math.comb(3, j) * ordered_evidence(other, residual, 3 - j, f.sigma2) / z
                q.append(value)
                self.assertAlmostEqual(value, posterior @ (states[:, w] == j), places=13)
                moment = math.comb(3, j) * ordered_evidence(f.phi, residual, 3 - j, f.sigma2) / z
                direct = posterior @ np.array([math.comb(int(a), j) if a >= j else 0 for a in states[:, w]])
                self.assertAlmostEqual(moment, direct, places=13)
            self.assertAlmostEqual(sum(q), 1, places=13)

    def test_polar_signature_inside_outside_full_decoder(self):
        f = self.fixture
        signatures = np.zeros((16, 32), dtype=int)
        embedding = np.zeros((12, 32))
        for p in range(4):
            signatures[4 * p:4 * p + 4, 8 * p:8 * p + 8] = f.signs.astype(int)
            embedding[f.base.positions[p], 8 * p + np.arange(8)] = np.sqrt(f.energy[p])
        np.testing.assert_allclose(embedding @ signatures.T, f.phi, atol=1e-15)
        for sampling, k in itertools.product(("iid", "distinct"), (1, 3)):
            decoded, q, z, _ = exact_tree_decode(signatures, embedding, f.y, k, f.sigma2, sampling)
            states, posterior = exact_posterior(f.phi, f.y, k, f.sigma2, sampling)
            exact_q = np.stack([posterior @ (states == j) for j in range(k + 1)], axis=1)
            np.testing.assert_allclose(q, exact_q, atol=2e-14)
            np.testing.assert_allclose(q.sum(1), 1, atol=2e-14)
            coefficient = np.array([1 / math.prod(math.factorial(int(a)) for a in row) for row in states])
            likelihood = np.exp(-np.square(states @ f.phi.T - f.y).sum(1) / (2 * f.sigma2))
            self.assertAlmostEqual(z, coefficient @ likelihood, places=13)
            risk = 1 - posterior @ np.minimum(states, decoded).sum(1) / k
            all_risks = [1 - posterior @ np.minimum(states, a).sum(1) / k for a in count_states(16, k, "iid")]
            self.assertAlmostEqual(risk, min(all_risks), places=13)

    def test_equal_signal_merging_preserves_distinct_message_count_queries(self):
        # Deliberate encoder ambiguity: two messages share a waveform, but their counts are separate queries.
        signatures = np.array([[1], [1], [-1]])
        embedding = np.ones((1, 1))
        y, k, sigma2 = np.array([.7]), 3, .4
        for sampling in ("iid", "distinct"):
            _, q, _, atoms = exact_tree_decode(signatures, embedding, y, k, sigma2, sampling)
            states, posterior = exact_posterior(embedding @ signatures.T, y, k, sigma2, sampling)
            expected = np.stack([posterior @ (states == j) for j in range(k + 1)], axis=1)
            np.testing.assert_allclose(q, expected, atol=1e-14)
            self.assertLessEqual(atoms, len(states))
            if sampling == "iid":
                self.assertLess(atoms, len(states))
                self.assertGreater(q[0, 1], 0)
                self.assertLess(q[0, 1], 1)

    def test_grouped_polar_walsh_scores_and_association_obstruction(self):
        f = self.fixture
        generator = f.base.polar.encode(np.eye(2, dtype=np.uint8)).T
        llr = np.linspace(-1.2, 2.3, 8)
        indices = generator @ np.array([2, 1])
        fields = np.bincount(indices, weights=llr / 2, minlength=4)
        transformed = fields.copy()
        width = 1
        while width < len(transformed):
            for start in range(0, len(transformed), 2 * width):
                a = transformed[start:start + width].copy()
                b = transformed[start + width:start + 2 * width].copy()
                transformed[start:start + width] = a + b
                transformed[start + width:start + 2 * width] = a - b
            width *= 2
        np.testing.assert_allclose(transformed, f.signs @ llr / 2, atol=1e-14)
        first = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        second = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(first.sum(0), second.sum(0))
        self.assertNotEqual(int(np.bitwise_xor(first[:, 0], first[:, 1]).sum()),
                            int(np.bitwise_xor(second[:, 0], second[:, 1]).sum()))

    def test_positive_subtree_bounds_and_bayes_decision_certificate(self):
        f = self.fixture
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(f.phi, f.y, 3, f.sigma2, sampling)
            weights = np.exp(-np.square(states @ f.phi.T - f.y).sum(1) / (2 * f.sigma2))
            weights /= np.array([math.prod(math.factorial(int(a)) for a in row) for row in states])
            z = weights.sum()
            numerator = np.stack([weights @ (states >= j) for j in range(1, 4)], axis=1)
            # A complete disjoint frontier after assigning the first two full-message counts.
            blocks = {}
            for i, a in enumerate(states):
                blocks.setdefault(tuple(a[:2]), []).append(i)
            lower, upper = np.zeros_like(numerator), np.zeros_like(numerator)
            upper_z = 0.0
            for prefix, indices in blocks.items():
                h = 3 - sum(prefix)
                r = f.y - f.phi[:, :2] @ prefix
                prior_mass = (14 ** h / math.factorial(h) / math.prod(math.factorial(a) for a in prefix)
                              if sampling == "iid" else math.comb(14, h))
                bound = prior_mass * math.exp(-max(0, np.linalg.norm(r) - h) ** 2 / (2 * f.sigma2))
                self.assertLessEqual(weights[indices].sum(), bound + 1e-13)
                upper_z += bound
                possible = np.stack([(states[indices] >= j).any(0) for j in range(1, 4)], axis=1)
                upper += bound * possible
            self.assertLessEqual(z, upper_z + 1e-13)
            self.assertTrue(np.all(numerator <= upper + 1e-13))
            # Resolve blocks one by one; whenever certified, independently verify optimality.
            certificates, evaluated_evidence = 0, 0.0
            for prefix, indices in blocks.items():
                h, r = 3 - sum(prefix), f.y - f.phi[:, :2] @ prefix
                prior_mass = (14 ** h / math.factorial(h) / math.prod(math.factorial(a) for a in prefix)
                              if sampling == "iid" else math.comb(14, h))
                bound = prior_mass * math.exp(-max(0, np.linalg.norm(r) - h) ** 2 / (2 * f.sigma2))
                possible = np.stack([(states[indices] >= j).any(0) for j in range(1, 4)], axis=1)
                exact = np.stack([weights[indices] @ (states[indices] >= j) for j in range(1, 4)], axis=1)
                lower += exact
                upper += exact - bound * possible
                evaluated_evidence += weights[indices].sum()
                self.assertTrue(np.all(lower <= numerator + 1e-12))
                self.assertTrue(np.all(numerator <= upper + 1e-12))
                action = np.zeros(16, dtype=int)
                for _ in range(3):
                    action[np.argmax([lower[w, action[w]] if action[w] < 3 else -np.inf for w in range(16)])] += 1
                selected = min(lower[w, action[w] - 1] for w in range(16) if action[w] > 0)
                unselected = max(upper[w, action[w]] for w in range(16) if action[w] < 3)
                achieved = sum(numerator[w, :action[w]].sum() for w in range(16))
                optimum = np.sort(numerator.ravel())[-3:].sum()
                achieved_lower = sum(lower[w, :action[w]].sum() for w in range(16))
                risk_bound = (np.sort(upper.ravel())[-3:].sum() - achieved_lower) / (3 * evaluated_evidence)
                self.assertLessEqual((optimum - achieved) / (3 * z), risk_bound + 1e-12)
                if selected >= unselected - 1e-13:
                    certificates += 1
                    self.assertAlmostEqual(achieved, optimum, places=12)
            self.assertGreater(certificates, 0)
            np.testing.assert_allclose(lower / z, np.stack([posterior @ (states >= j) for j in range(1, 4)], axis=1))


if __name__ == "__main__":
    unittest.main()
