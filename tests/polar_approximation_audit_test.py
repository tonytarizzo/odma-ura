"""Correctness checks for the proposal and finite approximation diagnostic, not a performance acceptance test."""

import itertools
import unittest

import numpy as np
from scipy.special import softmax

from baselines import message_bits
from baselines.polar import PolarCode
from tests.polar_approximation_audit import block_options, compact_moments, crc_constrained_list, encode_words, least_squares_gauge
from tests.polar_approximation_audit import polar_beam, remaining_moments
from tests.polar_assumptions_test import count_forest, enumerated_surrogate
from tests import polar_posterior_test as reference
from tests.polar_posterior_test import count_states


class ApproximationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.PolarPosteriorTests.setUpClass()
        cls.fixture = reference.PolarPosteriorTests

    def test_implicit_and_excluded_prior_moments_match_enumeration(self):
        f = self.fixture
        mean, second = compact_moments(f.base, f.energy)
        np.testing.assert_array_equal(encode_words(f.base, f.energy, np.arange(16)), f.phi)
        np.testing.assert_allclose(mean, f.phi.mean(1), atol=1e-15)
        np.testing.assert_allclose(second, (f.phi ** 2).mean(1), atol=1e-15)
        excluded = [1, 7]
        allowed = np.delete(f.phi, excluded, axis=1)
        for source in ("iid", "distinct"):
            shift, variance = remaining_moments(mean, second, 16, f.phi[:, excluded].T, 2, source)
            alphabet = allowed if source == "distinct" else f.phi
            states = count_states(alphabet.shape[1], 2, source)
            # IID count states are not uniformly weighted; enumerate ordered draws for this check.
            if source == "iid":
                sums = np.array([alphabet[:, words].sum(1) for words in itertools.product(range(16), repeat=2)])
            else:
                sums = states @ alphabet.T
            np.testing.assert_allclose(shift, sums.mean(0), atol=1e-14)
            np.testing.assert_allclose(variance, sums.var(0), atol=1e-14)

    def test_fixed_cardinality_gauge_preserves_full_law_and_four_word_star_is_exact(self):
        f = self.fixture
        phi = f.phi[:, [0, 3, 5, 12]]
        gram = phi.T @ phi / f.sigma2
        gauge = np.zeros(4)
        gauge[1] = (gram[1, 2] + gram[1, 3] - gram[2, 3]) / 2
        gauge[2] = gram[1, 2] - gauge[1]
        gauge[3] = gram[1, 3] - gauge[1]
        for sampling in ("iid", "distinct"):
            for k in (1, 2, 3, 4):
                states, p, z = enumerated_surrogate(phi, f.y, k, f.sigma2, list(itertools.combinations(range(4), 2)), sampling)
                probabilities, star_z = count_forest(phi, f.y, k, f.sigma2, [(0, 1), (0, 2), (0, 3)], sampling, gauge)
                expected = np.array([p @ (states == j) for j in range(k + 1)]).T
                np.testing.assert_allclose(probabilities, expected, rtol=2e-12, atol=1e-13)
                self.assertAlmostEqual(star_z / z, 1., places=12)
                arbitrary = np.array([-.6, .1, .4, .2])
                _, q, gauge_z = enumerated_surrogate(phi, f.y, k, f.sigma2,
                                                    list(itertools.combinations(range(4), 2)), sampling, arbitrary)
                np.testing.assert_allclose(q, p, atol=1e-13)
                self.assertAlmostEqual(gauge_z / z, 1., places=12)

    def test_least_squares_gauge_is_additive_projection_and_constant_coupling_is_removed(self):
        rng = np.random.default_rng(17)
        for size in (2, 3, 6):
            raw = rng.normal(size=(size, size))
            gram = raw + raw.T
            gauge = least_squares_gauge(gram)
            residual = gram - gauge[:, None] - gauge[None, :]
            np.fill_diagonal(residual, 0.)
            np.testing.assert_allclose(residual.sum(1), 0., atol=1e-13)
            if size <= 3:
                np.testing.assert_allclose(residual, 0., atol=1e-13)
        gram = np.full((8, 8), .7)
        np.fill_diagonal(gram, 1.)
        np.testing.assert_allclose(least_squares_gauge(gram), .35, atol=1e-14)

    def test_bounded_proposals_use_valid_complete_words_and_respect_source(self):
        f = self.fixture
        for source in ("iid", "distinct"):
            candidate, info = polar_beam(f.base, f.energy, f.y, 3, f.sigma2, source, 4, 16, True)
            self.assertGreater(len(candidate), 0)
            self.assertLessEqual(len(candidate), 12)
            self.assertTrue(np.all((candidate >= 0) & (candidate < 16)))
            self.assertLessEqual(info["polar_calls"], 4 * (1 + 2 * 4))
            self.assertLessEqual(info["complete_states"], 4)
        # Exhaustive internal polar paths recover the same best one-user physical word.
        candidate, _ = polar_beam(f.base, f.energy, f.y, 1, f.sigma2, "iid", 1, 64, True)
        self.assertEqual(candidate.tolist(), [int(np.argmax(f.phi.T @ f.y))])

    def test_crc_constraints_inside_search_preserve_likelihood_and_do_not_return_empty_lists(self):
        rng = np.random.default_rng(37)
        for payload_bits, length, crc in ((2, 8, 4), (6, 32, 8)):
            for size in (1, 2, 1 << payload_bits):
                code = PolarCode(payload_bits, length, crc, size)
                llr = rng.normal(size=length)
                payloads, metrics = crc_constrained_list(code, llr)
                self.assertEqual(len(payloads), size)
                coded = code.encode(payloads)
                exact_metric = np.logaddexp(0., -(1. - 2. * coded) * llr).sum(1)
                np.testing.assert_allclose(metrics, exact_metric, atol=1e-13)
                if size == 1 << payload_bits:
                    all_payloads = message_bits(np.arange(size), payload_bits)
                    scores = (1. - 2. * code.encode(all_payloads)) @ llr
                    np.testing.assert_array_equal(payloads, all_payloads[np.argsort(-scores, kind="stable")])

    def test_single_and_pair_gibbs_updates_preserve_ordered_physical_posterior(self):
        f = self.fixture
        phi, k = f.phi[:, :4], 3
        gram, linear = phi.T @ phi / f.sigma2, phi.T @ f.y / f.sigma2
        for source in ("iid", "distinct"):
            slots = np.array([s for s in itertools.product(range(4), repeat=k) if source == "iid" or len(set(s)) == k])
            p = softmax(-np.square(f.y - phi.T[slots].sum(1)).sum(1) / (2 * f.sigma2))
            index = {tuple(words): i for i, words in enumerate(slots)}
            for block in ((1,), (0, 2)):
                updated = np.zeros(len(slots))
                for words, mass in zip(slots, p):
                    options, probabilities = block_options(gram, linear, words, block, source)
                    for option, probability in zip(options, probabilities):
                        destination = words.copy()
                        destination[list(block)] = option
                        updated[index[tuple(destination)]] += mass * probability
                np.testing.assert_allclose(updated, p, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
