"""Whole-configuration normalization, orbit accounting and expected-risk controls."""

from collections import Counter
import itertools
import math
import unittest

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode
from framework.polar_decoders import run
from framework.polar_tractable import (ExchangeableProposal, PolarModel, PolarScenario, Work, beam_states,
                                      expected_is_regret, sparse_readout)
from tests.polar_tractable_execution_test import model100


class ExchangeableTests(unittest.TestCase):
    def fixture(self, k=2):
        model = PolarModel(PolarCode(2, 4, 0, 4), np.arange(4)[None], np.full((1, 4), .5), 4)
        scenario = PolarScenario(model, np.array([.4, -.7, .2, 1.1]), k, .6)
        work = Work(model)
        states, score = beam_states(scenario, work, 3, 1, 4, 'basis', 1, 4, np.random.default_rng(7))
        return scenario, work, states, score

    def test_global_envelope_and_mixture_for_every_tiny_ordered_tuple(self):
        for k in (1, 2, 3):
            s, work, states, score = self.fixture(k)
            tuples = list(itertools.product(range(4), repeat=k))
            logf = np.array([-np.square(s.y-sum((s.model.waveform(w) for w in words), np.zeros(4))).sum() / 1.2
                             - k * math.log(4) for words in tuples])
            for mixing in (0., .5, .9):
                q = ExchangeableProposal(s, work, states, score, mixing, 1, 4)
                logq = np.array([q.logq(w) for w in tuples])
                self.assertAlmostEqual(logsumexp(logq), 0., places=12)
                self.assertTrue(np.all(logf-logq <= q.log_weight_bound+1e-12))
                beam_mask = [tuple(sorted(w)) in states for w in tuples]
                self.assertAlmostEqual(logsumexp(logf[beam_mask]), q.log_evidence_lower, places=12)
                for words in tuples:
                    self.assertAlmostEqual(q.logq(words), q.logq(tuple(reversed(words))), places=12)
                exact_weights = np.exp(logq) * np.exp(logf-logq-logsumexp(logf))
                direct_weights = np.exp(logf-logsumexp(logf))
                actual = dict(sparse_readout(tuples, exact_weights, k)[1])
                expected = dict(sparse_readout(tuples, direct_weights, k)[1])
                self.assertEqual(actual.keys(), expected.keys())
                for word in expected:
                    np.testing.assert_allclose(actual[word], expected[word], atol=1e-12)

    def test_union_contains_both_candidate_rules(self):
        from framework.polar_tractable import candidates
        s, work, _, _ = self.fixture()
        a = set(candidates(work, s.y, .6, 1, 1, 'scl'))
        b = set(candidates(work, s.y, .6, 1, 1, 'basis', 0, 1))
        self.assertEqual(set(candidates(work, s.y, .6, 1, 1, 'union', 0, 1)), a | b)

    def test_expected_bound_dominates_exact_sampling_regret(self):
        # A two-word K1 law permits exact expectation over every H-draw outcome.
        p = np.array([.8, .2]); q = np.array([.4, .6]); kappa = max(p/q)
        for h in (1, 4, 10, 128):
            regret = 0.
            for count in range(h+1):
                prob = math.comb(h, count)*q[0]**count*q[1]**(h-count)
                chosen = int(count*p[0]/q[0] < (h-count)*p[1]/q[1])
                regret += prob*(max(p)-p[chosen])
            self.assertLessEqual(regret, expected_is_regret(math.log(kappa), h, 1, 1))
        self.assertLess(expected_is_regret(0., 10000, 1, 1), .04)

    def test_new_decoders_execute_at_b100_and_counts_are_legal(self):
        model = model100(); word = (1 << 99) + (1 << 77) + 13579
        s = PolarScenario(model, model.waveform(word), 1, 1e-4)
        for name in ('DEC-33', 'DEC-34'):
            params = {'width': 2, 'patterns': 2, 'list_size': 4, 'max_refine': 4, 'seed': 18}
            if name == 'DEC-33':
                params['samples'] = 16
            counts, meta = run(name, s, **params)
            self.assertEqual(counts, {word: 1})
            self.assertFalse(meta['legacy_dense_adapter'])
            if name == 'DEC-33':
                self.assertTrue(0 <= meta['expected_excess_risk_bound'] <= 1)
            pair = PolarScenario(model, model.waveform(word)+model.waveform(7), 2, .02)
            first, _ = run(name, pair, **params)
            second, _ = run(name, pair, **params)
            self.assertEqual(first, second)
            self.assertEqual(sum(first.values()), 2)


if __name__ == '__main__':
    unittest.main()
