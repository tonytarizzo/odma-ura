"""Exhaustive source categories, exact physical objectives, and count moves for full-pattern continuation."""

from collections import Counter
from fractions import Fraction
from itertools import product
import math
import unittest

import numpy as np

from baselines.polar import PolarCode
from framework.polar_full_pattern import CountBank
from framework.polar_tractable import PolarModel, PolarScenario


def fixture():
    code = PolarCode(1, 2, 0)
    model = PolarModel(code, np.arange(8).reshape(4, 2), np.full((4, 2), 1 / np.sqrt(2)), 8)
    return model, np.stack([model.waveform(word) for word in range(8)])


def states(j, k):
    return [np.array(a, dtype=np.int64) for a in product(range(k + 1), repeat=j) if sum(a) <= k]


class FullPatternTest(unittest.TestCase):
    def test_exact_iid_source_categories_and_conditional_moments(self):
        model, atoms = fixture()
        scenario = PolarScenario(model, np.arange(8) / 8, 3, .2)
        bank = CountBank(scenario, [0, 2, 4])
        categories = Counter(tuple(Counter(words)[w] for w in bank.words) for words in product(range(8), repeat=3))
        total = Fraction(0)
        for count in states(3, 3):
            h = 3 - int(count.sum())
            expected = Fraction(math.factorial(3) * 5**h, 8**3 * math.factorial(h) * math.prod(math.factorial(int(c)) for c in count))
            self.assertEqual(expected, Fraction(categories[tuple(count)], 8**3))
            total += expected
        self.assertEqual(total, 1)
        outside = atoms[[1, 3, 5, 6, 7]]
        np.testing.assert_allclose(bank.mean, outside.mean(0), rtol=0, atol=1e-15)
        trace = np.mean(np.sum((outside - outside.mean(0))**2, axis=1))
        self.assertAlmostEqual(bank.kappa * model.n, trace, places=14)

    def test_sparse_best_move_agrees_with_exhaustive_common_objective(self):
        model, _ = fixture()
        scenario = PolarScenario(model, np.random.default_rng(8172).normal(size=8), 3, .2)
        bank = CountBank(scenario, [0, 2, 4])
        for count in states(3, 3):
            h = scenario.k - int(count.sum())
            residual = scenario.y - bank.matrix @ count - h * bank.mean
            best, _ = bank.gains(count, residual, {})
            base, gains = bank.objective(count), [0.]
            for new in range(3):
                if h:
                    proposal = count.copy(); proposal[new] += 1
                    gains.append(bank.objective(proposal) - base)
            for old in np.flatnonzero(count):
                proposal = count.copy(); proposal[old] -= 1
                gains.append(bank.objective(proposal) - base)
                for new in range(3):
                    if old != new:
                        proposal = count.copy(); proposal[old] -= 1; proposal[new] += 1
                        gains.append(bank.objective(proposal) - base)
            self.assertAlmostEqual(best[0], max(gains), places=12)

    def test_single_outside_word_gives_exact_full_source_posterior(self):
        model, atoms = fixture()
        scenario = PolarScenario(model, np.linspace(-.3, .8, 8), 2, .15)
        bank = CountBank(scenario, list(range(7)))
        np.testing.assert_allclose(bank.mean, atoms[7], rtol=0, atol=1e-15)
        self.assertLess(bank.kappa, 1e-15)
        for count in states(7, 2):
            h = 2 - int(count.sum())
            residual = scenario.y - count @ atoms[:7] - h * atoms[7]
            expected = -model.n / 2 * math.log(scenario.noise_var) - residual @ residual / (2 * scenario.noise_var)
            expected -= math.lgamma(h + 1) + sum(math.lgamma(int(c) + 1) for c in count)
            self.assertAlmostEqual(bank.objective(count), expected, places=12)

    def test_refit_removes_wrong_anchors_and_allows_repeated_message_counts(self):
        model, atoms = fixture()
        scenario = PolarScenario(model, 2 * atoms[0] + atoms[2], 3, .02)
        bank = CountBank(scenario, [0, 2, 4])
        action, meta = bank.refit(Counter({4: 3}))
        self.assertTrue(meta["converged"])
        self.assertEqual(action, Counter({0: 2, 2: 1}))
        optimum = max(bank.objective(c) for c in states(3, 3))
        self.assertAlmostEqual(meta["final_objective"], optimum, places=12)
        self.assertGreater(meta["final_objective"], meta["initial_objective"])
        np.testing.assert_allclose(bank.last_raw_residual, np.zeros(model.n), rtol=0, atol=1e-15)
        stopped, stopped_meta = bank.refit(Counter({4: 3}), max_moves=0)
        self.assertFalse(stopped_meta["converged"])
        self.assertTrue(stopped_meta["move_cap_hit"])
        self.assertEqual(stopped, Counter({4: 3}))

    def test_full_bank_eligibility_growth_and_invalid_counts(self):
        model, atoms = fixture()
        scenario = PolarScenario(model, atoms[6], 1, .02)
        bank = CountBank(scenario, [0, 2])
        fresh, _ = bank.extend([4, 6])
        self.assertEqual(fresh, [4, 6])
        action, meta = bank.refit(Counter())
        self.assertTrue(meta["converged"])
        self.assertEqual(action, Counter({6: 1}))
        self.assertEqual(meta["bank_size"], 4)
        with self.assertRaises(ValueError):
            bank.vector({0: 1.5})
        with self.assertRaises(ValueError):
            bank.objective(np.array([0., .5, 0., 0.]))
        with self.assertRaises(ValueError):
            bank.vector({0: -1})
        with self.assertRaises(ValueError):
            bank.vector({0: 2})
        with self.assertRaises(ValueError):
            bank.extend([1.5])


if __name__ == "__main__":
    unittest.main()
