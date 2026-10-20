"""Finite invariance and interface checks for revisable polar proposals; no finite-run accuracy assertion."""

import itertools
import unittest

import numpy as np
from scipy.special import softmax

from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.polar_energy_geometry import balanced_energy
from tests.polar_approximation_audit import encode_words, posterior_tails
from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import count_states
from tests.polar_revisable_sampling import BlockProposal, restricted_update, revisable_tails, sparse_action, uniform_allowed


class RevisableSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = ODMAPolarBaseline(2, 10, 3218, prefix_bits=1, code_length=8, crc_bits=4, list_size=1)
        cls.energy, _ = balanced_energy(cls.base.positions, 10)
        cls.phi = encode_words(cls.base, cls.energy, np.arange(4))
        cls.y, cls.sigma2 = cls.phi[:, [1, 2, 3]].sum(1) + np.linspace(-.2, .3, 10), .3

    def test_uniform_allowed_rank_map_is_bijective_without_alphabet_scan(self):
        class Rank:
            def __init__(self, value):
                self.value = value

            def integers(self, size):
                assert self.value < size
                return self.value

        for excluded in (set(), {0, 2, 7}, {1, 3, 4, 6}):
            actual = [uniform_allowed(Rank(rank), 8, excluded) for rank in range(8 - len(excluded))]
            self.assertEqual(actual, [word for word in range(8) if word not in excluded])

    def test_single_and_sequential_pair_proposals_normalize_and_have_full_legal_support(self):
        for sampling, block in itertools.product(("iid", "distinct"), (1, 2)):
            fixed = [3] if block == 2 else [2, 3]
            residual = self.y - self.phi[:, fixed].sum(1)
            proposal = BlockProposal(self.base, self.energy, self.base.polar, residual, fixed,
                                     self.sigma2, sampling, .1, block)
            words = [row for row in itertools.product(range(4), repeat=block)
                     if sampling == "iid" or len(set((*row, *fixed))) == len(row) + len(fixed)]
            q = np.exp([proposal.log_probability(row) for row in words])
            self.assertTrue(np.all(q > 0))
            self.assertAlmostEqual(q.sum(), 1., places=13)
            rng = np.random.default_rng(24)
            self.assertTrue(all(proposal.draw(rng) in words for _ in range(30)))

    def test_hastings_correction_preserves_physical_conditional_for_both_sources(self):
        for sampling, block in itertools.product(("iid", "distinct"), (1, 2)):
            fixed = [3] if block == 2 else [2, 3]
            residual = self.y - self.phi[:, fixed].sum(1)
            proposal = BlockProposal(self.base, self.energy, self.base.polar, residual, fixed,
                                     self.sigma2, sampling, .15, block)
            words = [row for row in itertools.product(range(4), repeat=block)
                     if sampling == "iid" or len(set((*row, *fixed))) == len(row) + len(fixed)]
            p = softmax([-np.square(residual - self.phi[:, row].sum(1)).sum() / (2 * self.sigma2) for row in words])
            q = np.exp([proposal.log_probability(row) for row in words])
            transition = np.array([[q[j] * np.exp(proposal.log_acceptance(old, new)) for j, new in enumerate(words)]
                                   for old in words])
            transition[np.diag_indices(len(words))] += 1 - transition.sum(1)
            np.testing.assert_allclose(p @ transition, p, atol=2e-14)
            np.testing.assert_allclose(p[:, None] * transition, p[None, :] * transition.T, atol=2e-14)
            c = (q / p).min()
            self.assertTrue(np.all(transition >= c * p[None, :] - 2e-14))
            worst_tv = np.max(np.abs(np.linalg.matrix_power(transition, 3) - p[None, :]).sum(1) / 2)
            self.assertLessEqual(worst_tv, (1 - c) ** 3 + 2e-14)
            if block == 1:
                local, _, _ = proposal.context(())
                inside = np.array([row[0] in local for row in words])
                if np.any(inside):
                    bound = min(1 - proposal.epsilon, proposal.epsilon * p[inside].sum() / (len(words) * p[~inside].max())
                                if np.any(~inside) else 1.)
                    self.assertLessEqual(bound, c + 2e-14)

    def test_sparse_tail_action_and_fixed_budget_preserve_count_semantics(self):
        for source in ("iid", "distinct"):
            states = count_states(4, 3, source)
            posterior = np.random.default_rng(51).dirichlet(np.ones(len(states)))
            dense = posterior_tails(states, posterior, 3)
            sparse = {w: list(values[:1] if source == "distinct" else values) for w, values in enumerate(dense)}
            action = sparse_action(sparse, 3)
            np.testing.assert_array_equal([action.get(w, 0) for w in range(4)], tail_action(dense, 3))
            for mode in ("single", "mixed", "pair", "hybrid"):
                tails, work = revisable_tails(self.base, self.energy, self.y, 3, self.sigma2, source,
                    np.array([0, 1, 2]), 1, mode, 24, 22, chains=1)
                self.assertLessEqual(work["actual_contexts"], work["reserved_contexts"])
                self.assertLessEqual(work["reserved_contexts"], 24)
                self.assertAlmostEqual(sum(sum(values) for values in tails.values()), 3., places=13)
                self.assertEqual(sum(sparse_action(tails, 3).values()), 3)
                self.assertTrue(all(np.all(np.diff(values) <= 0) for values in tails.values()))
                if source == "distinct":
                    self.assertTrue(all(len(values) == 1 for values in tails.values()))

    def test_restricted_refresh_with_outside_self_loops_preserves_full_posterior(self):
        class Capture:
            def integers(self, size):
                return 0

            def choice(self, options, p):
                self.options, self.probabilities = options, p
                return options[0]

        candidates = np.array([0, 1])
        columns = self.phi[:, candidates]
        for source in ("iid", "distinct"):
            states = [s for s in itertools.product(range(4), repeat=2) if source == "iid" or len(set(s)) == 2]
            state_index = {s: i for i, s in enumerate(states)}
            p = softmax([-np.square(self.y - self.phi[:, s].sum(1)).sum() / (2 * self.sigma2) for s in states])
            transition = np.zeros((len(states), len(states)))
            for i, state in enumerate(states):
                slots, rng = np.array(state), Capture()
                signals = self.phi[:, slots].copy()
                restricted_update(slots, signals, signals.sum(1), candidates, columns, {0: 0, 1: 1},
                                  self.y, self.sigma2, source, rng)
                if state[0] not in candidates:
                    self.assertFalse(hasattr(rng, "options"))
                    transition[i, i] = 1.
                else:
                    for option, mass in zip(rng.options, rng.probabilities):
                        transition[i, state_index[(int(candidates[option]), state[1])]] += mass
            np.testing.assert_allclose(p @ transition, p, atol=2e-14)


if __name__ == "__main__":
    unittest.main()
