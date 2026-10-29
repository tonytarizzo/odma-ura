"""Exhaustive finite checks for procedural polar proposals and count decisions."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode
from benchmarks.polar_decoders import build_encoder, normalized_config
from framework.polar_decoder_core import Scenario, allocate, count_tails
from framework.polar_decoders import run
from framework.polar_tractable import (ListProposal, ParityProposal, ParityReference, PolarModel, PolarScenario, Work,
                                      bits, integer, sparse_readout, tractable_beam)


class ReplayRNG:
    """Replay an interior point of each finite sampler branch, with its exact mass checked separately."""

    def __init__(self, uniforms, choice=None):
        self.uniforms, self.choice_value, self.used = uniforms, choice, 0

    def random(self):
        value = self.uniforms[self.used]
        self.used += 1
        return value

    def choice(self, size, p):
        index, expected = self.choice_value
        np.testing.assert_allclose(p, expected, atol=2e-14, rtol=1e-13)
        assert size == len(expected)
        return index


class TractableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = PolarCode(3, 4, 0, 8)
        positions = np.array([[0, 1, 2, 3], [1, 2, 4, 5]])
        amplitudes = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]])
        amplitudes /= np.linalg.norm(amplitudes, axis=1, keepdims=True)
        cls.model = PolarModel(code, positions, amplitudes, 6)
        cls.phi = np.stack([cls.model.waveform(w) for w in range(1 << cls.model.B)], axis=1)
        cls.residual = np.array([.6, -.4, 1.2, -.7, .3, -.2])

    def test_procedural_waveforms_and_mean_match_existing_dense_encoder(self):
        for mapping in ("uniform", "balanced"):
            config = normalized_config({"decoders": ["REF"], "loads": [1], "mapping": mapping})
            base, encoder, phi, _ = build_encoder(config)
            model = PolarModel(base.polar, base.positions, encoder.codebook.amplitudes().detach().numpy(), base.n)
            actual = np.stack([model.waveform(w) for w in range(1 << model.B)], axis=1)
            np.testing.assert_allclose(actual, phi, atol=2e-15)
            np.testing.assert_allclose(model.mean, phi.mean(axis=1), atol=2e-15)
            for word in range(1 << model.B):
                pattern, payload = model.split(word)
                self.assertEqual(model.word(pattern, bits(payload, model.b)), word)
                self.assertEqual(integer(bits(word, model.B)), word)

    def test_parity_inverse_roundtrip_and_physical_scores(self):
        for variance in (.03, .7, 20.):
            for pattern in range(self.model.p):
                ref = ParityReference(self.model, pattern, self.residual, variance, Work(self.model))
                for payload in range(1 << self.model.b):
                    coordinate = ref.coordinates(payload)
                    self.assertEqual(ref.payload(coordinate), payload)
                    self.assertEqual(ref.coordinates(ref.payload(payload)), payload)
                    word = (pattern << self.model.b) | payload
                    self.assertAlmostEqual(ref.score(payload), self.residual @ self.model.waveform(word) / variance, places=12)

    def test_parity_envelope_normalization_and_nested_repairs(self):
        for variance in (.07, .8, 10.):
            previous = np.inf
            for budget in (0, 1, 3, 1 << self.model.b):
                ref = ParityReference(self.model, 1, self.residual, variance, Work(self.model), self.model.b,
                                      max(1, budget), bool(budget))
                payloads = range(1 << self.model.b)
                scores = np.array([ref.score(v) for v in payloads])
                envelope = np.array([ref.envelope(v) for v in payloads])
                logq0 = np.array([ref.base_logq(v) for v in payloads])
                self.assertAlmostEqual(logsumexp(logq0), 0., places=13)
                self.assertAlmostEqual(logsumexp(envelope), ref.log_bound, places=12)
                self.assertTrue(np.all(envelope >= scores - 2e-13))
                self.assertLessEqual(logsumexp(scores), ref.log_bound + 2e-13)
                self.assertLessEqual(ref.log_bound, previous + 2e-13)
                previous = ref.log_bound
                remaining = [v for v in payloads if v not in ref.explored]
                if remaining:
                    self.assertAlmostEqual(logsumexp(logq0[remaining]), ref.log_tau, places=12)
                else:
                    self.assertEqual(ref.log_tau, -np.inf)
                    self.assertAlmostEqual(logsumexp(scores), ref.log_bound, places=12)
                for v in ref.explored:
                    self.assertEqual(ref.envelope(v), ref.score(v))

    def test_exact_sampler_branch_law_matches_repaired_envelope(self):
        for budget in (0, 1, 3, 1 << self.model.b):
            ref = ParityReference(self.model, 1, self.residual, .8, Work(self.model), self.model.b,
                                  max(1, budget), bool(budget))
            count = 1 << self.model.b
            q0 = np.exp([ref.base_logq(v) for v in range(count)])
            coordinates = np.array([ref.coordinates(v) for v in range(count)])
            explored_mass = math.exp(ref.log_za - ref.log_bound)
            law = []
            for payload in range(count):
                if payload in ref.explored:
                    exact_weights = np.exp(np.array([ref.score(v) for v in ref.explored_ids]) - ref.log_za)
                    index = ref.explored_ids.index(payload)
                    rng = ReplayRNG([explored_mass / 2], (index, exact_weights))
                    mass = explored_mass * exact_weights[index]
                else:
                    uniforms = [(1 + explored_mass) / 2] if ref.explored else []
                    remaining = np.array([v not in ref.explored for v in range(count)])
                    mass = 1 - explored_mass
                    for bit in range(self.model.b):
                        one = ((coordinates >> bit) & 1).astype(bool)
                        probability = q0[remaining & one].sum() / q0[remaining].sum()
                        value = bool((coordinates[payload] >> bit) & 1)
                        uniforms.append(probability / 2 if value else (1 + probability) / 2)
                        mass *= probability if value else 1 - probability
                        remaining &= one if value else ~one
                    rng = ReplayRNG(uniforms)
                self.assertEqual(ref.draw(rng), payload)
                self.assertEqual(rng.used, len(rng.uniforms))
                law.append(mass)
            np.testing.assert_allclose(law, np.exp([ref.envelope(v) - ref.log_bound for v in range(count)]), atol=2e-14)
            self.assertAlmostEqual(sum(law), 1., places=13)

    def test_pattern_proposal_keeps_physical_cross_pattern_constants(self):
        for repair in (False, True):
            q = ParityProposal(Work(self.model), self.residual, .4, repair, 2, 4)
            words = range(1 << self.model.B)
            probability = np.exp([q.logq(w) for w in words])
            self.assertAlmostEqual(probability.sum(), 1., places=13)
            for word in words:
                pattern, payload = self.model.split(word)
                self.assertAlmostEqual(q.logq(word), q.references[pattern].envelope(payload) - q.log_z, places=13)
            exact = ParityProposal(Work(self.model), self.residual, .4, True, self.model.b, 1 << self.model.b)
            physical = self.residual @ self.phi / .4
            np.testing.assert_allclose([exact.logq(w) for w in words], physical - logsumexp(physical), atol=2e-14)

    def test_sparse_count_tails_and_allocation_match_dense_with_collisions(self):
        states = [(0, 0, 0), (0, 2, 5), (2, 2, 5), (0, 5, 5)]
        weights = np.array([.15, .2, .4, .25])
        expected = count_tails(np.array(states), weights, 8)
        counts, rows = sparse_readout(states, weights, 3)
        actual = np.zeros_like(expected)
        for word, tails in rows:
            actual[word, :len(tails)] = tails
        np.testing.assert_allclose(actual, expected, atol=2e-15)
        hard = np.zeros(8, dtype=int)
        for word, count in counts.items():
            hard[word] = count
        np.testing.assert_array_equal(hard, allocate(expected, 3))
        counts, _ = sparse_readout([(5, 5), (2, 2)], [.5, .5], 2)
        self.assertEqual(counts, {2: 2})

    def test_full_width_beams_recover_exact_count_posterior(self):
        for method in ("scl", "basis"):
            for k in (1, 2, 3):
                y = self.residual + self.phi[:, [1] * k].sum(axis=1)
                scenario = PolarScenario(self.model, y, k, .6)
                width = math.comb((1 << self.model.B) + k - 1, k)
                counts, meta = tractable_beam(scenario, width=width, patterns=self.model.p, list_size=8,
                                             flip_order=self.model.b, max_refine=8, candidate_method=method)
                expected_counts, ref = run("REF", Scenario(self.phi, y, k, .6))
                actual_tails = np.zeros_like(ref["tails"])
                actual_counts = np.zeros(len(expected_counts), dtype=int)
                for word, tails in meta["tail_probabilities"]:
                    actual_tails[word, :len(tails)] = tails
                for word, count in counts.items():
                    actual_counts[word] = count
                self.assertEqual(meta["retained_states"], width)
                np.testing.assert_allclose(actual_tails, ref["tails"], atol=5e-14)
                np.testing.assert_array_equal(actual_counts, expected_counts)

    def test_single_and_pair_metropolis_kernels_preserve_physical_posterior(self):
        for method in ("parity", "refined", "cached", "scl", "joint", "xor"):
            size = 2 if method in ("joint", "xor") else 1
            work = Work(self.model)
            if method in ("parity", "refined", "cached"):
                center = self.residual + .4 if method == "cached" else self.residual
                proposal = ParityProposal(work, center, .7, method != "parity", 1, 3)
                logq = lambda words: proposal.logq(words[0])
            else:
                proposal = ListProposal(work, self.residual, .7, self.model.p, 4, .15,
                                        method if size == 2 else None)
                logq = proposal.logq
            states = list(itertools.product(range(1 << self.model.B), repeat=size))
            q = np.exp([logq(words) for words in states])
            self.assertAlmostEqual(q.sum(), 1., places=12)
            self.assertTrue(np.all(q > 0))
            likelihood = np.array([-np.square(self.residual - self.phi[:, words].sum(axis=1)).sum() / 1.4
                                   for words in states])
            p = np.exp(likelihood - logsumexp(likelihood))
            ratio = likelihood[None, :] - likelihood[:, None] + np.log(q[:, None]) - np.log(q[None, :])
            transition = q[None, :] * np.exp(np.minimum(0., ratio))
            transition[np.diag_indices(len(states))] += 1 - transition.sum(axis=1)
            np.testing.assert_allclose(p[:, None] * transition, p[None, :] * transition.T, atol=2e-14)
            np.testing.assert_allclose(p @ transition, p, atol=2e-14)
            if size == 2:
                different_patterns = [i for i, words in enumerate(states)
                                      if self.model.split(words[0])[0] != self.model.split(words[1])[0]]
                np.testing.assert_allclose(q[different_patterns], .15 / (1 << (2 * self.model.B)), atol=1e-16)

    def test_registered_receivers_match_procedural_and_legacy_readout(self):
        sparse_scenario = PolarScenario(self.model, self.residual, 2, .7)
        dense_scenario = Scenario(self.phi, self.residual, 2, .7)
        for number in range(25, 33):
            name = f"DEC-{number}"
            params = {"patterns": 2, "list_size": 4, "max_refine": 3, "seed": 741}
            params.update({"width": 4} if number < 27 else {"samples": 4, "burn_in": 0, "init_width": 2, "cache_centers": 1})
            counts, meta = run(name, sparse_scenario, **params)
            dense, adapted = run(name, dense_scenario, model=self.model, **params)
            self.assertEqual(counts, {int(w): int(dense[w]) for w in np.flatnonzero(dense)})
            self.assertEqual(meta["tail_probabilities"], adapted["tail_probabilities"])
            self.assertEqual(meta["work"], adapted["work"])
            self.assertFalse(meta["legacy_dense_adapter"])
            self.assertTrue(adapted["legacy_dense_adapter"])
            self.assertFalse(meta["full_alphabet_scan"])
            self.assertEqual(sum(counts.values()), 2)
            self.assertAlmostEqual(sum(sum(row) for _, row in meta["tail_probabilities"]), 2., places=12)
            if number in (30, 31):
                self.assertEqual(meta["work"]["pair_moves"], 2)
                self.assertEqual(meta["work"]["single_moves"], 2)


if __name__ == "__main__":
    unittest.main()
