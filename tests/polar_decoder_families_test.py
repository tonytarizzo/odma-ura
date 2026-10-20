"""Independent small-alphabet limits and numerical identities for all Report-10 families."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import gammaln, logsumexp

from framework.polar_decoders import REGISTRY, Scenario, run
from framework.polar_decoder_queries import geometry
from framework.polar_decoder_search import mixture_log_density


class DecoderFamiliesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(892)
        cls.phi = rng.normal(size=(3, 4))
        cls.phi /= np.linalg.norm(cls.phi, axis=0)
        cls.y = np.array([.2, -.5, .8])

    def scenario(self, k=3, source="iid", variance=.7):
        return Scenario(self.phi, self.y, k, variance, source)

    def exact(self, s):
        return run("REF", s)

    def test_every_catalogue_entry_is_callable_and_legal(self):
        for name in (f"DEC-{i:02}" for i in range(1, 14)):
            self.assertIsNotNone(REGISTRY[name].run)
            for source in REGISTRY[name].sources:
                s = self.scenario(source=source)
                counts, meta = run(name, s)
                self.assertEqual(counts.sum(), s.k, (name, source))
                self.assertTrue(np.isfinite(meta["excess_risk_bound"]))
                self.assertTrue(0 <= meta["excess_risk_bound"] <= 1)
                if source == "distinct":
                    self.assertLessEqual(counts.max(), 1)

    def test_single_user_deterministic_limits(self):
        s = self.scenario(k=1)
        counts, exact = self.exact(s)
        for i in (1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13):
            got, meta = run(f"DEC-{i:02}", s)
            np.testing.assert_array_equal(got, counts)
            if "tails" in meta:
                np.testing.assert_allclose(meta["tails"], exact["tails"], atol=2e-12)

    def test_excluded_gaussian_matches_direct_covariance_formula_both_sources(self):
        for source in ("iid", "distinct"):
            s = self.scenario(source=source)
            _, meta = run("DEC-01", s)
            expected = []
            for w in range(4):
                other = np.delete(self.phi, w, axis=1)
                mean = other.mean(1)
                centered = other - mean[:, None]
                covariance = centered @ centered.T / 3
                masses = np.full(4, -np.inf)
                for j in range(4) if source == "iid" else (0, 1):
                    h = 3 - j
                    scale = h if source == "iid" else h * (3 - h) / 2
                    matrix = .7 * np.eye(3) + scale * covariance
                    residual = self.y - j * self.phi[:, w] - h * mean
                    prior = (gammaln(4) - gammaln(j + 1) - gammaln(h + 1) + j * math.log(.25) + h * math.log(.75)
                             if source == "iid" else math.log(.75 if j else .25))
                    masses[j] = prior - .5 * (np.linalg.slogdet(matrix / .7)[1] + residual @ np.linalg.solve(matrix, residual))
                probability = np.exp(masses - logsumexp(masses))
                expected.append(np.cumsum(probability[:0:-1])[::-1])
            np.testing.assert_allclose(meta["tails"], expected, atol=1e-12)
        full = Scenario(self.phi, self.y, 4, .7, "distinct")
        np.testing.assert_array_equal(run("DEC-01", full)[0], np.ones(4))

    def test_moment_cache_is_encoder_specific(self):
        s = self.scenario()
        for name in ("DEC-01", "DEC-13"):
            with self.assertRaisesRegex(ValueError, "another encoder"):
                run(name, s, geometry_cache=geometry(self.phi.copy()))

    def test_prefix_refinement_closes_the_product_decision(self):
        s = self.scenario()
        product, _ = run("DEC-02", s)
        prefix, meta = run("DEC-03", s, splits=32)
        np.testing.assert_array_equal(prefix, product)
        self.assertLess(meta["product_decision_gap"], 1e-10)
        self.assertLessEqual(meta["visited_prefixes"], 2 * self.phi.shape[1] - 1)

    def test_configuration_exhaustion_recovers_exact_both_sources(self):
        for source in ("iid", "distinct"):
            s = self.scenario(source=source)
            _, exact = self.exact(s)
            _, full = run("DEC-04", s, width=64)
            np.testing.assert_allclose(full["tails"], exact["tails"], atol=1e-12)
            self.assertEqual(full["omitted_mass_bound"], 0)
            counts, small = run("DEC-04", s, width=2)
            risk = 1 - sum(exact["tails"][w, :c].sum() for w, c in enumerate(counts)) / s.k
            self.assertLessEqual(risk - exact["posterior_risk"], small["excess_risk_bound"] + 1e-12)

    def test_global_chain_proposal_and_metropolis_detailed_balance(self):
        logp = np.log([.1, .2, .3, .4])
        for distinct in (False, True):
            tuples = np.array(list(itertools.permutations(range(4), 2) if distinct else itertools.product(range(4), repeat=2)))
            q = np.exp(mixture_log_density(logp, tuples, distinct, .2))
            self.assertAlmostEqual(q.sum(), 1., places=13)
            score = -np.square(self.y - self.phi[:, tuples].sum(2).T).sum(1) / 1.4
            p = np.exp(score - logsumexp(score))
            flow = np.minimum(p[:, None] * q[None], p[None] * q[:, None])
            transition = flow / p[:, None]
            transition[np.diag_indices(len(p))] += 1 - transition.sum(1)
            np.testing.assert_allclose(p @ transition, p, atol=1e-14)

    def test_chain_frequencies_and_all_block_sizes(self):
        s = self.scenario(k=2, variance=2.)
        _, exact = self.exact(s)
        _, meta = run("DEC-06", s, samples=1500, burn_in=50, chains=4, global_probability=.5, seed=82)
        self.assertLess(np.max(np.abs(meta["tails"] - exact["tails"])), .04)
        self.assertEqual(meta["recorded_states"], 6000)
        self.assertEqual(meta["excess_risk_bound"], 1)
        for size in (2, 3):
            s = self.scenario(source="distinct")
            counts, meta = run("DEC-06", s, samples=30, burn_in=10, block_size=size)
            self.assertEqual(counts.sum(), 3)
            self.assertEqual(meta["block_size"], size)

    def test_particle_batch_evidence_and_marks_match_enumeration(self):
        for source in ("iid", "distinct"):
            s = self.scenario(k=2, source=source, variance=2.)
            _, exact = self.exact(s)
            _, meta = run("DEC-07", s, batches=256, particles=8, seed=491)
            self.assertLess(abs(math.exp(meta["log_mean_weight"]) - math.exp(exact["log_evidence"])), .025)
            self.assertLess(np.max(np.abs(meta["tails"] - exact["tails"])), .04)
            self.assertEqual(meta["independent_confidence_units"], 256)

    def test_quantized_zero_error_limit_and_forced_merge_certificate(self):
        for source in ("iid", "distinct"):
            s = self.scenario(source=source)
            _, exact = self.exact(s)
            _, full = run("DEC-08", s, candidates=4, representatives=1000, spacing=0.)
            np.testing.assert_allclose(full["tails"], exact["tails"], atol=1e-11)
            counts, merged = run("DEC-08", s, candidates=4, representatives=1, spacing=.5)
            self.assertAlmostEqual(merged["tails"].sum(), s.k, places=10)
            risk = 1 - sum(exact["tails"][w, :c].sum() for w, c in enumerate(counts)) / s.k
            self.assertLessEqual(risk - exact["posterior_risk"], merged["excess_risk_bound"] + 1e-11)

    def test_regions_full_region_and_orthogonal_singletons_are_exact(self):
        for source in ("iid", "distinct"):
            s = self.scenario(source=source)
            _, exact = self.exact(s)
            _, full = run("DEC-09", s, region_size=4)
            np.testing.assert_allclose(full["tails"], exact["tails"], atol=1e-11)
            self.assertEqual(full["excess_risk_bound"], 0.)
            orthogonal = Scenario(np.eye(4), np.arange(4) / 4, 3, .5, source)
            _, exact = self.exact(orthogonal)
            _, independent = run("DEC-09", orthogonal)
            np.testing.assert_allclose(independent["tails"], exact["tails"], atol=1e-11)

    def test_connected_full_pair_and_triangle_match_exact_posterior(self):
        for k in (2, 3):
            s = self.scenario(k=k)
            _, exact = self.exact(s)
            _, full = run("DEC-10", s, candidates=4, order=k)
            np.testing.assert_allclose(full["tails"], exact["tails"], atol=1e-11)
            self.assertEqual(full["excess_risk_bound"], 0.)

    def test_fourier_mark_identity_intervals_and_signed_failure_is_explicit(self):
        s = self.scenario(k=2, variance=2.)
        _, exact = self.exact(s)
        _, meta = run("DEC-11", s, frequencies=50_000, seed=492)
        self.assertLess(meta["max_word_evidence_discrepancy"], 1e-12)
        self.assertTrue(np.all(meta["tails"] <= exact["tails"] + 1e-12))
        self.assertTrue(np.all(meta["tail_upper"] >= exact["tails"] - 1e-12))
        self.assertTrue(meta["positive_evidence_certified"])
        _, weak = run("DEC-11", self.scenario(variance=.01), frequencies=2)
        self.assertFalse(weak["positive_evidence_certified"])
        self.assertEqual(weak["excess_risk_bound"], 1.)

    def test_signed_cut_full_rank_limit_and_actual_residual_envelope(self):
        for source in ("iid", "distinct"):
            s = self.scenario(source=source)
            _, exact = self.exact(s)
            _, full = run("DEC-12", s, candidates=4, width=64, rank=64)
            np.testing.assert_allclose(full["tails"], exact["tails"], atol=1e-11)
            self.assertLess(full["excess_risk_bound"], 1e-10)
            counts, low = run("DEC-12", s, candidates=4, width=64, rank=1)
            risk = 1 - sum(exact["tails"][w, :c].sum() for w, c in enumerate(counts)) / s.k
            self.assertLessEqual(risk - exact["posterior_risk"], low["excess_risk_bound"] + 1e-11)
            self.assertGreater(low["relative_envelope"], 0.)

    def test_joint_fields_correct_both_zero_and_retained_rank(self):
        s = self.scenario(k=2, variance=2.)
        _, exact = self.exact(s)
        for rank in (0, 2):
            _, meta = run("DEC-13", s, samples=12_000, field_rank=rank, seed=87)
            self.assertLess(np.max(np.abs(meta["tails"] - exact["tails"])), .04)
            self.assertLess(abs(math.exp(meta["log_mean_weight"]) - math.exp(exact["log_evidence"])), .03)
            self.assertEqual(meta["field_dimension"], rank)


if __name__ == "__main__":
    unittest.main()
