"""Independent equation and interpretation checks for the job-032 reference."""

import math
import unittest

import numpy as np
from scipy.stats import laplace, norm

from benchmarks.ura_bounds import (collision_rates, minimum_information_cdf, polyanskiy_achievability,
                                   polyanskiy_gallager, q1_bound, reference_curves, required_ebn0)
from src.ura_bound import _E_of_t


def scalar_exponent(power, t, n, b, k, grid):
    """Literal scalar transcription of Polyanskiy (2017), equations (5)--(10)."""
    r1 = (b * math.log(2) - math.lgamma(t + 1) / t) / n
    r2 = math.log(math.comb(k, t)) / n
    values = [0.0]  # rho=0 is a valid boundary point.
    for rho in np.linspace(1e-4, 1, grid):
        for rho1 in np.linspace(1e-4, 1, grid):
            pt = power * t
            d = (pt - 1) ** 2 + 4 * pt * (1 + rho * rho1) / (1 + rho)
            lam = (pt - 1 + math.sqrt(d)) / (4 * (1 + rho1 * rho) * pt)
            mu = rho * lam / (1 + 2 * pt * lam)
            a = rho / 2 * math.log(1 + 2 * pt * lam) + .5 * math.log(1 + 2 * pt * mu)
            beta = rho * lam - mu / (1 + 2 * pt * mu)
            domain = 1 - 2 * beta * rho1
            if domain > 0:
                values.append(rho1 * a + .5 * math.log(domain) - rho * rho1 * t * r1 - rho1 * r2)
    return max(values)


class PublishedBoundsTests(unittest.TestCase):
    def test_single_user_information_cdf_has_exact_laplace_case(self):
        # At n=2, diagonalizing the Gaussian information-density quadratic form
        # gives log(1+P') + sqrt(P'/(1+P')) * Laplace(0,1).
        p = .7
        center, scale = math.log1p(p), math.sqrt(p / (1 + p))
        gamma = center + scale * np.array([-2, -1, 0, 1, 2])
        actual = minimum_information_cdf(gamma, p, 2, 1, order=384)
        np.testing.assert_allclose(actual, laplace.cdf(gamma, loc=center, scale=scale), atol=3e-5)

    def test_user_minimum_matches_shared_noise_monte_carlo(self):
        rng = np.random.default_rng(32701)
        n, p, k, samples = 16, .6, 5, 30000
        center = n / 2 * math.log1p(p)
        gamma = center - math.sqrt(n * p / (1 + p))
        hits = 0
        for _ in range(samples // 1000):
            z = rng.standard_normal((1000, 1, n))
            x = math.sqrt(p) * rng.standard_normal((1000, k, n))
            information = center + .5 * (((x + z) ** 2).sum(-1) / (1 + p) - (z ** 2).sum(-1))
            hits += int((information.min(1) <= gamma).sum())
        expected = float(minimum_information_cdf(gamma, p, n, k, order=192))
        self.assertLess(abs(hits / samples - expected), 5 * math.sqrt(expected * (1 - expected) / samples))
        single = float(minimum_information_cdf(gamma, p, n, 1, order=192))
        self.assertGreater(abs(expected - (1 - (1 - single) ** k)), .01)

    def test_quadrature_refinement_and_q1_threshold_optimization(self):
        for n, p, k in [(64, .2, 3), (30000, .008, 50)]:
            gamma = n / 2 * math.log1p(p) - 2 * math.sqrt(n * p / (1 + p))
            coarse = minimum_information_cdf(gamma, p, n, k, order=96)
            fine = minimum_information_cdf(gamma, p, n, k, order=192)
            self.assertLess(abs(coarse - fine), 4e-6)
        result = q1_bound(6, 64, 1, .2)
        gamma = np.linspace(6 * math.log(2), 20, 501)
        values = minimum_information_cdf(gamma, .2, 64, 1) + np.exp(6 * math.log(2) - gamma)
        self.assertLessEqual(result["value"], values.min() + 1e-7)

    def test_q1_reference_never_worsens_gallager_control(self):
        for b, n, k, snr in [(6, 64, 1, 0), (6, 64, 3, 4), (14, 256, 7, 3), (100, 30000, 50, .5)]:
            row = polyanskiy_achievability(b, n, k, snr, details=True)
            self.assertLessEqual(row["upper_bound"], row["gallager_only_upper_bound"] + 1e-12)
            self.assertAlmostEqual(row["unclipped_upper_bound"],
                                   row["coding_term"] + row["clipping_term"] + row["collision_union_term"])
        self.assertLess(polyanskiy_achievability(6, 64, 1, 0), .95)

    def test_required_energy_reports_collision_limitation_without_fabricating_crossing(self):
        result = required_ebn0(12, 256, 26, target=.05)
        self.assertEqual(result["status"], "collision_correction_exceeds_target")
        self.assertIsNone(result["ebn0_db"])
        result = required_ebn0(6, 64, 1, target=.1, grid=11, power_grid=11)
        self.assertEqual(result["status"], "crossing")
        self.assertAlmostEqual(polyanskiy_achievability(6, 64, 1, result["ebn0_db"], grid=11, power_grid=11), .1, delta=.001)

    def test_gallager_exponent_matches_independent_scalar_equations(self):
        for b, n, k, power in [(6, 64, 3, .25), (14, 256, 7, .15), (100, 30000, 50, .01)]:
            ts = np.array([1, max(1, k // 2), k])
            r1 = np.array([(b * math.log(2) - math.lgamma(t + 1) / t) / n for t in ts])
            r2 = np.array([math.log(math.comb(k, int(t))) / n for t in ts])
            expected = [scalar_exponent(power, int(t), n, b, k, 11) for t in ts]
            np.testing.assert_allclose(np.maximum(0, _E_of_t(power, ts, n, r1, r2, 11)), expected, atol=2e-15)

    def test_components_reconstruct_bound_and_sampling_correction(self):
        for distinct in (False, True):
            result = polyanskiy_gallager(6, 64, 3, 4, distinct=distinct, details=True)
            total = result["coding_term"] + result["clipping_term"] + result["collision_union_term"]
            self.assertAlmostEqual(result["unclipped_upper_bound"], total)
            self.assertEqual(result["upper_bound"], min(1, total))
            self.assertEqual(result["collision_union_term"], 0 if distinct else 3 / 64)
        iid = polyanskiy_gallager(6, 64, 3, 4)
        distinct = polyanskiy_gallager(6, 64, 3, 4, distinct=True)
        self.assertAlmostEqual(iid - distinct, 3 / 64)

    def test_upper_bound_need_not_be_a_performance_floor(self):
        # B=1,K=1: antipodal unit-energy signalling is exactly optimal and has
        # PUPE Q(sqrt(2 Eb/N0)). A valid achievability upper bound can be worse.
        for n in (8, 64, 256):
            for snr in (-4, 0, 4):
                optimum = norm.sf(math.sqrt(2 * 10 ** (snr / 10)))
                upper = polyanskiy_gallager(1, n, 1, snr, grid=11, power_grid=11)
                self.assertGreaterEqual(upper, optimum)
        self.assertGreater(polyanskiy_gallager(1, 64, 1, 0), 3 * norm.sf(math.sqrt(2)))

    def test_grid_refinement_cannot_worsen_upper_bound(self):
        # 21 grid points contain the 11-point grid, including the same endpoints.
        coarse = polyanskiy_gallager(12, 256, 7, 3, grid=11, power_grid=11)
        fine = polyanskiy_gallager(12, 256, 7, 3, grid=21, power_grid=21)
        self.assertLessEqual(fine, coarse + 1e-12)

    def test_exact_collision_probabilities_against_enumeration(self):
        from itertools import product
        messages = list(product(range(4), repeat=3))
        per_user = np.mean([sum(row.count(x) > 1 for x in row) / 3 for row in messages])
        any_frame = np.mean([len(set(row)) < 3 for row in messages])
        rates = collision_rates(2, 3)
        self.assertAlmostEqual(rates["per_user_duplicate_probability"], per_user)
        self.assertAlmostEqual(rates["any_duplicate_probability"], any_frame)
        self.assertGreaterEqual(rates["any_duplicate_union_bound"], any_frame)

    def test_reference_metadata_and_finite_snr(self):
        curve = reference_curves(6, 64, 2, iter([0, 4]), distinct=True)
        self.assertEqual(len(curve["achievability_components"]), 2)
        self.assertIn("upper bound", curve["achievability_label"])
        self.assertIn("below", curve["achievability_direction"])
        for snr in (math.inf, math.nan):
            with self.assertRaises(ValueError): polyanskiy_gallager(6, 64, 2, snr)


if __name__ == "__main__": unittest.main()
