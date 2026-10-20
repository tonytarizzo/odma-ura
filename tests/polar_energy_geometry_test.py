"""Small explicit certificates for the energy-balancing derivation in Report 8."""

import itertools
import unittest

import numpy as np
import torch

from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.polar_energy_geometry import balanced_energy, defect_gradient, diagnostics, moments
from framework.polar_mapping import PolarExplicitEncoder


class PolarEnergyGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(90)
        base = ODMAPolarBaseline(6, 64, 3217, prefix_bits=3, code_length=32, crc_bits=8, list_size=8)
        self.encoder = PolarExplicitEncoder(base, [2], "energy", True).double()
        with torch.no_grad(): self.encoder.codebook.values.normal_(0, .2)

    def test_exact_moments_defect_and_d2_interference(self):
        phi = self.encoder.explicit_matrix()
        c, mean = moments(self.encoder.codebook)
        m, n = phi.shape[1], phi.shape[0]
        torch.testing.assert_close(c, phi @ phi.T / m)
        torch.testing.assert_close(mean, phi.mean(1))
        gram = phi.T @ phi
        average = (gram.square().sum() - m) / m
        d = diagnostics(self.encoder.codebook)
        self.assertAlmostEqual(float(average.detach()), m / n * (1 + d["total_defect"]) - 1, places=10)
        self.assertAlmostEqual(d["total_defect"], float(n * (c - torch.eye(n) / n).square().sum().detach()), places=10)

    def test_closed_form_gradient(self):
        c, _ = moments(self.encoder.codebook)
        loss = (c - torch.eye(self.encoder.n) / self.encoder.n).square().sum()
        gradient = torch.autograd.grad(loss, self.encoder.codebook.values)[0]
        torch.testing.assert_close(defect_gradient(self.encoder.codebook), gradient, atol=1e-12, rtol=1e-10)

    def test_balance_and_structural_zeros(self):
        pos = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [0, 3]])
        e, info = balanced_energy(pos, 4)
        np.testing.assert_allclose(e.sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(np.bincount(pos.ravel(), weights=e.ravel()), 5 / 4, atol=1e-9)
        self.assertLess(info["max_relative_row_error"], 1e-10)
        # KKT: log(e/e0) is a pattern potential plus a resource potential.
        design = np.zeros((pos.size, len(pos) + 4))
        for p, row in enumerate(pos):
            for i, r in enumerate(row): design[p * 2 + i, p] = design[p * 2 + i, len(pos) + r] = 1
        rhs = np.log(e.ravel() * 2)
        fit = design @ np.linalg.lstsq(design, rhs, rcond=None)[0]
        np.testing.assert_allclose(fit, rhs, atol=1e-10)
        with self.assertRaises(ValueError): balanced_energy(np.array([[0, 1]]), 3)
        # Every row covered, but three patterns can only spend energy on resources 0/1.
        with self.assertRaises(ValueError):
            balanced_energy(np.array([[0, 1], [0, 1], [0, 1], [2, 3]]), 4, max_iterations=20)

    def test_known_K_covariance_identity(self):
        phi = self.encoder.explicit_matrix()[:, :8].detach()
        c, mean = phi @ phi.T / 8, phi.mean(1)
        sums = torch.stack([phi[:, pair].sum(1) for pair in itertools.combinations(range(8), 2)])
        centered = sums - sums.mean(0)
        torch.testing.assert_close(centered.T @ centered / len(sums), 2 * (8 - 2) / (8 - 1) * (c - mean[:, None] * mean))

    def test_tight_frames_need_not_have_same_recovery(self):
        duplicate = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]], dtype=torch.float64)
        unique = torch.cat((torch.eye(2, dtype=torch.float64),
                            torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64) / 2 ** .5), dim=1)
        torch.testing.assert_close(duplicate @ duplicate.T, unique @ unique.T)
        self.assertEqual(torch.unique(duplicate.T, dim=0).shape[0], 2)
        self.assertEqual(torch.unique(unique.T, dim=0).shape[0], 4)


if __name__ == "__main__": unittest.main()
