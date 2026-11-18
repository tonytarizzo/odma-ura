"""Exhaustive original-code checks on the common coarsening used to rank the last fitted slot."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode
from framework.polar_folded_terminal import folded_log_bf, signed_prefix_end, terminal_query
from framework.polar_tractable import PolarModel, PolarScenario, bits


def fixture(b):
    code = PolarCode(b, 16, 8)
    rows = np.arange(16) + np.array([[0], [0], [2], [11]])
    amp = np.sqrt(1. + np.arange(64).reshape(4, 16) % 9)
    amp /= np.linalg.norm(amp, axis=1)[:, None]
    return PolarModel(code, rows, amp, 27)


def waveform_bank(model):
    messages = np.array([bits(i, model.b) for i in range(1 << model.b)])
    signs = 1. - 2. * model.code.encode(messages)
    result = np.zeros((model.p, len(messages), model.n))
    for p in range(model.p):
        result[p][:, model.positions[p]] = signs * model.amplitudes[p]
    return result


def absolute_log_density(waveforms, raw, end, variance):
    # Fold by adding the two actual Gaussian image densities, including folded noise-only resources.
    base = -.5 * np.log(2 * np.pi * variance)
    values = base - (raw - waveforms)**2 / (2 * variance)
    values[..., end+1:] = np.logaddexp(values[..., end+1:], base - (-raw[end+1:] - waveforms[..., end+1:])**2 / (2 * variance))
    return values.sum(axis=-1)


class FoldedTerminalTest(unittest.TestCase):
    def test_all_payloads_and_normalization(self):
        for b in (7, 8):
            model = fixture(b)
            waves = waveform_bank(model)
            self.assertEqual(signed_prefix_end(model), b)
            for raw in (np.zeros(model.n), np.sin(np.arange(model.n)), 40.*np.cos(np.arange(model.n))):
                for end in (-1, 0, b):
                    for variance in (.07, .4):
                        bf, _ = folded_log_bf(model, raw, variance, end)
                        density = absolute_log_density(waves, raw, end, variance)
                        noise = absolute_log_density(np.zeros(model.n), raw, end, variance)
                        by_pattern = logsumexp(density, axis=1) - b*np.log(2.)
                        np.testing.assert_allclose(noise+bf, by_pattern, atol=5e-10, rtol=0)
                        np.testing.assert_allclose(bf-logsumexp(bf), by_pattern-logsumexp(by_pattern), atol=5e-10, rtol=0)
                        self.assertAlmostEqual(logsumexp(bf)+noise-np.log(model.p), logsumexp(density)-np.log(density.size), places=8)
                        for p in range(model.p):
                            rank = np.count_nonzero(model.positions[p, 1:] <= end)
                            self.assertLessEqual(np.exp(density[p].max()-logsumexp(density[p])), 2.**(int(rank)-b)+1e-8)
                        flipped = raw.copy()
                        flipped[end+1:] *= -1
                        np.testing.assert_array_equal(folded_log_bf(model, flipped, variance, end)[0], bf)

    def test_signed_boundary_is_physical(self):
        m = fixture(7)
        raw = np.zeros(m.n)
        raw[0] = -1.
        self.assertFalse(np.allclose(folded_log_bf(m, raw, .2)[0], folded_log_bf(m, -raw, .2)[0]))
        with self.assertRaises(ValueError):
            folded_log_bf(m, raw, .2, 8)
        raw = 1. - 2.*(np.arange(m.n) % 2)
        waves = waveform_bank(m)
        actual = logsumexp(absolute_log_density(waves[0], raw, 8, .05)) - m.b*np.log(2)
        noise = absolute_log_density(np.zeros(m.n), raw, 8, .05)
        naive, _ = folded_log_bf(m, raw, .05, 7)
        self.assertGreater(abs(actual-noise-naive[0]), .01)

    def test_two_word_folded_xor(self):
        m = fixture(7)
        waves = waveform_bank(m)
        z = .35*np.cos(np.arange(m.n))
        for p,q in [(0, 0), (0, 1), (0, 3)]:
            positions, i, j = np.intersect1d(m.positions[p], m.positions[q], return_indices=True)
            # Check every ordered pair including identical messages in bounded slabs.
            private = np.setdiff1d(np.arange(m.n), positions)
            for x in range(0, 128, 8):
                a, b = waves[p, x:x+8, None, :], waves[q, None, :, :]
                actual = absolute_log_density(a+b, z, -1, .2)
                same = (a[..., positions]*b[..., positions]) > 0
                magnitude = np.where(same, m.amplitudes[p,i]+m.amplitudes[q,j], abs(m.amplitudes[p,i]-m.amplitudes[q,j]))
                expected = absolute_log_density(magnitude, z[positions], -1, .2)
                private_amplitude = abs(waves[p,0,private]) + abs(waves[q,0,private])
                expected += absolute_log_density(private_amplitude, z[private], -1, .2)
                np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=0)
                if p == q:
                    all_xor = np.arange(x, x+8)[:, None] ^ np.arange(128)
                    np.testing.assert_array_equal(~same, waves[p][all_xor][..., positions] < 0)

    def test_query_association_control_and_repeated_output(self):
        m = fixture(7)
        # Count 1 is fitted and a duplicate of this same complete message is physically present.
        p, v = m.symbols(0)
        y = np.zeros(m.n)
        y[m.positions[p]] = 2*v
        def list_for_fixture(code, llr):
            return np.array([bits(0, m.b), bits(1, m.b)]), np.array([0., 2.])
        with patch.object(PolarCode, 'decode_list', list_for_fixture):
            for mode in ('folded', 'magnitude'):
                action, meta = terminal_query(PolarScenario(m, y, 2, .001), Counter({0:1}), [0], mode)
                self.assertEqual(meta['pattern'], 0)
                self.assertEqual(action, {0:2})
                self.assertEqual(meta['new_ids'], [1])
                self.assertTrue(meta['fit']['converged'])
            y = np.zeros(m.n)
            y[0] = -10.
            scenario = PolarScenario(m, y, 1, .2)
            _, a = terminal_query(scenario, Counter(), [], 'magnitude')
            _, b = terminal_query(scenario, Counter(), [], 'folded')
            self.assertNotEqual(a['pattern'], b['pattern'])
            with self.assertRaises(ValueError):
                terminal_query(scenario, Counter({0:1}), [0], 'folded')


if __name__ == '__main__':
    unittest.main()
