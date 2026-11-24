"""Checks for repeated complete IDs and the balanced fixed-code load experiment."""
from collections import Counter
from importlib import import_module
import json
import unittest
from unittest.mock import patch

import numpy as np

receiver = import_module('jobs.043_polar_balanced_load.receiver')
from framework.polar_tractable import bits


class BalancedLoadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.cost = receiver.geometry(1043091)

    def test_balanced_energy_and_paired_noise(self):
        m = self.model
        energy = m.amplitudes ** 2
        np.testing.assert_allclose(energy.sum(1), 1., rtol=0, atol=1e-10)
        chip_sum = np.bincount(m.positions.ravel(), weights=energy.ravel(), minlength=m.n)
        np.testing.assert_allclose(chip_sum, m.p/m.n, rtol=1e-10, atol=1e-14)
        for k in (25, 50, 100, 300):
            lo, labels, src = receiver.paper_curve.make_frame(m, k, 1043091, -50)
            hi, other, second = receiver.paper_curve.make_frame(m, k, 1043091, 100)
            self.assertEqual(labels, other)
            for field in receiver.PHYSICAL_HASHES[:-1]:
                self.assertEqual(src[field], second[field])
            signal = sum((m.waveform(w) for w in labels), np.zeros(m.n))
            np.testing.assert_allclose((lo.y-signal)/np.sqrt(lo.noise_var), (hi.y-signal)/np.sqrt(hi.noise_var), rtol=0, atol=1e-13)

    def test_coincident_ids_are_charged_as_copies(self):
        m = self.model
        a, b = (318 << 87) + 11, (318 << 87) + 97
        c = (1020 << 87) + 125
        words = [a, a, b, c]
        returned = Counter({a:1, b:2, c:1})
        truth = np.stack([bits(w, 100) for w in words])
        rng = np.random.default_rng(100)
        noise = rng.standard_normal(m.n)
        with patch.object(receiver.paper_curve.np.random, 'default_rng') as factory:
            factory.return_value.integers.return_value = truth
            factory.return_value.standard_normal.return_value = noise
            sc, labels, src = receiver.paper_curve.make_frame(m, 4, 1043091, 500)
        self.assertEqual(labels, words)
        self.assertEqual((src['repeated_messages'], src['repeated_patterns']), (1, 2))
        np.testing.assert_allclose(sc.y, np.sqrt(sc.noise_var)*noise + 2*m.waveform(a) + m.waveform(b) + m.waveform(c), atol=1e-15)
        measured = receiver.paper_curve.score(labels, returned)
        self.assertEqual((measured['list_misses'], measured['count_misses'], measured['false_copies']), (0, 1, 1))
        for name in receiver.ARMS:
            action, meta = receiver.run(name, sc)
            self.assertTrue(meta['complete'])
            estimate = Counter(action)
            self.assertEqual(estimate, Counter(labels))
            vals = receiver.paper_curve.score(labels, action)
            self.assertEqual(vals['count_misses'], (Counter(labels)-estimate).total())
            self.assertEqual(vals['false_copies'], (estimate-Counter(labels)).total())

    def test_saved_complete_actions_and_source_integrity(self):
        row = receiver.case(self.model, self.cost, 25, 1043091, 100)
        restored = json.loads(json.dumps(row))
        receiver.validate(restored, 25, 1043091, 100)
        for name in receiver.ARMS:
            truth, estimate = Counter(restored['labels']), Counter(dict(restored['arms'][name]['action']))
            self.assertEqual(restored['arms'][name]['metrics']['count_misses'], (truth-estimate).total())
            self.assertEqual(restored['arms'][name]['metrics']['false_copies'], (estimate-truth).total())
        restored['source']['standard_noise_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'noise'):
            receiver.validate(restored, 25, 1043091, 100)


if __name__ == '__main__':
    unittest.main()
