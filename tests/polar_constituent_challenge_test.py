import math
import unittest
from collections import Counter

import numpy as np

from baselines.polar import PolarCode
from framework.polar_constituent_challenge import _geometry, _select, finish
from framework.polar_full_pattern import CountBank
from framework.polar_tractable import PolarModel, PolarScenario


class ConstituentChallengeTest(unittest.TestCase):
    @staticmethod
    def model():
        code = PolarCode(87, 512, 16, 128)
        return PolarModel(code, np.arange(512)[None], np.full((1, 512), 512**-.5), 512)

    def test_original_words_have_their_predicted_global_sector(self):
        code, sequences, columns = _geometry()
        payloads = np.random.default_rng(1042).integers(2, size=(40, 87), dtype=np.uint8)
        wave = code.encode(payloads).reshape(-1, 8, 64)
        a = sequences[payloads[:, :11] @ (1 << np.arange(10, -1, -1))].astype(int)
        for j, groups in enumerate(columns):
            group = 2 * payloads[:, 11] + a[:, j]
            matches = np.all(groups[group] == wave[:, None, :, j], axis=2)
            self.assertTrue(matches.any(axis=1).all())

    def test_flat_restored_channel_counts_every_nonincumbent_alternative(self):
        m = self.model()
        w = 1
        s = PolarScenario(m, m.waveform(w), 2, .01)
        bank = CountBank(s, [w])
        fit = Counter({w:2})
        who, odds = _select(s, bank, fit, s.y - bank.matrix @ bank.vector(fit))
        self.assertEqual(who, w)
        self.assertAlmostEqual(odds, math.log(2 * ((1 << 87) - 1)), places=11)

    def test_saturated_physical_rollback_uses_the_new_complete_word(self):
        m = self.model()
        transmitted, incorrect = 1234, 151
        s = PolarScenario(m, m.waveform(transmitted), 1, .0001)
        old = dict(complete=True, bank=[incorrect], cancellation_action=[(incorrect, 1)])
        before = s.y.copy()
        counts, meta = finish(s, {incorrect:1}, old)
        self.assertEqual(counts, {transmitted:1})
        self.assertTrue(meta['constituent_challenge']['issued'])
        self.assertIn(transmitted, meta['constituent_challenge']['ids'])
        self.assertEqual(meta['cancellation_action'], [(transmitted, 1)])
        self.assertEqual(old['bank'], [incorrect])
        np.testing.assert_array_equal(before, s.y)


if __name__=='__main__':
    unittest.main()
