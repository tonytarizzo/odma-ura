"""Bounded full-message execution checks; B=100 success here is not a performance benchmark."""

import json
import unittest

import numpy as np

from baselines.polar import PolarCode
from framework.polar_decoders import run
from framework.polar_tractable import PolarModel, PolarScenario


def model100():
    code = PolarCode(98, 128, crc_bits=16, list_size=4)
    positions = (np.arange(128)[None, :] + np.arange(4)[:, None] * 64) % 256
    return PolarModel(code, positions, np.full((4, 128), 1 / np.sqrt(128)), 256)


def budgets(number):
    common = {"patterns": 2, "list_size": 4, "max_refine": 4, "seed": 71}
    return {**common, **({"width": 2} if number < 27 else {"samples": 2, "burn_in": 0, "init_width": 1})}


class ProceduralExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model100()

    def test_b100_single_user_high_snr(self):
        model = self.model
        word = (1 << 99) + (1 << 80) + 123456789
        scenario = PolarScenario(model, model.waveform(word), 1, 1e-4)
        self.assertFalse(hasattr(scenario, "phi"))
        for number in range(25, 33):
            with self.subTest(decoder=number):
                counts, meta = run(f"DEC-{number}", scenario, **budgets(number))
                self.assertEqual(counts, {word: 1})
                self.assertFalse(meta["legacy_dense_adapter"])
                self.assertFalse(meta["full_alphabet_scan"])
                self.assertLess(len(meta["candidate_ids"]), 100)
                self.assertEqual(json.loads(json.dumps(list(counts.items())))[0][0], word)
                json.dumps(meta, allow_nan=False)

    def test_b100_two_user_execution_is_reproducible(self):
        model = self.model
        words = [(1 << 99) + 987654321, (1 << 98) + (1 << 76) + 123456789]
        y = sum((model.waveform(w) for w in words), np.zeros(model.n))
        scenario = PolarScenario(model, y, 2, .05)
        for number in range(25, 33):
            with self.subTest(decoder=number):
                counts, meta = run(f"DEC-{number}", scenario, **budgets(number))
                repeat = run(f"DEC-{number}", scenario, **budgets(number))
                self.assertEqual((counts, meta), repeat)
                self.assertEqual(sum(counts.values()), 2)
                self.assertTrue(all(0 <= w < 1 << 100 for w in counts))
                self.assertLess(len(meta["candidate_ids"]), 200)
                self.assertLessEqual(sum(len(t) for _, t in meta["tail_probabilities"]), 4)
                if number in (30, 31):
                    self.assertEqual(meta["work"]["pair_moves"], 1)

    def test_reject_explicit_family_without_alphabet(self):
        scenario = PolarScenario(self.model, np.zeros(256), 1, 1.)
        with self.assertRaisesRegex(ValueError, "explicit"):
            run("DEC-14", scenario)


if __name__ == "__main__":
    unittest.main()
