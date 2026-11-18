"""Private native substitution and complete repeated-message progress-wrapper contract."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_feedback import run as original
from framework.polar_native_repetition import run as native_repetition
from framework.polar_progress_continuation import run
from framework.polar_repetition_scl import RepetitionPolarCode
from tests.polar_count_cavity_test import smoke_scenario


class ProgressReceiverTest(unittest.TestCase):
    def test_private_native_only_substitution_forwards_fixed_progress_policy(self):
        scenario, _ = smoke_scenario()
        model, code = scenario.model, scenario.model.code
        arrays = {key: getattr(model, key).copy() for key in ('positions', 'amplitudes', 'mean')}

        def inspect(copied, *, memoize, complete_support, progress_extra_sweep):
            self.assertIsNot(copied, scenario)
            self.assertIsNot(copied.model, model)
            self.assertIs(type(copied.model.code), RepetitionPolarCode)
            self.assertIs(copied.y, scenario.y)
            self.assertEqual((copied.k, copied.noise_var, copied.sampling), (scenario.k, scenario.noise_var, scenario.sampling))
            self.assertTrue(memoize and complete_support and progress_extra_sweep)
            for key in arrays:
                self.assertIs(getattr(copied.model, key), getattr(model, key))
            basis = np.eye(code.payload_bits, dtype=np.uint8)
            np.testing.assert_array_equal(copied.model.code.encode(basis), code.encode(basis))
            copied.model.code.list_size = 1
            return {209: 2}, {'total_seconds': 0.}

        with patch('framework.polar_progress_continuation.count_feedback', side_effect=inspect) as call:
            counts, meta = run(scenario)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(counts, {209: 2})
        self.assertIs(scenario.model, model)
        self.assertIs(scenario.model.code, code)
        self.assertEqual(code.list_size, 128)
        for key, value in arrays.items():
            np.testing.assert_array_equal(getattr(model, key), value)
        self.assertGreaterEqual(meta['total_seconds'], meta['native_repetition_preparation_seconds'])

    def test_complete_repeated_message_case_matches_both_three_block_references(self):
        scenario, truth = smoke_scenario()
        before = scenario.y.copy()
        reference_counts, reference = original(scenario, memoize=True, complete_support=True)
        native_counts, native = native_repetition(scenario)
        counts, meta = run(scenario)
        self.assertEqual(Counter(counts), truth)
        self.assertEqual(counts, reference_counts)
        self.assertEqual(counts, native_counts)
        for other in (reference, native):
            for key in ('rollback_action', 'rollback_batches', 'base_bank', 'bank', 'cancellation_action', 'complete'):
                self.assertEqual(meta[key], other[key])
            for key in ('calls', 'rounds', 'valid_ids', 'residual_sha256'):
                self.assertEqual(meta['native'][key], other['native'][key])
            for left, right in zip(meta['continuation']['queries'], other['continuation']['queries'], strict=True):
                for key in ('pattern', 'pre_action_sha256', 'llr_sha256', 'ids', 'new_ids', 'refit_index', 'cache_hit'):
                    self.assertEqual(left[key], right[key])
        self.assertTrue(meta['continuation']['progress_extra_sweep'])
        self.assertFalse(meta['continuation']['extra_sweep_triggered'])
        self.assertEqual(meta['support_completion']['additions'], [])
        np.testing.assert_array_equal(scenario.y, before)

    def test_requires_original_encoder_and_native_budget(self):
        scenario, _ = smoke_scenario()
        scenario.model.code.list_size = 512
        with self.assertRaisesRegex(ValueError, 'native SCL128'):
            run(scenario)
        old = scenario.model.code
        scenario.model.code = RepetitionPolarCode(old.payload_bits, old.n, old.crc_bits, 128)
        with self.assertRaisesRegex(ValueError, 'original PolarCode'):
            run(scenario)


if __name__ == '__main__':
    unittest.main()
