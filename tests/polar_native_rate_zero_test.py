"""Complete finite native-only wrapper checks, with repeated messages and immutable caller geometry."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_feedback import run as reference
from framework.polar_fast_scl import RateZeroPolarCode
from framework.polar_native_rate_zero import run
from tests.polar_count_cavity_test import smoke_scenario


class NativeRateZeroTest(unittest.TestCase):
    def test_shallow_copy_preserves_original_encoding_arrays_and_configuration(self):
        scenario, _ = smoke_scenario()
        original, code = scenario.model, scenario.model.code
        arrays = {key: getattr(original, key).copy() for key in ("positions", "amplitudes", "mean")}
        observed = []

        def inspect(copied, *, memoize, complete_support):
            self.assertIsNot(copied, scenario)
            self.assertIsNot(copied.model, original)
            self.assertIs(type(copied.model.code), RateZeroPolarCode)
            self.assertIs(copied.y, scenario.y)
            for key in arrays:
                self.assertIs(getattr(copied.model, key), getattr(original, key))
            self.assertEqual((copied.k, copied.noise_var, copied.sampling),
                             (scenario.k, scenario.noise_var, scenario.sampling))
            basis = np.eye(code.payload_bits, dtype=np.uint8)
            np.testing.assert_array_equal(copied.model.code.encode(basis), code.encode(basis))
            np.testing.assert_array_equal(copied.model.code.information, code.information)
            self.assertTrue(memoize and complete_support)
            copied.model.code.list_size = 1
            observed.append(copied)
            return {209: 2}, {"total_seconds": 0.}

        with patch("framework.polar_native_rate_zero.count_feedback", side_effect=inspect):
            counts, meta = run(scenario)
        self.assertEqual(counts, {209: 2})
        self.assertEqual(len(observed), 1)
        self.assertIs(scenario.model, original)
        self.assertIs(scenario.model.code, code)
        self.assertIs(type(code), PolarCode)
        self.assertEqual(code.list_size, 128)
        for key, value in arrays.items():
            np.testing.assert_array_equal(getattr(original, key), value)
        self.assertGreaterEqual(meta["total_seconds"], meta["native_rate_zero_preparation_seconds"])

    def test_complete_repeated_message_receiver_matches_reference_semantics(self):
        scenario, truth = smoke_scenario()
        before = scenario.y.copy()
        original, old = reference(scenario, memoize=True, complete_support=True)
        counts, meta = run(scenario)
        self.assertEqual(Counter(counts), truth)
        self.assertEqual(counts, original)
        for key in ("rollback_action", "rollback_batches", "base_bank", "bank", "cancellation_action", "complete"):
            self.assertEqual(meta[key], old[key])
        for key in ("calls", "rounds", "valid_ids", "residual_sha256"):
            self.assertEqual(meta["native"][key], old["native"][key])
        for left, right in zip(meta["continuation"]["queries"], old["continuation"]["queries"], strict=True):
            for key in ("pattern", "pre_action_sha256", "llr_sha256", "ids", "new_ids", "refit_index", "cache_hit"):
                self.assertEqual(left[key], right[key])
        self.assertEqual(meta["continuation"]["stop"], old["continuation"]["stop"])
        self.assertEqual(meta["support_completion"]["additions"], old["support_completion"]["additions"])
        self.assertGreaterEqual(meta["total_seconds"],
                                meta["native_rate_zero_preparation_seconds"] + meta["underlying_receiver_seconds"])
        np.testing.assert_array_equal(scenario.y, before)

    def test_refuses_nonreference_code_or_native_budget(self):
        scenario, _ = smoke_scenario()
        scenario.model.code.list_size = 512
        with self.assertRaisesRegex(ValueError, "native SCL128"):
            run(scenario)
        original = scenario.model.code
        scenario.model.code = RateZeroPolarCode(original.payload_bits, original.n, original.crc_bits, 128)
        with self.assertRaisesRegex(ValueError, "original PolarCode"):
            run(scenario)


if __name__ == "__main__":
    unittest.main()
