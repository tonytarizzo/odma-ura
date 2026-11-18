"""Exact-law checks for rate-zero collapse; finite-precision list equivalence is a separate experiment."""

from itertools import product
import unittest

import numpy as np

from baselines.polar import PolarCode
from framework.polar_fast_scl import RateZeroPolarCode


class RateZeroPolarCodeTest(unittest.TestCase):
    def test_encoding_crc_information_and_frozen_subtrees_are_preserved(self):
        for bits, length, crc in ((2, 4, 0), (3, 8, 0), (3, 16, 4), (6, 32, 8), (87, 512, 16)):
            original, fast = PolarCode(bits, length, crc, 128), RateZeroPolarCode(bits, length, crc, 128)
            np.testing.assert_array_equal(fast.information, original.information)
            np.testing.assert_array_equal(fast.frozen, original.frozen)
            inputs = np.array([np.zeros(bits), np.ones(bits), np.arange(bits) % 2], dtype=np.uint8)
            np.testing.assert_array_equal(fast.encode(inputs), original.encode(inputs))
            width = 1
            while width <= length:
                for start in range(0, length, width):
                    self.assertEqual(bool(fast._rate_zero[(length + start) // width]),
                                     bool(original.frozen[start:start + width].all()))
                width *= 2

    def test_full_list_matches_exhaustive_legal_codewords_and_exact_channel_metric(self):
        for bits, length, crc, coordinates in ((2, 4, 0, (0, 1, 2, 3)), (3, 8, 0, (0, 1, 4, 7)),
                                               (3, 16, 4, (0, 3, 8, 15))):
            code = RateZeroPolarCode(bits, length, crc, 1 << (bits + crc))
            payloads = np.array(list(product((0, 1), repeat=bits)), dtype=np.uint8)
            signs = 1. - 2. * code.encode(payloads)
            for values in product((-2., 0., 2.), repeat=4):
                llr = np.zeros(length); llr[list(coordinates)] = values
                decoded, metrics = code.decode_list(llr)
                expected = dict(zip(map(tuple, payloads), np.logaddexp(0., -signs * llr).sum(axis=1), strict=True))
                self.assertEqual(set(map(tuple, decoded)), set(expected))
                for payload, metric in zip(decoded, metrics, strict=True):
                    self.assertAlmostEqual(metric, expected[tuple(payload)], places=12)

    def test_zero_ties_and_bounded_lists_preserve_legality_and_metric(self):
        llrs = [np.zeros(16), np.linspace(-2., 3., 16), np.array([0., 2., -2., 0.] * 4)]
        for list_size in (1, 2, 4, 8, 128):
            code, original = RateZeroPolarCode(3, 16, 4, list_size), PolarCode(3, 16, 4, list_size)
            zero, _ = code.decode_list(llrs[0])
            old_zero, _ = original.decode_list(llrs[0])
            np.testing.assert_array_equal(zero, old_zero)
            for llr in llrs:
                payloads, metrics = code.decode_list(llr)
                self.assertLessEqual(len(payloads), list_size)
                self.assertEqual(payloads.shape, (len(metrics), 3))
                self.assertTrue(np.all(np.diff(metrics) >= 0))
                expected = (np.logaddexp(0., -(1. - 2. * code.encode(payloads)) * llr).sum(axis=1)
                            if len(payloads) else np.empty(0))
                np.testing.assert_allclose(metrics, expected, rtol=5e-12, atol=5e-12)

    def test_invalid_llr_contract_and_input_immutability(self):
        code = RateZeroPolarCode(3, 16, 4, 128)
        llr = np.linspace(-1., 2., 16)
        before = llr.copy()
        code.decode_list(llr)
        np.testing.assert_array_equal(llr, before)
        for invalid in (np.zeros(15), np.zeros((1, 16)), np.full(16, np.inf), np.full(16, np.nan)):
            with self.assertRaisesRegex(ValueError, "finite LLR"):
                code.decode_list(invalid)


if __name__ == "__main__":
    unittest.main()
