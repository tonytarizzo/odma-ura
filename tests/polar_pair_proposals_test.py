"""Independent tiny likelihood and frozen/CRC controls for bounded polar pair lists."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode, polar_transform
from framework.polar_pair_proposals import joint_pair_list, xor_pair_list


def all_pairs(code):
    payloads = np.array(list(itertools.product((0, 1), repeat=code.payload_bits)), dtype=np.uint8)
    return np.array(list(itertools.product(payloads, repeat=2)), dtype=np.uint8)


def likelihood(code, pairs, residual, amplitudes, variance):
    signal = amplitudes * (1 - 2.0 * code.encode(pairs)).sum(axis=1)
    return -((residual - signal) ** 2).sum(axis=1) / (2 * variance)


class PairProposalTests(unittest.TestCase):
    def test_full_joint_list_matches_independent_physical_likelihood_with_frozen_bits(self):
        rng = np.random.default_rng(81274)
        for b, d in [(1, 2), (2, 4), (4, 4), (3, 8)]:
            code = PolarCode(b, d, crc_bits=0)
            residual, amplitudes = rng.normal(size=d), rng.uniform(.1, 1., size=d)
            pairs, meta = joint_pair_list(code, residual, amplitudes, .71, 4 ** b)
            expected = all_pairs(code)
            self.assertEqual({tuple(p.ravel()) for p in pairs}, {tuple(p.ravel()) for p in expected})
            weights = likelihood(code, pairs, residual, amplitudes, .71)
            np.testing.assert_allclose(meta['log_weights'], weights, atol=1e-13)
            # The full SC chain metric differs from negative physical log likelihood only by a common constant.
            np.testing.assert_allclose(meta['path_metrics'] + weights,
                                       np.full(len(pairs), meta['path_metrics'][0] + weights[0]), atol=2e-13)
            self.assertTrue(np.all(np.diff(weights) <= 1e-13))

    def test_single_joint_path_matches_brute_conditional_sc_not_ml_oracle(self):
        rng = np.random.default_rng(452)
        code = PolarCode(2, 4, crc_bits=0)
        group_inputs = np.array(list(itertools.product(range(4), repeat=code.n)), dtype=np.uint8)
        group_coded = polar_transform(group_inputs)
        signs = (1 - 2.0 * (group_coded >> 1)) + (1 - 2.0 * (group_coded & 1))
        for _ in range(6):
            residual, amplitudes, variance = rng.normal(size=4), rng.uniform(.1, 1., size=4), .38
            log_weight = -((residual - amplitudes * signs) ** 2).sum(axis=1) / (2 * variance)
            allowed, chosen = np.ones(len(group_inputs), dtype=bool), []
            for j in range(code.n):
                symbol = 0 if code.frozen[j] else int(np.argmax([
                    logsumexp(log_weight[allowed & (group_inputs[:, j] == symbol)]) for symbol in range(4)]))
                allowed &= group_inputs[:, j] == symbol
                chosen.append(symbol)
            chosen = np.array(chosen, dtype=np.uint8)[code.information]
            expected = np.stack((chosen >> 1, chosen & 1))
            pairs, _ = joint_pair_list(code, residual, amplitudes, variance, 1)
            np.testing.assert_array_equal(pairs, expected[None])

    def test_full_xor_lists_cover_ordered_pairs_including_identical_messages(self):
        rng = np.random.default_rng(932)
        for b, d in [(1, 2), (2, 4), (3, 8)]:
            code = PolarCode(b, d, crc_bits=0)
            residual, amplitudes = rng.normal(size=d), rng.uniform(0., 1., size=d)
            pairs, meta = xor_pair_list(code, residual, amplitudes, .53, 2 ** b, 2 ** b)
            self.assertEqual({tuple(p.ravel()) for p in pairs}, {tuple(p.ravel()) for p in all_pairs(code)})
            np.testing.assert_allclose(meta['log_weights'], likelihood(code, pairs, residual, amplitudes, .53), atol=1e-13)
            self.assertEqual(sum(np.array_equal(p[0], p[1]) for p in pairs), 2 ** b)
            self.assertTrue(meta['guide_omits_code_consistency'])

    def test_xor_search_llrs_match_direct_local_channel_marginal_and_conditional(self):
        code = PolarCode(2, 4, crc_bits=0)
        residual, amplitudes, variance = np.array([-.9, .2, .5, 1.3]), np.array([.1, .3, .6, .2]), .47
        original, calls, decoded = PolarCode.decode_list, [], []

        def record(decoder, llr):
            result = original(decoder, llr)
            calls.append(llr.copy())
            decoded.append(result[0])
            return result

        with patch.object(PolarCode, 'decode_list', record):
            xor_pair_list(code, residual, amplitudes, variance, 4, 4)
        local_weights = -np.square(residual[:, None] - amplitudes[:, None] * np.array([2., 0., 0., -2.])) / (2 * variance)
        expected_guide = logsumexp(local_weights[:, [0, 3]], axis=1) - logsumexp(local_weights[:, [1, 2]], axis=1)
        np.testing.assert_allclose(calls[0], expected_guide, atol=2e-15)
        for llr, delta in zip(calls[1:], code.encode(decoded[0])):
            expected_conditional = local_weights[np.arange(code.n), delta] - local_weights[np.arange(code.n), 3 ^ delta]
            np.testing.assert_allclose(llr, expected_conditional, atol=2e-15)

    def test_full_crc_joint_list_filters_both_crc_constraints(self):
        code = PolarCode(1, 8, crc_bits=4)
        residual, amplitudes = np.linspace(-.8, .5, 8), np.linspace(.1, .4, 8)
        pairs, meta = joint_pair_list(code, residual, amplitudes, .7, 4 ** code.k)
        self.assertEqual(meta['terminal_paths'], 4 ** code.k)
        self.assertEqual(meta['crc_valid_paths'], 4)
        self.assertEqual({tuple(p.ravel()) for p in pairs}, {tuple(p.ravel()) for p in all_pairs(code)})
        weights = likelihood(code, pairs, residual, amplitudes, .7)
        np.testing.assert_allclose(meta['path_metrics'] + weights,
                                   np.full(len(pairs), meta['path_metrics'][0] + weights[0]), atol=2e-13)

    def test_noiseless_crc_pair_is_retained_and_both_orderings_are_represented(self):
        code = PolarCode(2, 8, crc_bits=4)
        true_pair = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        amplitudes = np.sqrt(np.arange(1., 9.) / 36.)
        residual = amplitudes * (1 - 2.0 * code.encode(true_pair)).sum(axis=0)
        for decoder, sizes in [(joint_pair_list, (256,)), (xor_pair_list, (64, 64))]:
            pairs, meta = decoder(code, residual, amplitudes, .02, *sizes)
            retained = {tuple(pair.ravel()) for pair in pairs}
            self.assertIn(tuple(true_pair.ravel()), retained)
            self.assertIn(tuple(true_pair[::-1].ravel()), retained)
            self.assertAlmostEqual(meta['log_weights'][0], 0., places=12)

    def test_invalid_crc_is_not_silently_replaced(self):
        code = PolarCode(2, 8, crc_bits=4)
        invalid_u = np.zeros(code.n, dtype=np.uint8)
        invalid_u[code.information] = [0, 0, 1, 0, 0, 0]
        amplitudes = np.full(code.n, .4)
        residual = 2 * amplitudes * (1 - 2.0 * polar_transform(invalid_u))
        for decoder, sizes in [(joint_pair_list, (1,)), (xor_pair_list, (1, 1))]:
            pairs, meta = decoder(code, residual, amplitudes, .001, *sizes)
            self.assertEqual(pairs.shape, (0, 2, code.payload_bits))
            self.assertEqual(meta['candidate_count'], 0)

    def test_bounded_search_determinism_and_no_code_mutation(self):
        code = PolarCode(6, 32, crc_bits=8, list_size=17)
        residual = np.random.default_rng(17).normal(size=32)
        amplitudes = np.full(code.n, .3)
        for decoder, sizes, maximum in [(joint_pair_list, (19,), 19), (xor_pair_list, (7, 11), 154)]:
            pairs, meta = decoder(code, residual, amplitudes, .4, *sizes)
            again, second = decoder(code, residual, amplitudes, .4, *sizes)
            np.testing.assert_array_equal(pairs, again)
            np.testing.assert_array_equal(meta['log_weights'], second['log_weights'])
            self.assertLessEqual(len(pairs), maximum)
            self.assertEqual(code.list_size, 17)

    def test_long_payload_runtime_never_requests_an_alphabet_sized_encode(self):
        code = PolarCode(88, 512, crc_bits=16)
        encode = code.encode
        largest = [0]

        def guarded_encode(messages):
            words = np.asarray(messages)
            count = int(np.prod(words.shape[:-1]))
            largest[0] = max(largest[0], count)
            self.assertLessEqual(count, 64)
            return encode(words)

        with patch.object(code, 'encode', side_effect=guarded_encode):
            joint_pair_list(code, np.zeros(code.n), np.ones(code.n), 1., 4)
            xor_pair_list(code, np.zeros(code.n), np.ones(code.n), 1., 4, 4)
        self.assertLessEqual(largest[0], 64)

    def test_invalid_inputs(self):
        code = PolarCode(2, 4, crc_bits=0)
        for residual, amplitudes, variance in [(np.zeros(3), np.ones(4), 1.), (np.zeros(4), -np.ones(4), 1.),
                                                (np.full(4, np.nan), np.ones(4), 1.), (np.zeros(4), np.ones(4), 0.),
                                                (np.zeros(4), np.ones(4), np.inf), (np.ones(4) * 1j, np.ones(4), 1.)]:
            for decoder, sizes in [(joint_pair_list, (4,)), (xor_pair_list, (4, 4))]:
                with self.assertRaises(ValueError):
                    decoder(code, residual, amplitudes, variance, *sizes)
        for bad in [0, -1, 1.2, True]:
            with self.assertRaises(ValueError):
                joint_pair_list(code, np.zeros(4), np.ones(4), 1., bad)
            with self.assertRaises(ValueError):
                xor_pair_list(code, np.zeros(4), np.ones(4), 1., 4, bad)


if __name__ == '__main__':
    unittest.main()
