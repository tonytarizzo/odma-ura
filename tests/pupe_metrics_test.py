"""Multiplicity PUPE, decoder readout, and historical-result isolation."""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from baselines import message_bits
from benchmarks.ura_comparison import decoded_lists, frame_metrics
from benchmarks.ura_merge import aggregate, pupe_definition
from framework.metrics import batch_evaluate, evaluate_counts as torch_metrics
from tests.ccs_bound_curve import tree_survivors
from src.cache import hash_key, lookup
from src.metrics import LEGACY_PUPE_DEFINITION, PUPE_DEFINITION, evaluate_counts, reported_counts


class PUPETests(unittest.TestCase):
    def test_repeated_messages_require_repeated_outputs(self):
        truth = message_bits([1, 1, 3], 3)
        for decoded, expected in (([1, 1, 3], 0), ([1, 3], 1 / 3), ([1, 3, 3], 1 / 3), ([1], 2 / 3), ([], 1)):
            with self.subTest(decoded=decoded):
                result = frame_metrics(truth, message_bits(decoded, 3))
                self.assertAlmostEqual(result["pupe"], expected)
                self.assertEqual(result["pupe_metric_version"], 2)
        result = frame_metrics(truth, message_bits([1, 3, 3], 3))
        self.assertEqual(result["list_pupe"], 0)
        self.assertAlmostEqual(result["multiset_false_alarm_rate"], 1 / 3)

    def test_numpy_torch_and_bit_multisets_agree_exhaustively(self):
        states = [np.bincount(w, minlength=4) for w in itertools.combinations_with_replacement(range(4), 3)]
        for truth, estimate in itertools.product(states, repeat=2):
            expected = np.abs(truth - estimate).sum() / 6
            truth_bits = message_bits(np.repeat(np.arange(4), truth), 2)
            decoded = message_bits(np.repeat(np.arange(4), estimate), 2)
            array = evaluate_counts(truth, estimate)
            tensor = torch_metrics(torch.tensor(truth), torch.tensor(estimate))
            self.assertEqual(array, tensor)
            self.assertAlmostEqual(array["pupe"], expected)
            self.assertAlmostEqual(frame_metrics(truth_bits, decoded)["pupe"], expected)

    def test_distinct_truth_preserves_historical_pupe(self):
        truth = np.array([1, 1, 1, 0, 0])
        for w in itertools.combinations_with_replacement(range(5), 3):
            result = evaluate_counts(truth, np.bincount(w, minlength=5))
            self.assertEqual(result["pupe"], result["list_pupe"])

    def test_count_readout_retains_multiplicity_with_candidate_indices(self):
        output = SimpleNamespace(counts=torch.tensor([[0., 2., 1.]]))
        for indices, expected in ((None, [1, 1, 2]), (torch.tensor([[2, 5, 7]]), [5, 5, 7])):
            actual = decoded_lists(output, indices, 3, 3)[0]
            np.testing.assert_array_equal(actual, message_bits(expected, 3))

    def test_budget_counts_occurrences_and_never_uses_truth_to_rank(self):
        np.testing.assert_array_equal(reported_counts(np.array([3, 3, 0]), 3), [3, 0, 0])
        result = evaluate_counts(np.array([1, 2, 0]), np.array([3, 3, 0]))
        self.assertAlmostEqual(result["pupe"], 2 / 3)
        self.assertEqual(result["list_size"], 3)
        self.assertEqual(result["list_overflow"], 3)
        self.assertEqual(result["unique_list_size"], 1)
        with self.assertRaises(ValueError):
            frame_metrics(message_bits([1, 1, 3], 3), message_bits([1, 1, 3, 3], 3))

    def test_short_outputs_count_missing_copies(self):
        result = evaluate_counts(np.array([2, 1]), np.array([1, 1]))
        self.assertAlmostEqual(result["pupe"], 1 / 3)
        self.assertEqual(result["list_pupe"], 0)
        self.assertEqual(result["false_instances"], 0)
        # L1/(2K) is not the right formula when the estimate totals less than K.
        self.assertNotEqual(result["pupe"], result["l1_err"] / 2)

    def test_tree_output_budget_preserves_root_ranking(self):
        code = SimpleNamespace(section_bits=2, num_sections=1, message_bits=[2])
        decoded, meta = tree_survivors(code, [np.array([3, 1, 2])], 2, 100)
        self.assertEqual(decoded, [3, 1])
        self.assertEqual(meta["raw_survivors"], 3)
        self.assertEqual(meta["output_overflow"], 1)

    def test_soft_negative_and_nonfinite_counts_are_rejected(self):
        for bad in ([.5, .5], [-1, 2], [np.nan, 1], [np.inf, 1], [1j, 1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate_counts(np.array([1, 1]), np.array(bad))
        with self.assertRaises(ValueError):
            evaluate_counts(np.array([1, 1]), np.array([1, 1]), max_list_size=3)

    def test_batch_metric_keeps_version(self):
        truth = torch.tensor([[2, 1], [2, 1]])
        per, mean = batch_evaluate(truth, torch.tensor([[1, 1], [2, 1]]))
        self.assertAlmostEqual(mean["pupe"], 1 / 6)
        self.assertEqual(mean["pupe_metric_version"], 2)
        self.assertEqual(len(per), 2)

    def test_legacy_cache_is_not_reused(self):
        key = hash_key({}, "test", {}, 0)
        row = {"metrics": {"pupe": 0}}
        self.assertIsNone(lookup({key: row}, {}, "test", {}, 0))
        row["metrics"]["pupe_metric_version"] = 2
        self.assertIs(lookup({key: row}, {}, "test", {}, 0), row)

    def test_mixed_metric_results_cannot_be_merged(self):
        legacy = {"evaluation": [{"means": {"pupe": 0}}]}
        new = {"evaluation": [{"means": {"pupe": 0, "pupe_metric_version": 2}}]}
        self.assertEqual(pupe_definition([legacy]), LEGACY_PUPE_DEFINITION)
        self.assertEqual(pupe_definition([new]), PUPE_DEFINITION)
        with self.assertRaisesRegex(ValueError, "different PUPE"):
            aggregate([legacy, new])


if __name__ == "__main__":
    unittest.main()
