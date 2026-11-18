"""Finite provenance, prefix and literal-gate checks; no historical noisy query is decoded."""

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'jobs/037_polar_tractable'))
import diagnose_count_growth as driver
from framework.polar_count_feedback import continue_feedback, support_completion
from tests.polar_count_cavity_test import smoke_scenario
from tests.polar_count_growth_test import scripted


def scripted_arm(enabled):
    scenario, bank, code, _, _ = scripted('universal')
    with patch('framework.polar_count_feedback.CountBank', bank):
        action, seen, meta = continue_feedback(scenario, Counter(), set(), code_factory=code,
            progress_extra_sweep=True, count_growth=enabled)
    return dict(action=sorted(action.items()), cancellation_action=sorted(action.items()), bank=sorted(seen),
                meta=meta, completion={}, wall_seconds=1., completion_wall_seconds=0.)


def rows(gains):
    result = []
    for index, seed in enumerate(driver.SEEDS):
        gain = gains[index] if index < len(gains) else 0
        metrics, arms = {}, {}
        for name, offset in zip(driver.MODES, (0, gain), strict=True):
            metrics[name] = dict(matched_copies=90+offset, misses=10-offset, false_copies=2,
                output_copies=92+offset, multiplicity_capable_oracle_bank_miss_floor=10-offset)
            meta = dict(queries=[], refits=[], initial_refit=dict(converged=True), complete=True, actual_scl_calls=0,
                        cache_hits=0, seconds=.5, sweeps=[], later_sweeps=0)
            arms[name] = dict(meta=meta, wall_seconds=1. if name == 'conditional_fourth' else 500., completion_wall_seconds=.1)
        result.append(dict(seed=seed, frame=0, metrics=metrics, arms=arms,
            legal_actions={name: True for name in driver.MODES}, inputs_unchanged={name: True for name in driver.MODES},
            reference_matches_archive=True, prefix_checks=dict(identical=True, continued=False), integer_bounds_verified=True,
            geometry_seconds=.1, source_seconds=.01, verification_seconds=.02,
            historical_initialization=dict(native_seconds=10., rollback_seconds=5., native_preparation_seconds=.01)))
    return result


class CountGrowthDriverTest(unittest.TestCase):
    def test_literal_design_and_full_panel(self):
        design = json.loads((driver.BASE/'design.json').read_text())
        driver.check_design(design)
        self.assertEqual(driver.SEEDS, list(range(7101, 7229)))
        self.assertEqual(len(driver.SEEDS), 128)
        design['frozen_development_gates']['minimum_net_saved_count_copies'] = 3
        with self.assertRaises(AssertionError):
            driver.check_design(design)

    def test_initialization_cannot_inherit_final_bank_or_completion(self):
        saved = dict(arms=dict(C=dict(meta=dict(rollback_action=[[1, 2]], base_bank=[1, 3],
            bank=[1, 3, 7], cancellation_action=[[7, 2]], native=dict(seconds=2.), rollback_seconds=3.,
            native_repetition_preparation_seconds=.1))))
        initial, bank, historical = driver.saved_initialization(saved)
        self.assertEqual(initial, Counter({1: 2}))
        self.assertEqual(bank, {1, 3})
        self.assertEqual(historical['native_seconds'], 2.)
        initial[8], _ = 1, bank.add(8)
        self.assertEqual(saved['arms']['C']['meta']['base_bank'], [1, 3])
        self.assertEqual(saved['arms']['C']['meta']['rollback_action'], [[1, 2]])

    def test_entire_reference_metadata_compared_except_time(self):
        arm = scripted_arm(False)
        arm['meta']['initial_refit']['objective_probe'] = -123.
        saved = dict(arms=dict(C=dict(action=arm['action'], meta=dict(continuation=copy.deepcopy(arm['meta']),
            cancellation_action=arm['cancellation_action'], bank=arm['bank'], support_completion=arm['completion']))))
        saved = driver.normalized(saved)
        arm['meta']['initial_refit']['seconds'] += 10.
        driver.verify_reference(arm, saved)
        arm['meta']['initial_refit']['objective_probe'] += 1.
        with self.assertRaises(AssertionError):
            driver.verify_reference(arm, saved)

    def test_first_four_refits_cache_bank_and_fifth_start(self):
        old, new = scripted_arm(False), scripted_arm(True)
        checked = driver.verify_prefix(old, new, set(), 3)
        self.assertTrue(checked['continued'])
        self.assertEqual(checked['shared_prefix_bank'], [0, 1, 2, 3])
        self.assertEqual(checked['shared_prefix_cancellation'], [(3, 1)])
        self.assertTrue(driver.verify_growth(new, 3))
        damaged = copy.deepcopy(new)
        damaged['meta']['queries'][4]['pre_action_sha256'] = 'wrong'
        with self.assertRaises(AssertionError):
            driver.verify_prefix(old, damaged, set(), 3)
        damaged = copy.deepcopy(new)
        damaged['meta']['refits'][0]['meta']['new_unexpected_field'] = 1
        with self.assertRaises(AssertionError):
            driver.verify_prefix(old, damaged, set(), 3)

    def test_growth_bound_recomputed_from_refit_moves(self):
        arm = scripted_arm(True)
        self.assertTrue(driver.verify_growth(arm, 3))
        for key, value in (('later_sweep_bound', 2), ('fourth_endpoint_count', 2)):
            bad = copy.deepcopy(arm)
            bad['meta'][key] = value
            with self.assertRaises(AssertionError):
                driver.verify_growth(bad, 3)
        bad = copy.deepcopy(arm)
        bad['meta']['count_growth_trace'][0]['start_count'] = 1
        with self.assertRaises(AssertionError):
            driver.verify_growth(bad, 3)

    def test_terminal_completion_once_after_cold_continuation(self):
        scenario, truth = smoke_scenario()
        initial, bank = truth.copy(), set(truth)
        with patch.object(driver, 'continue_feedback', wraps=continue_feedback) as continuation:
            with patch.object(driver, 'support_completion', wraps=support_completion) as completion:
                arm = driver.run_arm(scenario, initial, bank, 'count_growth')
        self.assertEqual(continuation.call_count, 1)
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(arm['action'], sorted(truth.items()))
        self.assertEqual(sorted(completion.call_args.args[1].items()), arm['cancellation_action'])
        self.assertEqual(initial, truth)
        self.assertEqual(bank, set(truth))
        self.assertGreaterEqual(arm['wall_seconds'], arm['completion_wall_seconds'])
        self.assertGreaterEqual(arm['wall_seconds'], arm['meta']['seconds'])

    def test_capability_gate_without_retrospective_cost_cutoff(self):
        checked = driver.summarize(rows([3, 1]))
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['net_misses_saved'], 4)
        self.assertEqual(checked['net_without_best_frame'], 1)
        self.assertEqual(checked['better_frames'], 2)
        self.assertEqual(checked['new_continuation_and_completion_seconds']['count_growth'], 500*128)
        self.assertNotIn('mixed_time_standalone_estimate', checked)
        self.assertFalse(driver.summarize(rows([4]))['passed'])
        bad = driver.summarize(rows([4, 3, -3]))
        self.assertFalse(bad['gates']['no_frame_regression_over_two'])
        bad_rows = rows([3, 1])
        bad_rows[0]['arms']['count_growth']['meta']['complete'] = False
        self.assertFalse(driver.summarize(bad_rows)['passed'])

    def test_reject_partial_or_reordered_panel_and_inconsistent_cache_counts(self):
        for bad in (rows([3, 1])[:-1], list(reversed(rows([3, 1])))):
            with self.assertRaises(ValueError):
                driver.summarize(bad)
        bad = rows([3, 1])
        bad[0]['arms']['count_growth']['meta']['cache_hits'] = 1
        with self.assertRaises(AssertionError):
            driver.summarize(bad)


if __name__ == '__main__':
    unittest.main()
