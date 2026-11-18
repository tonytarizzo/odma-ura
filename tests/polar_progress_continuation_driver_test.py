"""Prospective fourth-block gates and prefix integrity, without channel observations."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

JOB = Path(__file__).resolve().parents[1] / 'jobs/037_polar_tractable'
sys.path.insert(0, str(JOB))
SPEC = importlib.util.spec_from_file_location('progress_driver', JOB / 'diagnose_progress_continuation.py')
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def fixture():
    rows = []
    for seed in DRIVER.SEEDS:
        arm = dict(metrics=dict(misses=1, false_copies=0, output_copies=99, bank_misses=1), wall_seconds=.1,
            meta=dict(complete=True, initial_refit=dict(converged=True), refits=[], queries=[],
                actual_scl_calls=0, cache_hits=0, ranking_seconds=.01))
        rows.append(dict(seed=seed, arms={name:deepcopy(arm) for name in DRIVER.MODES},
            exclusive_retained=[], geometry_seconds=.2, paid_initialization_seconds=1., triggered=False,
            three_trace_matches_original=True, prefix_identical=True))
    for i in (0, 1):
        rows[i]['arms']['conditional_fourth']['metrics'].update(misses=0, output_copies=100, bank_misses=0)
        rows[i]['exclusive_retained'] = [1000+i]
        rows[i]['triggered'] = True
    return rows


def query(index):
    return dict(query=index, sweep=index, stage='ranked_patterns', pattern=0, list_size=128, context='raw',
        restored=[], restored_count=0, outside_count=99, pending=99, variance=.1, pre_action_sha256='a',
        llr_sha256='l', ids=[7], new_ids=[], refit_index=None, cache_hit=index > 0)


def prefix_fixture(trigger):
    meta = dict(initial_refit_action=[(7, 1)], initial_refit=dict(moves=[]), queries=[query(i) for i in range(3)],
        refits=[], complete=True, stop='sweep_cap', sweeps=[dict(sweep=i, moves=int(trigger)) for i in range(3)])
    old = dict(meta=meta, action=[(7, 1)], cancellation_action=[(7, 1)], bank=[7])
    new = deepcopy(old)
    new['meta']['extra_sweep_triggered'] = trigger
    new['meta']['extra_sweep_initial_action'] = [(7, 1)] if trigger else None
    if trigger:
        new['meta']['queries'].append(query(3))
        new['meta']['sweeps'].append(dict(sweep=3, moves=0))
    return old, new


class ProgressDriverTest(unittest.TestCase):
    def test_boundary_gates_are_separate_and_cost_is_explicitly_mixed(self):
        result = DRIVER.summarize(fixture())
        self.assertTrue(result['advance'])
        self.assertEqual((result['net_misses_saved'], result['net_without_best_frame']), (2, 1))
        for name in DRIVER.MODES:
            self.assertAlmostEqual(result['new_continuation_wall_seconds'][name], 9.6)
            self.assertAlmostEqual(result['mixed_time_standalone_estimate'][name], 124.8)
        rows = fixture()
        rows[1]['exclusive_retained'] = rows[0]['exclusive_retained']
        self.assertFalse(DRIVER.summarize(rows)['gates']['exclusive_IDs'])
        rows = fixture()
        rows[0]['exclusive_retained'] += rows[1]['exclusive_retained']
        rows[1]['exclusive_retained'] = []
        self.assertFalse(DRIVER.summarize(rows)['gates']['exclusive_frames'])

    def test_concentration_caps_and_prefix_integrity_cannot_be_ignored(self):
        rows = fixture()
        rows[0]['arms']['three']['metrics']['misses'] = 2
        rows[1]['arms']['conditional_fourth']['metrics']['misses'] = 1
        self.assertFalse(DRIVER.summarize(rows)['gates']['net_without_best_frame'])
        for mode, field, value, gate in [('conditional_fourth', 'complete', False, 'all96_complete')]:
            rows = fixture(); rows[0]['arms'][mode]['meta'][field] = value
            self.assertFalse(DRIVER.summarize(rows)['gates'][gate])
        rows = fixture(); rows[0]['arms']['conditional_fourth']['meta']['initial_refit']['converged'] = False
        self.assertFalse(DRIVER.summarize(rows)['gates']['all_fits_converged'])
        rows = fixture(); rows[0]['arms']['conditional_fourth']['metrics']['output_copies'] = 101
        self.assertFalse(DRIVER.summarize(rows)['gates']['count_caps'])
        rows = fixture(); rows[0]['prefix_identical'] = False
        self.assertFalse(DRIVER.summarize(rows)['gates']['identical_original_prefix'])

    def test_fixed_panel_and_actual_call_counters_are_required(self):
        with self.assertRaises(ValueError):
            DRIVER.summarize(fixture()[:-1])
        rows = fixture(); rows[-1]['seed'] = rows[0]['seed']
        with self.assertRaises(ValueError):
            DRIVER.summarize(rows)
        rows = fixture(); rows[0]['arms']['conditional_fourth']['meta']['actual_scl_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'counters'):
            DRIVER.summarize(rows)

    def test_trigger_prefix_and_nontrigger_final_action_are_verified(self):
        for trigger in (False, True):
            old, new = prefix_fixture(trigger)
            self.assertEqual(DRIVER.verify_prefix(old, new), trigger)
            new['meta']['queries'][0]['ids'] = [8]
            with self.assertRaises(AssertionError):
                DRIVER.verify_prefix(old, new)
        old, new = prefix_fixture(True)
        new['meta']['queries'][0]['cache_hit'] = True
        with self.assertRaises(AssertionError):
            DRIVER.verify_prefix(old, new)
        old, new = prefix_fixture(False)
        new['action'] = [(8, 1)]
        with self.assertRaises(AssertionError):
            DRIVER.verify_prefix(old, new)

    def test_old_uncached_lists_compare_logically_and_normalize_JSON_pairs(self):
        old, _ = prefix_fixture(False)
        old['meta']['sweep_cap_with_new_ids'] = False
        prior = DRIVER.normalized(old)
        prior['meta']['queries'][1]['cache_hit'] = False
        saved = dict(seed=6501, rollback_action=[(7, 1)], initial_bank=[7], native_wall_seconds=1., rollback_wall_seconds=2.,
            arms=dict(raw128=dict(meta=prior['meta'], action=prior['cancellation_action'], bank=[7])))
        DRIVER.verify_three(old, saved)
        saved['arms']['raw128']['meta']['queries'][0]['llr_sha256'] = 'changed'
        with self.assertRaises(AssertionError):
            DRIVER.verify_three(old, saved)


if __name__ == '__main__':
    unittest.main()
