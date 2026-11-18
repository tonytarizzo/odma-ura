"""Fresh-panel capability, reinvestment and attribution gates remain logically separate."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

from tests.polar_count_cavity_test import smoke_scenario
from tests.polar_progress_continuation_driver_test import prefix_fixture

JOB = Path(__file__).resolve().parents[1] / 'jobs/037_polar_tractable'
sys.path.insert(0, str(JOB))
SPEC = importlib.util.spec_from_file_location('progress_confirmation', JOB / 'diagnose_progress_confirmation.py')
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def fixture():
    rows = []
    for seed in DRIVER.SEEDS:
        continuation = dict(initial_refit=dict(converged=True), refits=[], queries=[], actual_scl_calls=0,
            cache_hits=0, seconds=.1, extra_sweep_triggered=False)
        arm = dict(action=[(7, 98)], wall_seconds=8., meta=dict(native=dict(seconds=6.), rollback_seconds=1.,
            continuation=continuation, support_completion=dict(seconds=.01), complete=True))
        metric = dict(matched_copies=98, misses=2, false_copies=0, output_copies=98, multiplicity_capable_oracle_bank_miss_floor=2)
        rows.append(dict(seed=seed, frame=0, arms={name:deepcopy(arm) for name in DRIVER.MODES},
            metrics={name:deepcopy(metric) for name in DRIVER.MODES}, receiver_with_geometry_seconds=dict(A=10., B=8., C=8.2),
            geometry_seconds=.5, source_seconds=.01, verification_seconds=.01,
            legal_actions={name:True for name in DRIVER.MODES}, original_inputs_unchanged={name:True for name in DRIVER.MODES},
            prefix_checks=dict(B_C_initialization=True, B_C_original_prefix=True), A_B_native_aggregates=True,
            A_B_complete_semantics=True, exclusive_retained=[], source_repeated_copies=0, source_pattern_collision_copies=1))
    for index in (0, 1):
        rows[index]['metrics']['C'].update(misses=0, matched_copies=100, output_copies=100, multiplicity_capable_oracle_bank_miss_floor=0)
        rows[index]['arms']['C']['meta']['continuation']['extra_sweep_triggered'] = True
    return rows


class ProgressConfirmationTest(unittest.TestCase):
    def test_capability_boundary_and_new_complete_cost_accounting(self):
        result = DRIVER.summarize(fixture())
        self.assertTrue(result['passed'])
        self.assertTrue(result['capability_passed'] and result['reinvestment_passed'])
        self.assertTrue(result['pure_acceleration_attribution_passed'])
        self.assertEqual((result['net_misses_saved_C_vs_B'], result['better_frames'], result['net_without_best_frame']), (4, 2, 2))
        self.assertAlmostEqual(result['receiver_with_geometry_seconds']['A'], 1280.)
        self.assertAlmostEqual(result['C_over_A_cost_ratio'], .82)
        self.assertAlmostEqual(result['C_over_B_cost_ratio'], 1.025)
        self.assertAlmostEqual(result['C_minus_B_seconds'], 25.6)
        self.assertEqual(len(DRIVER.ORDERS), 6)
        self.assertEqual(set(DRIVER.ORDERS), set(__import__('itertools').permutations(DRIVER.MODES)))

    def test_attribution_failure_does_not_erase_C_B_capability_or_measured_C_A_result(self):
        rows = fixture()
        rows[0]['A_B_native_aggregates'] = False
        rows[0]['A_B_complete_semantics'] = False
        rows[0]['arms']['A']['action'] = [(8, 98)]
        result = DRIVER.summarize(rows)
        self.assertTrue(result['passed'])
        self.assertFalse(result['pure_acceleration_attribution_passed'])
        self.assertFalse(any(result['attribution_gate'].values()))

    def test_cost_and_original_reference_miss_tests_are_separate_from_capability(self):
        rows = fixture()
        for row in rows:
            row['receiver_with_geometry_seconds']['C'] = 11.
        result = DRIVER.summarize(rows)
        self.assertTrue(result['capability_passed'])
        self.assertFalse(result['reinvestment_passed'])
        self.assertFalse(result['passed'])
        rows = fixture()
        for row in rows:
            row['metrics']['A']['misses'] = 0
        result = DRIVER.summarize(rows)
        self.assertTrue(result['capability_passed'])
        self.assertFalse(result['reinvestment_gate']['C_misses_at_most_A'])

    def test_concentration_regression_and_integrity_can_each_reject(self):
        rows = fixture()
        rows[0]['metrics']['B']['misses'] = 4
        rows[1]['metrics']['C']['misses'] = 2
        result = DRIVER.summarize(rows)
        self.assertTrue(result['capability_gate']['four_net_saved_copies'])
        self.assertFalse(result['capability_gate']['two_better_frames'])
        self.assertFalse(result['capability_gate']['net_without_best_frame'])
        rows = fixture()
        for index in (0, 1):
            rows[index]['metrics']['B']['misses'] = 5
        rows[2]['metrics']['B']['misses'] = 0
        rows[2]['metrics']['C']['misses'] = 3
        self.assertFalse(DRIVER.summarize(rows)['capability_gate']['no_individual_regression_over_two'])
        for category, key, gate in [('prefix_checks', 'B_C_original_prefix', 'B_C_initialization_and_original_prefix'),
                                  ('original_inputs_unchanged', 'C', 'inputs_unchanged'),
                                  ('legal_actions', 'C', 'all_output_counts_legal')]:
            rows = fixture(); rows[0][category][key] = False
            self.assertFalse(DRIVER.summarize(rows)['capability_gate'][gate])
        rows = fixture(); rows[0]['arms']['C']['meta']['continuation']['initial_refit']['converged'] = False
        self.assertFalse(DRIVER.summarize(rows)['capability_gate']['all_refits_converged'])

    def test_prefix_mismatch_is_explicit_and_full_panel_and_counters_are_required(self):
        old, new = prefix_fixture(True)
        arms = {}
        for name, source in [('B', old), ('C', new)]:
            arms[name] = dict(action=source['action'], meta=dict(continuation=source['meta'],
                cancellation_action=source['cancellation_action'], bank=source['bank'], rollback_action=[],
                rollback_batches=[], base_bank=[], native=dict(calls=1, rounds=[1], valid_ids=[7], residual_sha256='r')))
        self.assertTrue(DRIVER.original_prefix_checks(arms)['B_C_original_prefix'])
        arms['C']['meta']['continuation']['queries'][0]['ids'] = [8]
        result = DRIVER.original_prefix_checks(arms)
        self.assertFalse(result['B_C_original_prefix'])
        self.assertIsNone(result['trigger'])
        with self.assertRaises(ValueError):
            DRIVER.summarize(fixture()[:-1])
        rows = fixture(); rows[-1]['seed'] = rows[0]['seed']
        with self.assertRaises(ValueError):
            DRIVER.summarize(rows)
        rows = fixture(); rows[0]['arms']['C']['meta']['continuation']['actual_scl_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'counters'):
            DRIVER.summarize(rows)

    def test_legal_counts_keep_multiplicities_and_reject_invalid_or_duplicate_pairs(self):
        scenario, truth = smoke_scenario()
        self.assertTrue(DRIVER.valid_action(scenario, sorted(truth.items())))
        word = next(iter(truth))
        for pairs in ([(word, 1), (word, 1)], [(word, scenario.k+1)], [(word, -1)], [(word, 1.5)], [(1 << scenario.model.B, 1)]):
            self.assertFalse(DRIVER.valid_action(scenario, pairs))

    def test_initial_and_prefix_fit_metadata_ignore_only_elapsed_seconds(self):
        old, new = prefix_fixture(True)
        arms = {}
        for index, (name, source) in enumerate([('B', old), ('C', new)]):
            source['meta']['initial_refit'].update(initial_objective=2., final_objective=3., converged=True, seconds=index+.1)
            source['meta']['refits'] = [dict(query=0, sweep=0, new_ids=[8], meta=dict(moves=[], bank_size=2,
                initial_objective=4., final_objective=4., converged=True, seconds=index+.2))]
            arms[name] = dict(action=source['action'], meta=dict(continuation=source['meta'],
                cancellation_action=source['cancellation_action'], bank=source['bank'], rollback_action=[],
                rollback_batches=[], base_bank=[], native=dict(calls=1, rounds=[1], valid_ids=[7], residual_sha256='r')))
        self.assertTrue(DRIVER.original_prefix_checks(arms)['B_C_count_fit_metadata'])
        self.assertTrue(DRIVER.original_prefix_checks(arms)['B_C_original_prefix'])
        arms['C']['meta']['continuation']['initial_refit']['initial_objective'] += 1
        checks = DRIVER.original_prefix_checks(arms)
        self.assertFalse(checks['B_C_initialization'])
        self.assertFalse(checks['B_C_original_prefix'])
        self.assertFalse(checks['B_C_count_fit_metadata'])
        arms['C']['meta']['continuation']['initial_refit']['initial_objective'] -= 1
        arms['C']['meta']['continuation']['refits'][0]['meta']['bank_size'] = 3
        checks = DRIVER.original_prefix_checks(arms)
        self.assertTrue(checks['B_C_initialization'])
        self.assertFalse(checks['B_C_original_prefix'])
        self.assertFalse(checks['B_C_count_fit_metadata'])


if __name__ == '__main__':
    unittest.main()
