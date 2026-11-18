"""Endpoint stopping fixtures; scripted count fits are scheduler tests, not recovery evidence."""

from collections import Counter
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from baselines.polar import PolarCode
from framework.polar_count_cavity import cavity_query
from framework.polar_count_feedback import continue_feedback, support_completion
from framework.polar_count_growth import run
from framework.polar_progress_continuation import run as reference_run
from framework.polar_tractable import PolarModel, PolarScenario, bits
from tests.polar_count_cavity_test import smoke_scenario
from tests.polar_progress_continuation_test import semantic


def scripted(mode):
    model = PolarModel(PolarCode(3, 8, 4, 128), np.arange(8).reshape(1, 8), np.full((1, 8), 1/np.sqrt(8)), 8)
    scenario = PolarScenario(model, np.full(8, 10.), 3, .1)
    states = [Counter(), Counter({0: 1}), Counter({1: 1}), Counter({2: 1})]
    states += [Counter({2: 1, 3: 1}), Counter({2: 1, 3: 1, 4: 1}), Counter({2: 1, 3: 1, 5: 1})]
    if mode in ('neutral', 'temporary', 'negative', 'no_new', 'incomplete'):
        states[3] = Counter({1: 1, 2: 1})
        states[4] = (Counter({2: 1, 3: 1}) if mode == 'neutral' else Counter({3: 1}) if mode == 'negative'
                     else states[3].copy() if mode in ('temporary', 'no_new') else Counter({1: 1, 2: 1, 3: 1}))
    if mode == 'capacity':
        states[4] = Counter({2: 2, 3: 1})
        states[5] = Counter({2: 2, 4: 1})
    if mode in ('universal', 'cache_stop'):
        states = [Counter(), Counter({0: 1}), Counter({1: 1}), Counter(), Counter({3: 1}),
                  Counter({3: 1, 4: 1}), Counter({3: 1, 4: 1, 5: 1}), Counter({3: 1, 4: 1, 6: 1})]
    channels, raw_states, instances, decoders = {}, [], [], []
    for phase, action in enumerate(states):
        raw = scenario.y+(1 if mode == 'cache_stop' and phase == 4 else phase)*100.
        raw_states.append(raw)
        llr, _ = cavity_query(scenario, action, raw, 0)
        payload = 2 if mode == 'no_new' and phase == 3 else 1 if mode == 'cache_stop' and phase == 4 else phase
        channels[llr.tobytes()] = bits(payload, 3)[None, :]

    class Code(PolarCode):
        def __init__(self, *args):
            super().__init__(*args)
            decoders.append(self)

        def decode_list(self, llr):
            return channels[llr.tobytes()].copy(), np.zeros(1)

    class Bank:
        def __init__(self, current, ids):
            instances.append(self)
            self.words = sorted(ids)
            self.last_raw_residual = current.y.copy()
            self.matrix = csc_matrix((8, 0))
            self.construction_seconds, self.refit_calls = 0., 0

        def extend(self, words):
            fresh = sorted(set(words)-set(self.words))
            self.words = sorted(set(self.words).union(words))
            return fresh, 0.

        def refit(self, initial, max_moves):
            assert max_moves == 2000
            phase = self.refit_calls
            self.refit_calls += 1
            action = states[phase].copy()
            assert set(action) <= set(self.words) and action.total() <= scenario.k
            moves = ([{'kind': 'drop', 'old': word, 'new': None} for word in sorted(initial)
                      for _ in range(max(0, initial[word]-action[word]))]
                     + [{'kind': 'add', 'old': None, 'new': word} for word in sorted(action)
                        for _ in range(max(0, action[word]-initial[word]))])
            if len(moves) == 2 and moves[0]['kind'] == 'drop' and moves[1]['kind'] == 'add':
                moves = [{'kind': 'swap', 'old': moves[0]['old'], 'new': moves[1]['new']}]
            if mode == 'temporary' and phase == 4:
                moves = [{'kind': 'add', 'old': None, 'new': 3}, {'kind': 'drop', 'old': 3, 'new': None}]
            self.last_raw_residual = raw_states[phase].copy()
            converged = not (mode == 'incomplete' and phase == 4 or mode == 'initial_incomplete' and phase == 0)
            return action, {'converged': converged, 'moves': moves, 'seconds': 0.}

    return scenario, Bank, Code, instances, decoders


class CountGrowthTest(unittest.TestCase):
    def execute(self, mode, enabled=True, integral_float_k=False):
        scenario, bank, code, instances, decoders = scripted(mode)
        if integral_float_k:
            scenario.k = float(scenario.k)
        initial, ids = Counter(), set()
        with patch('framework.polar_count_feedback.CountBank', bank):
            result = continue_feedback(scenario, initial, ids, code_factory=code,
                                       progress_extra_sweep=True, count_growth=enabled)
        self.assertEqual((initial, ids), (Counter(), set()))
        self.assertEqual((len(instances), len(decoders)), (1, 1))
        self.assertEqual(instances[0].refit_calls, len(result[2]['refits'])+1)
        self.assertLessEqual(len(result[2]['sweeps']), scenario.k+4)
        self.assertLessEqual(len(result[2]['queries']), (scenario.k+4)*(scenario.k+5))
        return result

    def test_strict_growth_reaches_sharp_and_universal_bounds(self):
        action, bank, meta = self.execute('universal')
        self.assertEqual(len(meta['sweeps']), 7)
        self.assertEqual(meta['fourth_endpoint_count'], 1)
        self.assertEqual(meta['later_sweeps'], meta['later_sweep_bound'])
        self.assertEqual([r['continue'] for r in meta['count_growth_trace']], [True, True, True, False])
        self.assertEqual(meta['stop'], 'no_endpoint_count_growth')
        self.assertEqual(action, Counter({3: 1, 4: 1, 6: 1}))
        self.assertEqual(bank, set(range(7)))
        self.assertTrue(meta['complete'])

    def test_growth_to_K_allows_one_final_block_and_preserves_multiplicity(self):
        action, _, meta = self.execute('capacity', integral_float_k=True)
        self.assertEqual(len(meta['sweeps']), 5)
        self.assertEqual(meta['fourth_endpoint_count'], 3)
        self.assertEqual(meta['later_sweep_bound'], 1)
        self.assertEqual(meta['count_growth_trace'][1]['start_count'], 3)
        self.assertEqual(action, Counter({2: 2, 4: 1}))

    def test_private_exact_cache_survives_later_block(self):
        action, bank, meta = self.execute('cache_stop')
        self.assertEqual(len(meta['sweeps']), 5)
        self.assertEqual(meta['actual_scl_calls'], 4)
        self.assertEqual(meta['cache_hits'], 1)
        self.assertTrue(meta['queries'][4]['cache_hit'])
        self.assertEqual(meta['queries'][4]['llr_sha256'], meta['queries'][1]['llr_sha256'])
        self.assertEqual(meta['queries'][4]['ids'], meta['queries'][1]['ids'])
        self.assertEqual(meta['stop'], 'full_sweep_no_new_ids')
        self.assertEqual(action, Counter({3: 1}))
        self.assertEqual(bank, {0, 1, 2, 3})

    def test_equal_negative_and_temporary_growth_do_not_continue(self):
        for mode in ('neutral', 'negative', 'temporary'):
            with self.subTest(mode=mode):
                _, _, meta = self.execute(mode)
                self.assertEqual(len(meta['sweeps']), 4)
                self.assertGreater(meta['sweeps'][3]['moves'], 0)
                self.assertFalse(meta['count_growth_trace'][0]['continue'])
                self.assertEqual(meta['stop'], 'no_endpoint_count_growth')

    def test_no_new_and_incomplete_stops_have_priority(self):
        for mode, reason in (('no_new', 'full_sweep_no_new_ids'), ('incomplete', 'count_move_cap')):
            _, _, meta = self.execute(mode)
            self.assertEqual(len(meta['sweeps']), 4)
            self.assertEqual(meta['stop'], reason)
            self.assertEqual(meta['count_growth_trace'][0]['reason'], reason)
            self.assertFalse(meta['count_growth_trace'][0]['continue'])
        _, _, meta = self.execute('initial_incomplete')
        self.assertEqual(meta['queries'], [])
        self.assertEqual(meta['count_growth_trace'], [])
        self.assertFalse(meta['complete'])

    def test_existing_prefix_is_identical_and_enabled_runs_do_not_restart(self):
        old_action, old_bank, old = self.execute('universal', False)
        _, _, new = self.execute('universal', True)
        self.assertEqual(len(old['sweeps']), 4)
        self.assertEqual(semantic(old['queries']), semantic(new['queries'][:len(old['queries'])]))
        self.assertEqual(semantic(old['refits']), semantic(new['refits'][:len(old['refits'])]))
        self.assertEqual(old['sweeps'], new['sweeps'][:4])
        self.assertEqual(old['initial_refit_action'], new['initial_refit_action'])
        self.assertEqual(old['count_action_path'], new['count_action_path'][:len(old['count_action_path'])])
        self.assertEqual(old_action.total(), new['count_growth_trace'][0]['end_count'])
        self.assertEqual(old_bank, set(range(4)))

    def test_archived_default_and_fourth_provider_semantics(self):
        path = Path(__file__).resolve().parents[1]/'results/polar_dec35_exploration_2026-11-11/count_growth/source_before/framework/polar_count_feedback.py'
        if not path.exists():
            self.skipTest('Historical ignored snapshot is absent on this checkout; phase preflight requires it and zero skips')
        spec = importlib.util.spec_from_file_location('archived_count_feedback', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        scenario, truth = smoke_scenario()
        for enabled in (False, True):
            left = old.continue_feedback(scenario, truth, set(truth), progress_extra_sweep=enabled)
            right = continue_feedback(scenario, truth, set(truth), progress_extra_sweep=enabled, count_growth=False)
            self.assertEqual(left[:2], right[:2])
            self.assertEqual(semantic(left[2]), semantic(right[2]))

    def test_real_repeated_wrapper_and_one_terminal_completion(self):
        scenario, truth = smoke_scenario()
        original_y, original_code = scenario.y.copy(), scenario.model.code
        expected, _ = reference_run(scenario)
        with patch('framework.polar_count_feedback.support_completion', wraps=support_completion) as finish:
            actual, meta = run(scenario)
        self.assertEqual(Counter(actual), truth)
        self.assertEqual(actual, expected)
        self.assertEqual(finish.call_count, 1)
        self.assertEqual(sorted(finish.call_args.args[1].items()), meta['cancellation_action'])
        self.assertEqual(meta['continuation']['later_sweeps'], 0)
        self.assertIs(scenario.model.code, original_code)
        np.testing.assert_array_equal(scenario.y, original_y)
        for options in ({'count_growth': True}, {'count_growth': 1, 'progress_extra_sweep': True}):
            with self.assertRaises(ValueError):
                continue_feedback(scenario, truth, set(truth), **options)


if __name__ == '__main__':
    unittest.main()
