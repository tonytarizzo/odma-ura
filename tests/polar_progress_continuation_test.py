"""One progress-triggered extra block, preserving the existing three-block policy by default."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from baselines.polar import PolarCode
from framework.polar_count_cavity import cavity_query, continue_cavity
from framework.polar_count_feedback import continue_feedback, run
from framework.polar_tractable import PolarModel, PolarScenario, bits
from tests.polar_count_cavity_test import smoke_scenario


def semantic(value):
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items() if not key.endswith("seconds")}
    if isinstance(value, (list, tuple)):
        return [semantic(item) for item in value]
    return value


def scripted(mode):
    """Only count-fit decisions are scripted; LLRs, exact memoization and policy loops remain real."""
    model = PolarModel(PolarCode(3, 8, 4, 128), np.arange(32).reshape(4, 8), np.full((4, 8), 1 / np.sqrt(8)), 32)
    scenario = PolarScenario(model, np.r_[np.full(8, 10.), np.zeros(24)], 3, .1)
    states = [Counter(), Counter({0: 1}), Counter({0: 1, 1: 1})]
    states += [Counter({1: 1, 2: 1}) if mode == "swap" else Counter({1: 1}), Counter({1: 1, 2: 1, 3: 1})]
    if mode == "no_action":
        states = [Counter(), Counter()]
    elif mode == "third_no_action":
        states[3] = states[2].copy()
    lookup = {}
    for phase, action in enumerate(states):
        raw = scenario.y.copy(); raw[:8] += phase
        if mode == "no_action":
            raw = scenario.y.copy()
        llr, _ = cavity_query(scenario, action, raw, 0)
        lookup[llr.tobytes()] = bits(phase, 3)[None, :]
    if mode == "no_action":
        lookup[llr.tobytes()] = bits(0, 3)[None, :]

    class ScriptedCode(PolarCode):
        def decode_list(self, llr):
            if not np.any(llr):
                return np.empty((0, 3), dtype=np.uint8), np.empty(0)
            return lookup[llr.tobytes()].copy(), np.zeros(1)

    class ScriptedBank:
        def __init__(self, current, ids):
            self.words = sorted(ids)
            self.last_raw_residual = current.y.copy()
            self.matrix = csc_matrix((32, 0))
            self.construction_seconds = 0.
            self.refit_calls = 0

        def extend(self, words):
            fresh = sorted(set(words) - set(self.words))
            self.words = sorted(set(self.words).union(words))
            return fresh, 0.

        def refit(self, initial, max_moves):
            phase = self.refit_calls
            self.refit_calls += 1
            action = states[min(phase, len(states) - 1)].copy()
            changed = action != initial
            self.last_raw_residual = scenario.y.copy()
            if mode != "no_action":
                self.last_raw_residual[:8] += 2 if mode == "third_no_action" and phase == 3 else phase
            moves = [{"kind": "swap" if mode == "swap" and phase == 3 else "drop" if phase == 3 else "add",
                      "before": sorted(initial.items()), "after": sorted(action.items())}] if changed else []
            return action, {"converged": mode != "incomplete", "moves": moves, "seconds": 0.}

    return scenario, ScriptedBank, ScriptedCode


class ProgressContinuationTest(unittest.TestCase):
    def execute(self, mode, enabled):
        scenario, bank, code = scripted(mode)
        with patch("framework.polar_count_feedback.CountBank", bank):
            return continue_feedback(scenario, Counter(), set(), code_factory=code, progress_extra_sweep=enabled)

    def test_swap_and_drop_trigger_one_block_without_net_count_growth(self):
        for mode in ("swap", "drop"):
            old_action, old_bank, old = self.execute(mode, False)
            action, bank, new = self.execute(mode, True)
            self.assertEqual(len(old["sweeps"]), 3)
            self.assertEqual(len(new["sweeps"]), 4)
            self.assertEqual(semantic(old["queries"]), semantic(new["queries"][:12]))
            self.assertEqual(semantic(old["refits"]), semantic(new["refits"][:3]))
            self.assertEqual(old["sweeps"], new["sweeps"][:3])
            self.assertEqual(new["extra_sweep_initial_action"], sorted(old_action.items()))
            self.assertTrue(new["extra_sweep_triggered"])
            self.assertEqual(new["third_sweep_count_moves"], 1)
            self.assertLessEqual(old["sweeps"][2]["output_count"], old["sweeps"][1]["output_count"])
            self.assertEqual((old["actual_scl_calls"], new["actual_scl_calls"]), (4, 5))
            self.assertEqual((len(old_bank), len(bank)), (3, 4))
            self.assertEqual(action.total(), 3)
            self.assertEqual(len(new["count_action_path"]), 4)
            self.assertEqual(new["count_action_path"][2]["before"], [(0, 1), (1, 1)])
            self.assertEqual(new["count_action_path"][2]["after"], sorted(old_action.items()))

    def test_new_zero_count_bank_word_does_not_enable_extra_block(self):
        _, _, meta = self.execute("third_no_action", True)
        self.assertEqual(len(meta["sweeps"]), 3)
        self.assertEqual(meta["sweeps"][2]["new_ids"], 1)
        self.assertEqual(meta["third_sweep_count_moves"], 0)
        self.assertFalse(meta["extra_sweep_triggered"])
        self.assertEqual(meta["stop"], "third_sweep_no_count_moves")

    def test_no_action_block_absorbs_even_if_bank_grew(self):
        action, bank, meta = self.execute("no_action", True)
        self.assertEqual(action, Counter())
        self.assertEqual(bank, {0})
        self.assertEqual([row["new_ids"] for row in meta["sweeps"]], [1, 0])
        self.assertEqual([row["moves"] for row in meta["sweeps"]], [0, 0])
        self.assertEqual(len(meta["refits"]), 1)
        self.assertEqual(meta["actual_scl_calls"], 2)
        self.assertTrue(all(query["cache_hit"] for query in meta["queries"][4:]))
        self.assertEqual(meta["stop"], "full_sweep_no_new_ids")
        self.assertFalse(meta["extra_sweep_triggered"])

    def test_incomplete_initial_fit_never_queries(self):
        _, _, meta = self.execute("incomplete", True)
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["queries"], [])
        self.assertEqual(meta["count_action_path"], [])
        self.assertFalse(meta["extra_sweep_triggered"])

    def test_default_false_matches_original_policy_and_real_repeated_wrapper(self):
        scenario, truth = smoke_scenario()
        old = continue_feedback(scenario, truth, set(truth))
        for options in ({}, {"progress_extra_sweep": False}):
            current = continue_feedback(scenario, truth, set(truth), **options)
            self.assertEqual(current[:2], old[:2])
            self.assertEqual(semantic(current[2]), semantic(old[2]))
            self.assertNotIn("progress_extra_sweep", current[2])
        prior = continue_cavity(scenario, truth, set(truth), mode="raw128")
        self.assertEqual(old[:2], prior[:2])
        for key in ("initial_refit_action", "sweeps", "refits", "complete", "stop", "sweep_cap_with_new_ids"):
            self.assertEqual(semantic(old[2][key]), semantic(prior[2][key]))
        action, meta = run(scenario, memoize=True, complete_support=True, progress_extra_sweep=True)
        self.assertEqual(Counter(action), truth)
        self.assertFalse(meta["continuation"]["extra_sweep_triggered"])
        self.assertTrue(meta["complete"])
        with self.assertRaises(ValueError):
            continue_feedback(scenario, truth, set(truth), progress_extra_sweep=1)


if __name__ == "__main__":
    unittest.main()
