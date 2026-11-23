"""Finite and stored-record contracts for the equal-search paper controls."""

from collections import Counter
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"jobs/039_polar_paper"))
import paper_controls as controls
from baselines.polar import PolarCode
from framework.polar_count_feedback import continue_feedback, support_completion
from framework.polar_tractable import PolarModel, PolarScenario
from tests.polar_count_cavity_test import smoke_scenario
from tests.polar_full_pattern_test import fixture, states


class ControlTest(unittest.TestCase):
    def test_add_gains_match_every_objective_increment(self):
        model, atoms = fixture()
        for y in (2*atoms[0]+atoms[2], np.zeros(8), np.linspace(-1.1,1.7,8)):
            bank = controls.AddOnlyBank(PolarScenario(model,y,3,.13),[0,2,4])
            for counts in states(3,3):
                h = 3-int(counts.sum())
                res = y-bank.matrix@counts-h*bank.mean
                actual, work = bank.gains(counts,res,{})
                increments = [bank.objective(counts+v)-bank.objective(counts) for v in np.eye(3,dtype=int)] if h else []
                self.assertAlmostEqual(actual[0],max([0.,*increments]),places=10)
                self.assertIn(actual[1],("none","add"))
                self.assertEqual(work["gram_rows_constructed"],0)
            action, _ = bank.refit(Counter({4:3}))
            self.assertEqual(action,Counter({4:3}))

    def test_common_prefix_and_equal_additional_work(self):
        s, truth = smoke_scenario()
        out, reference = controls.registered("DEC-35",s,**controls.OPTIONS)
        initial, ids, seen, prefix = controls.prefix(s,reference)
        self.assertEqual(prefix["calls512"],32)
        for rule in controls.RULES:
            result = controls.search(s,initial,ids,seen,prefix["seconds"],.125,rule,10,
                                     target=0. if rule=="conventional" else None,reference_calls=3 if rule=="conventional" else None)
            end = result["fixed"]
            self.assertEqual(end["work"]["actual_scl_calls"],10)
            self.assertAlmostEqual(end["geometry_inclusive_seconds"]-end["receiver_seconds"],.125)
            self.assertGreater(end["work"]["prior512_requeries"],0)
            if rule=="conventional":
                self.assertIsNone(result["bracket"]["under"])
                self.assertEqual(result["bracket"]["over"]["work"]["actual_scl_calls"],0)
                self.assertEqual(result["matched_registered_calls"]["work"]["actual_scl_calls"],3)
        replay = controls.replay(s,reference,out,list(truth.elements()))
        self.assertEqual(replay["opportunities"],0)
        self.assertTrue(all(q["live"]["ids"]==q["frozen"]["ids"] for q in replay["queries"]))

    def test_nontrivial_controlled_change_keeps_schedule_variance(self):
        s, truth = smoke_scenario()
        wrong = Counter({3:1,209:1,1258:1})
        ids = set(wrong)
        cancel, bank, hist = continue_feedback(s,wrong,ids,progress_extra_sweep=True,count_growth=True)
        out, _ = support_completion(s,cancel,bank)
        ref = dict(rollback_action=list(wrong.items()),base_bank=sorted(ids),bank=sorted(bank),continuation=hist)
        obs = controls.replay(s,ref,out,list(truth.elements()))
        self.assertEqual(obs["opportunities"],len(hist["queries"]))
        self.assertGreater(obs["changed_inputs"],0)
        for q, baseline in zip(obs["queries"],hist["queries"],strict=True):
            self.assertEqual((q["variance"],q["pattern"]),(baseline["variance"],baseline["pattern"]))
            self.assertIn("swap",q["preceding_refit_move_kinds"])
            if q["changed_input"]:
                self.assertNotEqual(q["live"]["llr"],q["frozen"]["llr"])
        self.assertEqual(dict(obs["arms"]["live"]["action"]),out)

    def test_cross_pattern_cache_reconstructs_ids_and_exhaustion_is_honest(self):
        # Identical supports under different pattern labels yield identical payload input bytes.
        # CRC0 returns a valid complete word on either label.
        p = 4
        model = PolarModel(PolarCode(8,8,0,128), np.tile(np.arange(8),(p,1)),np.full((p,8),8**-.5),8)
        s = PolarScenario(model,np.linspace(.1,.2,8),1,.2)
        full = Counter({0:1})
        result = controls.search(s,full,set(full),{128:set(),512:set()},0.,0.,"conventional",3,target=100.)
        self.assertTrue(result["exhausted"])
        self.assertIsNone(result["fixed"])
        self.assertIsNone(result["bracket"]["over"])
        self.assertEqual(result["cache"]["actual_scl_calls"],1)
        self.assertEqual(result["cache"]["cache_hits"],3)
        self.assertEqual({r["p"] for r in result["log"]},set(range(p)))
        self.assertTrue(all(w>>model.b==q["p"] for q in result["log"] for w in q["ids"]))
        residual = s.y-model.waveform(0)
        expected = 2*model.amplitudes[0]*residual[model.positions[0]]/s.noise_var
        self.assertEqual(result["log"][0]["llr"],hashlib.sha256(expected.tobytes()).hexdigest())

    def test_missing_panel_and_illegal_seed_fail(self):
        with self.assertRaises(ValueError):
            controls.summarize_controls([])
        with self.assertRaises(ValueError):
            controls.run_control_frame(933129)


if __name__=="__main__":
    unittest.main()
