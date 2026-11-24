"""Regression checks for the adaptive search, resumable receipts and independent confirmation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from importlib import import_module
job = import_module('jobs.043_polar_balanced_load.load_curve') if __package__ else import_module('load_curve')


class BalancedLoadSearchTest(unittest.TestCase):
    def test_pilot_midpoints_use_both_receivers(self):
        grid = [-100, 0, 100, 200]
        totals = np.array([[32, 40], [12, 40], [2, 25], [1, 0]])
        self.assertEqual(job.choose_grid(grid, totals, 200, False), [50, 150])
        valid = job.choose_grid(grid, totals, 200, True)
        self.assertTrue({0,100,200}.issubset(valid))
        self.assertGreaterEqual(len(valid),5)

    def test_open_limits_and_recrossing_witnesses(self):
        grid = [-100,0,100]
        ambiguous = np.array([[0,20],[30,20],[60,50]])
        self.assertEqual(job.choose_grid(grid, ambiguous, 100, False), [job.LIMITS[1]])
        with_limit = job.choose_grid([*grid, job.LIMITS[1]], np.vstack((ambiguous, [65,80])), 100, True)
        self.assertTrue({-100,job.LIMITS[1]}.issubset(with_limit))
        low = np.zeros((3,2),int)
        self.assertEqual(job.choose_grid(grid, low, 100, False), [job.LIMITS[0]])
        self.assertEqual(job.choose_grid([job.LIMITS[0],0,100],low,100,False),[])
        self.assertTrue(set(job.choose_grid([job.LIMITS[0],0,100],low,100,True)) <= set(range(-300,0)))

    def test_task_coverage_and_separation(self):
        used = set()
        for stage, loads, total in (('pilot',job.LOADS,48),('confirm',job.LOADS,768),('preview',job.PREVIEW,8)):
            sharded = [(k, s) for i in range(1,total+1) for k, sample in [job.shard(i, stage)] for s in sample]
            wanted = [(k, s) for k in loads for s in job.seeds(k, stage)]
            self.assertCountEqual(sharded, wanted)
            ids = {s for k, s in sharded}
            self.assertEqual(len(ids),len(sharded))
            self.assertFalse(used & ids)
            used.update(ids)
            with self.assertRaises(ValueError):
                job.shard(total+1,stage)

    def test_final_plan_replays_independent_of_confirm_outcomes(self):
        k=50
        def collect(out,revision,k,panel,energies):
            self.assertEqual(panel,'pilot')
            misses=np.array([[[5 if db < cutoff else 0 for cutoff in (0, 60)] for db in energies]]*job.PILOT_FRAMES)
            return {'count_misses':misses}, None
        with tempfile.TemporaryDirectory() as tmp, patch.object(job,'collect',side_effect=collect):
            out=Path(tmp)
            initial=job.plan(out,'test',k,4)
            self.assertEqual(initial,job.plan(out,'test',k,4,recompute=True))
            path=out/'plans/k50_stage4.json'
            stale=json.loads(path.read_text())
            stale['energies_cdb'].append(867)
            path.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError,'differs'):
                job.plan(out,'test',k,4,recompute=True)


if __name__ == '__main__':
    unittest.main()
