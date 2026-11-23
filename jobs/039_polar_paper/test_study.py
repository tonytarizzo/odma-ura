"""Submission coverage, atomic resume records and paper export preflight."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_study as runner
from paper_controls import SEEDS
from paper_outputs import NAMES, write_tables_and_figure


class StudyTest(unittest.TestCase):
    def test_pbs_manifest(self):
        grid = runner.shards()
        spec = (runner.JOB / "039_paper.sh").read_text()
        self.assertIn(f"#PBS -J 1-{len(grid)}\n", spec)
        keys = [key for batch in grid for key in batch]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual({key[2] for key in keys if key[0]=="control"}, set(SEEDS))
        self.assertEqual(len(keys), 6784)
        self.assertTrue(set(SEEDS).isdisjoint(runner.CURVE_SEEDS[100]))

    def test_resume_and_incomplete_merge(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            grid = [[("curve",100,1,45), ("curve",100,2,45)]]
            rows = [dict(seed=i,k=100,snr_cdb=45) for i in (1,2)]
            with patch.object(runner, "run_curve_frame", side_effect=[rows[0], RuntimeError("interrupted")]):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    runner.run_shard(1,grid,root,"abc")
            with self.assertRaisesRegex(ValueError,"Missing 1"):
                runner.merge(root,grid,"abc")
            with patch.object(runner, "run_curve_frame", return_value=rows[1]) as mock:
                runner.run_shard(1,grid,root,"abc")
                self.assertEqual(mock.call_count,1)
            path = runner.path_for(root,grid[0][1])
            self.assertFalse(runner.save_once(path,{}))
            self.assertEqual(runner.checked(path,grid[0][1],"abc"),rows[1])
            with self.assertRaisesRegex(ValueError,"Mismatched"):
                runner.checked(path,grid[0][1],"wrong")

    def test_performance_exports(self):
        arm = dict(list_misses=1,count_misses=1,false_copies=0,list_size=50,output_copies=50)
        for name in ("list_pupe","count_pupe"):
            arm.update({name:.01,name+"_pointwise_95":[0,.03]})
        for key in ("seconds_with_geometry","receiver_seconds","native_scl128","rollback_scl512","continuation_scl128"):
            arm[key] = dict(mean=1,p50=1,p95=2,maximum=3)
        endpoints = {a:arm for a in NAMES}
        load = dict(snr_cdb=[0,50], frames_per_snr=2, users_per_snr=100,
                    by_snr={s:dict(endpoints=endpoints) for s in ("0","50")},
                    crossings={"list_pupe":{"thresholds":{a:{"snr_db":None} for a in NAMES}}})
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            write_tables_and_figure(dict(loads={"50":load,"100":load}),root)
            with (root/"performance_cost.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),12)
            self.assertTrue((root/"list_pupe.pdf").read_bytes().startswith(b"%PDF"))
