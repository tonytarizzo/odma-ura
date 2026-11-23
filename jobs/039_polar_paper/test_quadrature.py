import unittest
from unittest.mock import patch

import numpy as np

from paper_curve import score
import paper_curve as curve
import paper_quadrature as q
from baselines.polar import PolarCode
from framework.polar_tractable import PolarModel, PolarScenario
import rollback_frames


class QuadratureTest(unittest.TestCase):
    def test_real_limit_and_complex_ranking(self):
        code = PolarCode(3,8,4,8)
        positions = np.arange(16*8).reshape(16,8)
        model = PolarModel(code, positions, np.ones_like(positions)/np.sqrt(8), 128)
        s = PolarScenario(model,np.random.default_rng(73).normal(size=128),4,.4)
        old,meta = rollback_frames.native(s)
        action,rounds,calls = q.native(s,np.zeros(128))
        self.assertEqual((list(action.items()),rounds,calls),(list(old.items()),meta['rounds'],meta['calls']))
        # A purely imaginary disturbance changes the search window; soft decoding still uses the real channel.
        with patch.object(code,'decode_list',wraps=code.decode_list) as decoder:
            imaginary = np.zeros(128)
            imaginary[positions[0]] = 1e4
            q.native(s,imaginary)
            np.testing.assert_array_equal(decoder.call_args_list[0].args[0],
                2*model.amplitudes[0]*s.y[positions[0]]/(s.noise_var+3/model.n))

    def test_paired_summary(self):
        grids = {50:[0,25,50],100:[0,25,50]}
        seeds = {50:[1,2],100:[3,4]}
        rows = []
        for k in grids:
            for seed in seeds[k]:
                for db in grids[k]:
                    labels = list(range(k))
                    action = dict.fromkeys(labels[k//5 if db==0 else 0:],1)
                    m = score(labels,action)
                    rows.append(dict(k=k,seed=seed,snr_cdb=db,labels=labels,endpoints={r:dict(metrics=m) for r in ("native","growth")},
                        complex_screen=dict(ordered=list(action),metrics=m,rounds=[len(action)],calls=k,seconds_with_geometry=10.,
                                            standard_imaginary_sha256=str(seed))))
        with patch.dict(curve.CURVE_DB,grids,clear=True),patch.dict(curve.CURVE_SEEDS,seeds,clear=True),patch.object(q,"BOOTSTRAPS",20):
            result = q.summarize(rows)
            self.assertEqual(result["100"]["contrasts"]["native_complex_minus_growth"]["snr_db_saved_by_second"],0.)
            self.assertEqual(result["100"]["by_snr"]["0"]["complex_count_misses"],40)
            rows[1]["complex_screen"]["standard_imaginary_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError,"Inconsistent"):
                q.summarize(rows)
