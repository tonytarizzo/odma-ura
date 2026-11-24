"""Finite checks for the balanced-encoder threshold and its paired uncertainty."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import BOOTSTRAPS, analyze


class BalancedThresholdTest(unittest.TestCase):
    def test_observed_brackets_and_signed_difference(self):
        x = np.tile([[8,7],[6,5],[4,3]], (8,1,1))
        r = analyze(100, [-50,0,50], x)
        a, b = r['arms']['DEC-38'], r['arms']['DEC-39']
        self.assertEqual((a['status'], a['snr_db'], a['observed_grid_bracket_db']), ('interpolated',.25,[0.,.5]))
        self.assertEqual((b['status'], b['snr_db'], b['observed_grid_bracket_db']), ('at_grid',0.,[0.,0.]))
        delta = r['contrast_38_minus_39']
        self.assertEqual(delta['snr_db_saved_by_39'], .25)
        self.assertEqual(delta['paired_bootstrap_95_db'], [.25,.25])
        self.assertEqual(delta['pupe_saved_by_39'], [.01]*3)
        self.assertEqual(delta['pointwise_95_pupe_saved_by_39'], [[.01,.01]]*3)
        self.assertEqual(delta['improved_frames_by_39'], [8]*3)
        self.assertEqual(delta['both_crossings_observed_resamples'], BOOTSTRAPS)
        json.dumps(r, allow_nan=False)

    def test_open_ended_required_energy_and_recrossing(self):
        def check(x):
            return analyze(100, [-50,0,50], np.tile(x, (4,1,1)))
        r = check([[9,4],[7,2],[6,3]])
        self.assertEqual(r['arms']['DEC-38']['paired_bootstrap_95_db'], [.5,None])
        self.assertEqual(r['arms']['DEC-39']['paired_bootstrap_95_db'], [None,-.5])
        self.assertEqual(r['contrast_38_minus_39']['censor_bounds_db'], [1.,None])
        self.assertIsNone(r['contrast_38_minus_39']['snr_db_saved_by_39'])
        r = check([[6,8],[4,6],[6,4]])
        a = r['arms']['DEC-38']
        self.assertEqual(a['status'], 'ambiguous')
        self.assertIsNone(a['observed_grid_bracket_db'])
        self.assertEqual(a['paired_bootstrap_95_db'], [None,None])
        self.assertEqual(a['bootstrap_status'], {'ambiguous': BOOTSTRAPS})
        self.assertEqual(r['contrast_38_minus_39']['paired_bootstrap_95_db'], [None,None])
        json.dumps(r, allow_nan=False)

    def test_frame_pairing_and_inclusion_of_censored_resamples(self):
        values = [[9,6,1], [7,2,1], [10,8,7], [9,8,7]]
        x = np.repeat(np.asarray(values)[:, :, None], 2, axis=2)
        r = analyze(100, [-50,0,50], x)
        d = r['contrast_38_minus_39']
        self.assertEqual(d['snr_db_saved_by_39'], 0.)
        self.assertEqual(d['pointwise_95_pupe_saved_by_39'], [[0.,0.]] * 3)
        self.assertEqual(r['arms']['DEC-38']['snr_db'], r['arms']['DEC-39']['snr_db'])
        self.assertLess(d['both_crossings_observed_resamples'], BOOTSTRAPS)
        self.assertGreater(r['arms']['DEC-38']['bootstrap_status']['right_censored'], BOOTSTRAPS*.025)
        self.assertEqual(r['arms']['DEC-38']['paired_bootstrap_95_db'][1], None)
        self.assertEqual(d['paired_bootstrap_95_db'], [None,None])

    def test_reject_partial_or_invalid_paired_data(self):
        x = np.full((2,3,2), 3)
        for k, db, a in ((100,[0,0,1],x), (100,[1,0,2],x), (100,[0.,1.,2.],x), (0,[0,1,2],x), (2,[0,1,2],x),
                         (100,[0,1,2],-x), (100,[0,1,2],x[0]), (100,[0,1,2],x.astype(float)), (100,[0,1,2],x[:1])):
            with self.subTest(k=k, db=db, shape=a.shape):
                with self.assertRaises(ValueError):
                    analyze(k,db,a)


if __name__ == '__main__':
    unittest.main()
