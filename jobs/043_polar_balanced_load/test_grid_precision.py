"""Threshold uncertainty and local bracket decisions on exact finite fixtures."""
import unittest

import numpy as np

from grid_precision import grid_bands
from preview12 import midpoints


class GridPrecisionTest(unittest.TestCase):
    def test_known_bins_even_with_zero_frame_variance(self):
        grid = [0,25,50]
        counts = np.tile([[[8,7],[2,0],[0,0]]], (12,1,1))
        result = grid_bands(100,grid,counts)
        for item in result['arms'].values():
            self.assertEqual(item['observed_threshold_bounds_db'],[0,.25])
            self.assertEqual(item['grid_bootstrap_95_db'],[0,.25])
        self.assertEqual(result['dec38_minus_dec39_grid_bootstrap_95_db'],[-.25,.25])

    def test_censor_and_recrossing_bands(self):
        counts = np.tile([[[10,7],[11,1],[10,8]]], (12,1,1))
        result = grid_bands(100,[0,10,20],counts)
        self.assertEqual(result['arms']['DEC-38']['grid_bootstrap_95_db'],[.2,None])
        self.assertEqual(result['arms']['DEC-39']['grid_bootstrap_95_db'],[None,None])
        self.assertEqual(result['dec38_minus_dec39_grid_bootstrap_95_db'],[None,None])

    def test_refine_both_arms_and_stop_at_declared_spacing(self):
        self.assertEqual(midpoints([-50,0,50],np.array([[50,40],[30,10],[10,1]]),400),[-25,25])
        self.assertEqual(midpoints([0,5],np.array([[30,31],[10,11]]),400),[])
        self.assertEqual(midpoints([-20,0,20],np.array([[50,50],[20,20],[1,0]]),400),[-5,5])
