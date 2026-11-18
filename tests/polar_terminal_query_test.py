from collections import Counter
import unittest
from unittest.mock import patch

from framework.polar_terminal_query import finish, run
from tests.polar_count_cavity_test import smoke_scenario


class TerminalQueryTest(unittest.TestCase):
    def test_normal_entry_respects_original_fitted_multiplicity(self):
        s, truth = smoke_scenario()
        for mode in ("magnitude","folded"):
            action, meta = run(s, selector=mode)
            self.assertEqual(Counter(action), truth)
            self.assertIsNone(meta["terminal_query"])
            self.assertEqual(meta["terminal_selector"], mode)
            self.assertTrue(meta["complete"])
        with self.assertRaises(ValueError):
            run(s, selector="invalid")

    def test_completion_never_drives_extra_query_and_base_remains_valid(self):
        s, _ = smoke_scenario()
        original = dict(bank=[2,5], cancellation_action=[(2,s.k-1)], complete=True, support_completion={"method":"all_fit"})
        previous = {2:s.k-1,5:1}
        fitted = {2:s.k-1,9:1}
        trace = dict(new_ids=[9],cancellation=sorted(fitted.items()),completion={"method":"no_capacity"},fit={"converged":True})
        for mode in ("magnitude","folded"):
            with patch("framework.polar_terminal_query.terminal_query",return_value=(fitted,trace)) as call:
                actual, meta = finish(s,previous,original,mode)
            call.assert_called_once_with(s,Counter({2:s.k-1}),[2,5],mode)
            self.assertEqual(actual,fitted)
            self.assertEqual(meta["bank"],[2,5,9])
            self.assertEqual(original["bank"],[2,5])
            self.assertEqual(original["cancellation_action"],[(2,s.k-1)])
        unfinished, meta = finish(s,previous,{**original,"complete":False},"magnitude")
        self.assertEqual(unfinished,previous)
        self.assertIsNone(meta["terminal_query"])
