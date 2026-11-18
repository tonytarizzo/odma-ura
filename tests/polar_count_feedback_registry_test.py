"""DEC-35's frozen default and explicit query-reuse/terminal-decision variants."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from framework.polar_count_cavity import run as direct
from framework.polar_decoders import REGISTRY, Scenario, run
from framework.polar_tractable import PolarScenario
from tests.polar_count_cavity_test import small_model, smoke_scenario


class CountFeedbackRegistryTest(unittest.TestCase):
    def test_registered_operation_matches_complete_repeated_message_fixture(self):
        scenario, truth = smoke_scenario()
        registered, meta = run("DEC-35", scenario)
        original, old = direct(scenario, mode="raw128")
        self.assertEqual(Counter(registered), truth)
        self.assertEqual(registered, original)
        self.assertEqual(meta["rollback_action"], old["rollback_action"])
        self.assertEqual(meta["base_bank"], old["base_bank"])
        self.assertEqual(meta["bank"], old["bank"])
        for key in ("mode", "list_size", "query_budget_per_sweep", "initial_refit_action", "complete", "stop"):
            self.assertEqual(meta["continuation"][key], old["continuation"][key])
        for left, right in zip(meta["continuation"]["queries"], old["continuation"]["queries"], strict=True):
            for key in ("pattern", "pre_action_sha256", "llr_sha256", "ids", "new_ids"):
                self.assertEqual(left[key], right[key])
        self.assertFalse(meta["legacy_dense_adapter"])
        self.assertTrue(REGISTRY["DEC-35"].partial_counts)
        self.assertEqual(meta["dec35_options"], {"memoize": False, "complete_support": False})
        self.assertEqual(dict(REGISTRY["DEC-35"].defaults),
                         {"max_sweeps": 3, "max_moves": 2000, "memoize": False, "complete_support": False,
                          "native_rate_zero": False, "native_repetition": False, "all_stage_repetition": False,
                          "progress_extra_sweep": False, "count_growth": False})

    def test_explicit_variants_preserve_complete_repeated_message_trajectory(self):
        scenario, truth = smoke_scenario()
        original, old = direct(scenario, mode="raw128")
        query_keys = ("query", "sweep", "stage", "pattern", "list_size", "context", "restored", "restored_count",
                      "outside_count", "pending", "variance", "pre_action_sha256", "llr_sha256", "ids", "new_ids", "refit_index")
        old_queries = [{key: row[key] for key in query_keys} for row in old["continuation"]["queries"]]
        for memoize, complete_support in ((True, False), (False, True), (True, True)):
            with self.subTest(memoize=memoize, complete_support=complete_support):
                counts, meta = run("DEC-35", scenario, memoize=memoize, complete_support=complete_support)
                self.assertEqual(counts, original)
                self.assertEqual(Counter(counts), truth)
                self.assertEqual(Counter(dict(meta["cancellation_action"])), truth)
                self.assertEqual(meta["bank"], old["bank"])
                self.assertEqual(meta["base_bank"], old["base_bank"])
                self.assertEqual(meta["rollback_action"], old["rollback_action"])
                self.assertEqual(meta["continuation"]["sweeps"], old["continuation"]["sweeps"])
                self.assertEqual(meta["continuation"]["stop"], old["continuation"]["stop"])
                self.assertEqual(meta["continuation"]["memoized"], memoize)
                self.assertEqual(meta["dec35_options"], {"memoize": memoize, "complete_support": complete_support})
                self.assertEqual([{key: row[key] for key in query_keys} for row in meta["continuation"]["queries"]], old_queries)
                self.assertEqual([row["meta"]["moves"] for row in meta["continuation"]["refits"]],
                                 [row["meta"]["moves"] for row in old["continuation"]["refits"]])
                self.assertTrue(meta["complete"])
                if complete_support:
                    self.assertEqual(meta["support_completion"]["additions"], [])
                else:
                    self.assertIsNone(meta["support_completion"])

    def test_variant_dispatch_and_terminal_output_are_separate_from_cancellation(self):
        scenario, _ = smoke_scenario()
        for memoize, complete_support in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(memoize=memoize, complete_support=complete_support):
                with patch("framework.polar_count_cavity.run", return_value=({209: 2}, {})) as frozen:
                    with patch("framework.polar_count_feedback.run", return_value=({209: 2}, {})) as optional:
                        run("DEC-35", scenario, memoize=memoize, complete_support=complete_support)
                        if memoize or complete_support:
                            frozen.assert_not_called()
                            optional.assert_called_once_with(scenario, memoize=memoize, complete_support=complete_support)
                        else:
                            optional.assert_not_called()
                            frozen.assert_called_once_with(scenario, mode="raw128", max_sweeps=3, max_moves=2000)
        cancellation, bank = Counter({209: 2}), {209, 1258}
        with patch("framework.polar_count_feedback.continue_feedback", return_value=(cancellation, bank, {"complete": True})):
            counts, meta = run("DEC-35", scenario, complete_support=True)
        self.assertEqual(counts, {209: 2, 1258: 1})
        self.assertEqual(meta["cancellation_action"], [(209, 2)])
        self.assertEqual(meta["support_completion"]["additions"], [(1258, 1)])
        self.assertEqual(cancellation, Counter({209: 2}))
        self.assertEqual(bank, {209, 1258})

    def test_variant_options_require_explicit_bools_and_fixed_budgets(self):
        scenario, _ = smoke_scenario()
        with patch("framework.polar_count_cavity.run") as frozen, patch("framework.polar_count_feedback.run") as optional:
            for key in ("memoize", "complete_support", "native_rate_zero", "native_repetition", "all_stage_repetition",
                        "progress_extra_sweep"):
                for value in (0, 1, "false", "true", None, np.bool_(True)):
                    with self.subTest(key=key, value=value):
                        with self.assertRaisesRegex(ValueError, "must be bool"):
                            run("DEC-35", scenario, **{key: value})
            for options in ({}, {"memoize": True}, {"complete_support": True}, {"memoize": True, "complete_support": True},
                            {"memoize": True, "complete_support": True, "native_repetition": True, "progress_extra_sweep": True}):
                for budget in ({"max_sweeps": 2}, {"max_sweeps": 4}, {"max_moves": 1999}, {"max_moves": 2001}):
                    with self.subTest(options=options, budget=budget):
                        with self.assertRaisesRegex(ValueError, "three sweeps and 2000 moves"):
                            run("DEC-35", scenario, **options, **budget)
            frozen.assert_not_called()
            optional.assert_not_called()

    def test_native_rate_zero_registered_variant_matches_the_verified_provider(self):
        from framework.polar_native_rate_zero import run as native_only
        scenario, truth = smoke_scenario()
        original_code = scenario.model.code
        expected, old = native_only(scenario)
        counts, meta = run("DEC-35", scenario, memoize=True, complete_support=True, native_rate_zero=True)
        self.assertEqual(counts, expected)
        self.assertEqual(Counter(counts), truth)
        self.assertIs(scenario.model.code, original_code)
        self.assertEqual(meta["dec35_options"], {"memoize": True, "complete_support": True, "native_rate_zero": True})
        for key in ("variant", "bank", "cancellation_action", "rollback_action", "provider_scope"):
            self.assertEqual(meta[key], old[key])
        for memoize, complete_support in ((False, False), (True, False), (False, True)):
            with self.assertRaisesRegex(ValueError, "requires memoize=True and complete_support=True"):
                run("DEC-35", scenario, memoize=memoize, complete_support=complete_support, native_rate_zero=True)

    def test_native_repetition_registered_variant_matches_the_verified_provider(self):
        from framework.polar_native_repetition import run as native_only
        scenario, truth = smoke_scenario()
        original_code = scenario.model.code
        expected, old = native_only(scenario)
        counts, meta = run("DEC-35", scenario, memoize=True, complete_support=True, native_repetition=True)
        self.assertEqual(counts, expected)
        self.assertEqual(Counter(counts), truth)
        self.assertIs(scenario.model.code, original_code)
        self.assertEqual(meta["dec35_options"], {"memoize": True, "complete_support": True, "native_repetition": True})
        for key in ("variant", "bank", "cancellation_action", "rollback_action", "provider_scope"):
            self.assertEqual(meta[key], old[key])
        for memoize, complete_support in ((False, False), (True, False), (False, True)):
            with self.assertRaisesRegex(ValueError, "requires memoize=True and complete_support=True"):
                run("DEC-35", scenario, memoize=memoize, complete_support=complete_support, native_repetition=True)
        with self.assertRaisesRegex(ValueError, "Select one native provider"):
            run("DEC-35", scenario, memoize=True, complete_support=True, native_rate_zero=True, native_repetition=True)

    def test_all_stage_repetition_registered_variant_matches_verified_provider(self):
        from framework.polar_all_stage_repetition import run as all_stages
        scenario, truth = smoke_scenario()
        original_code = scenario.model.code
        expected, old = all_stages(scenario)
        counts, meta = run("DEC-35", scenario, memoize=True, complete_support=True, all_stage_repetition=True)
        self.assertEqual(counts, expected)
        self.assertEqual(Counter(counts), truth)
        self.assertIs(scenario.model.code, original_code)
        self.assertEqual(meta["dec35_options"], {"memoize": True, "complete_support": True, "all_stage_repetition": True})
        for key in ("variant", "bank", "cancellation_action", "rollback_action", "provider_scope", "query_provider"):
            self.assertEqual(meta[key], old[key])
        for memoize, complete_support in ((False, False), (True, False), (False, True)):
            with self.assertRaisesRegex(ValueError, "requires memoize=True and complete_support=True"):
                run("DEC-35", scenario, memoize=memoize, complete_support=complete_support, all_stage_repetition=True)
        for conflicting in ("native_repetition", "native_rate_zero"):
            with self.assertRaisesRegex(ValueError, "Select one"):
                run("DEC-35", scenario, memoize=True, complete_support=True, all_stage_repetition=True, **{conflicting: True})

    def test_conditional_fourth_registered_variant_matches_verified_complete_provider(self):
        from framework.polar_progress_continuation import run as progress_receiver
        scenario, truth = smoke_scenario()
        original_model, original_code, original_y = scenario.model, scenario.model.code, scenario.y.copy()
        expected, old = progress_receiver(scenario)
        counts, meta = run("DEC-35", scenario, memoize=True, complete_support=True, native_repetition=True,
                           progress_extra_sweep=True)
        self.assertEqual(counts, expected)
        self.assertEqual(Counter(counts), truth)
        self.assertIs(scenario.model, original_model)
        self.assertIs(scenario.model.code, original_code)
        np.testing.assert_array_equal(scenario.y, original_y)
        self.assertEqual(meta["dec35_options"], {"memoize": True, "complete_support": True, "native_repetition": True,
                                                "progress_extra_sweep": True})
        for key in ("variant", "bank", "base_bank", "cancellation_action", "rollback_action", "provider_scope"):
            self.assertEqual(meta[key], old[key])
        self.assertEqual(Counter(dict(meta["cancellation_action"])), truth)
        self.assertEqual(meta["support_completion"]["additions"], old["support_completion"]["additions"])
        query_keys = ("query", "sweep", "pattern", "list_size", "variance", "pre_action_sha256", "llr_sha256",
                      "ids", "new_ids", "refit_index", "cache_hit")
        for left, right in zip(meta["continuation"]["queries"], old["continuation"]["queries"], strict=True):
            self.assertEqual({key: left[key] for key in query_keys}, {key: right[key] for key in query_keys})
        for key in ("sweeps", "stop", "progress_extra_sweep", "maximum_sweeps", "extra_sweep_triggered",
                    "third_sweep_count_moves", "extra_sweep_initial_action", "count_action_path"):
            self.assertEqual(meta["continuation"][key], old["continuation"][key])

    def test_conditional_fourth_dispatch_is_explicit_and_rejects_unverified_profiles(self):
        scenario, _ = smoke_scenario()
        with patch("framework.polar_progress_continuation.run", return_value=({209: 2}, {})) as progress:
            with patch("framework.polar_native_repetition.run", return_value=({209: 2}, {})) as native:
                run("DEC-35", scenario, memoize=True, complete_support=True, native_repetition=True,
                    progress_extra_sweep=True)
                progress.assert_called_once_with(scenario)
                native.assert_not_called()
                progress.reset_mock()
                run("DEC-35", scenario, memoize=True, complete_support=True, native_repetition=True,
                    progress_extra_sweep=False)
                progress.assert_not_called()
                native.assert_called_once_with(scenario)
        unverified = [{}, {"memoize": True, "complete_support": True},
                      {"memoize": True, "complete_support": True, "native_rate_zero": True},
                      {"memoize": True, "complete_support": True, "all_stage_repetition": True},
                      {"native_repetition": True}, {"native_repetition": True, "memoize": True},
                      {"native_repetition": True, "complete_support": True}]
        with patch("framework.polar_progress_continuation.run") as receiver:
            for options in unverified:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(ValueError, "progress_extra_sweep variant requires native_repetition=True"):
                        run("DEC-35", scenario, progress_extra_sweep=True, **options)
            receiver.assert_not_called()

    def test_registered_count_growth_uses_the_verified_receiver(self):
        from framework.polar_count_growth import run as count_growth
        scenario, truth = smoke_scenario()
        options = dict(memoize=True, complete_support=True, native_repetition=True, progress_extra_sweep=True, count_growth=True)
        counts, meta = run("DEC-35", scenario, **options)
        expected, direct_meta = count_growth(scenario)
        self.assertEqual(counts, expected)
        self.assertEqual(Counter(counts), truth)
        self.assertEqual(meta["dec35_options"], options)
        self.assertTrue(meta["continuation"]["count_growth"])
        for key in ("variant", "cancellation_action", "bank", "rollback_action", "base_bank"):
            self.assertEqual(meta[key], direct_meta[key])
        for key in options.keys() - {"count_growth"}:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                run("DEC-35", scenario, **{**options, key: False})
        with self.assertRaises(ValueError):
            run("DEC-35", scenario, **{**options, "count_growth": 1})

    def test_partial_counts_are_explicit_and_do_not_relax_other_decoders(self):
        scenario, _ = smoke_scenario()
        for counts in ({}, {209: 1}, {209: 2, 1258: 1}):
            with patch("framework.polar_count_cavity.run", return_value=(counts, {})) as receiver:
                got, _ = run("DEC-35", scenario)
                self.assertEqual(got, counts)
                receiver.assert_called_once_with(scenario, mode="raw128", max_sweeps=3, max_moves=2000)
        for counts in ({209: 4}, {209: 0}, {209: -1}, {209: .5}, {1 << scenario.model.B: 1}):
            with patch("framework.polar_count_cavity.run", return_value=(counts, {})):
                with self.assertRaisesRegex(ValueError, "output-budget contract"):
                    run("DEC-35", scenario)
        self.assertFalse(REGISTRY["DEC-25"].partial_counts)
        explicit = Scenario(np.eye(2), np.zeros(2), 1, 1.)
        incomplete = replace(REGISTRY["DEC-14"], run=lambda *args, **kwargs: ({}, {}))
        with patch.dict(REGISTRY, {"DEC-14": incomplete}):
            with self.assertRaisesRegex(ValueError, "output-budget contract"):
                run("DEC-14", explicit)

    def test_requires_procedural_geometry_and_frozen_native_budget(self):
        explicit = Scenario(np.eye(2), np.zeros(2), 1, 1.)
        with self.assertRaisesRegex(ValueError, "procedural PolarScenario"):
            run("DEC-35", explicit)
        with self.assertRaisesRegex(ValueError, "supports"):
            run("DEC-35", Scenario(explicit.phi, explicit.y, 1, 1., "distinct"))
        small = small_model()
        with self.assertRaisesRegex(ValueError, "at least 32 patterns"):
            run("DEC-35", PolarScenario(small, np.zeros(small.n), 1, 1.))
        scenario, _ = smoke_scenario()
        scenario.model.code.list_size = 16
        with self.assertRaisesRegex(ValueError, "native SCL128"):
            run("DEC-35", scenario)
        scenario.model.code.list_size = 128
        with self.assertRaisesRegex(ValueError, "three sweeps"):
            run("DEC-35", scenario, max_sweeps=4)


if __name__ == "__main__":
    unittest.main()
