"""Benchmark contract checks for procedural receivers in the explicit small-B harness."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from benchmarks.polar_decoders import normalized_config, run_experiment, sparse_tail_error
from framework.polar_decoders import DecoderSpec, REGISTRY, run


class TractableHarnessTests(unittest.TestCase):
    def test_sparse_tail_diagnostic_includes_missing_words(self):
        reference = np.array([[.8, .3], [.5, .2], [.7, .1]])
        sparse = [[0, [.7, .2]], [1, [.6, .1]]]
        dense = np.array([[.7, .2], [.6, .1], [0., 0.]])
        self.assertAlmostEqual(sparse_tail_error(reference, sparse), np.max(np.abs(reference - dense)))
        self.assertEqual(sparse_tail_error(reference, []), .8)
        self.assertEqual(sparse_tail_error(reference, [[i, row.tolist()] for i, row in enumerate(reference)]), 0.)
        truncated = [[0, [.8]], [1, [.5, .2]], [2, [.7, .1]]]
        self.assertEqual(sparse_tail_error(reference, truncated), .3)

    def test_new_zero_budgets_do_not_relax_existing_contracts(self):
        defaults = (("flip_order", 1), ("burn_in", 2), ("cache_centers", 4), ("samples", 8))
        spec = DecoderSpec("test procedural receiver", object(), ("iid",), defaults, True)
        with patch.dict(REGISTRY, {"DEC-25": spec}):
            for key in ("flip_order", "burn_in", "cache_centers"):
                normalized_config({"decoders": ["DEC-25"], "decoder_params": {"DEC-25": {key: 0}}})
                with self.assertRaisesRegex(ValueError, "Invalid integer budget"):
                    normalized_config({"decoders": ["DEC-25"], "decoder_params": {"DEC-25": {key: -1}}})
            with self.assertRaisesRegex(ValueError, "Invalid integer budget"):
                normalized_config({"decoders": ["DEC-25"], "decoder_params": {"DEC-25": {"samples": 0}}})
            with self.assertRaisesRegex(ValueError, "split the source-specific runs"):
                normalized_config({"decoders": ["DEC-25"], "eval_sampling": ["distinct"]})
        with self.assertRaisesRegex(ValueError, "Invalid integer budget"):
            normalized_config({"decoders": ["DEC-06"], "decoder_params": {"DEC-06": {"burn_in": 0}}})

    def test_model_is_shared_and_sparse_diagnostics_survive_saved_output(self):
        model = object()
        constructor = Mock(return_value=model)
        calls = []

        def decode(name, scenario, **params):
            if name == "REF":
                return run(name, scenario, **params)
            calls.append((name, id(params["model"]), params["seed"]))
            counts = np.zeros(scenario.phi.shape[1], dtype=int)
            counts[0] = scenario.k
            return counts, {"candidate_ids": [0], "tail_probabilities": [[0, [1.] * scenario.k]],
                            "legacy_dense_adapter": True}

        spec = DecoderSpec("test procedural receiver", object(), ("iid",), stochastic=True)
        config = {"decoders": ["REF", "DEC-25", "DEC-26"], "loads": [1, 2], "eval_ebn0": [0.],
                  "eval_frames": 2, "bounds": False, "threads": 1}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.dict(REGISTRY, {"DEC-25": spec, "DEC-26": spec}), \
                patch.dict(sys.modules, {"framework.polar_tractable": SimpleNamespace(PolarModel=constructor)}), \
                patch("benchmarks.polar_decoders.run", side_effect=decode), patch("benchmarks.polar_decoders.plots"), \
                patch("benchmarks.polar_decoders.source_hashes", return_value={"harness_test": "stable"}):
            result = run_experiment(config, Path(folder) / "run")
            records = [json.loads(line) for line in (Path(folder) / "run/frames.jsonl").read_text().splitlines()]
            saved = json.loads((Path(folder) / "run/summary.json").read_text())
        constructor.assert_called_once()
        self.assertEqual(len(calls), 8)
        self.assertEqual({row[1] for row in calls}, {id(model)})
        self.assertEqual(len({row[2] for row in calls}), 8)
        self.assertGreaterEqual(saved["tractable_model_seconds"], 0.)
        self.assertEqual(result["tractable_model_seconds"], saved["tractable_model_seconds"])
        self.assertEqual(len(records), 12)
        for row in records:
            if row["decoder"] == "REF":
                continue
            self.assertIn("conditional_excess_risk", row["meta"])
            self.assertIn("tail_max_error", row["meta"])
            self.assertIn("candidate_recall", row["meta"])
            self.assertTrue(row["meta"]["legacy_dense_adapter"])
            self.assertEqual(row["meta"]["tail_probabilities"], [[0, [1.] * row["K"]]])
        for frame in range(2):
            for k in (1, 2):
                self.assertEqual(len({r["input_sha256"] for r in records if r["frame"] == frame and r["K"] == k}), 1)


if __name__ == "__main__":
    unittest.main()
