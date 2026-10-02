"""Array/merger contracts, including rejection of partial pilot selection."""

import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ura_comparison import DEFAULTS, frame_metrics
from benchmarks.ura_merge import aggregate, load_results, paper_alignment, select_profiles
from baselines import message_bits


class ComparisonManifestTests(unittest.TestCase):
    def test_generated_manifests_and_array_sizes(self):
        folder = Path("jobs/032_published_baselines")
        generated = runpy.run_path(str(folder / "build_manifest.py"))["manifests"]()
        for phase, size in (("pilot", 80), ("comparison", 174), ("native", 36), ("checks", 20)):
            stored = [json.loads(s) for s in (folder / f"{phase}.jsonl").read_text().splitlines()]
            self.assertEqual(generated[phase], stored)
            self.assertEqual(len(stored), size)
            self.assertEqual(len({r["name"] for r in stored}), size)
            script = (folder / f"032_{phase}.sh").read_text()
            self.assertIn(f"#PBS -J 1-{size}", script)
        main = generated["comparison"]
        self.assertEqual(sum(r["decoder"] in {"d2", "d3", "d4"} for r in main), 72)
        self.assertTrue(all(r["max_epochs"] == 200 and r["patience"] == 10 for r in main))
        self.assertTrue(all(r["B"] <= 14 for r in main))
        self.assertFalse(any(r["family"] == "dynamic_cs" for r in main))
        odma = [r for r in main if r["family"] == "odma_polar"]
        self.assertEqual(len(odma), 18)
        self.assertTrue(all(r["baseline_overrides"] == {"max_iterations": 30} for r in odma))
        self.assertTrue(all(r["baseline_overrides"]["max_iterations"] > max(r["loads"]) for r in odma))
        native = generated["native"]
        self.assertTrue(all(r["B"] == (128 if r["family"] == "ccs_amp" else 100) for r in native))
        self.assertTrue(all(r["eval_frames"] == 64 and r["paper_reference"]["target_pupe"] == 0.05 for r in native))
        cells = [(r["family"], r["seed"], r["loads"][0], x) for r in native for x in r["eval_ebn0"]]
        self.assertEqual(len(cells), 84)
        self.assertEqual(len(set(cells)), 84)
        checks = generated["checks"]
        self.assertEqual(sum(r["family"] == "dynamic_cs" for r in checks), 4)
        self.assertTrue(all(r["baseline_params"]["cache_dtype"] == "float64"
                            for r in checks if r["family"] == "dynamic_cs"))

    def test_merger_selection_and_safety_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"name": f"v{i}", "B": 6, "n": 32, "seed": 39, "family": "odma_polar", "mode": "native",
                     "decoder": "native", "baseline_params": {"prefix_bits": i+2}, "loads": [1], "eval_ebn0": [4],
                     "eval_frames": 1, "eval_sampling": ["distinct"]} for i in range(2)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            for i, row in enumerate(rows):
                config = {**DEFAULTS, **row, "loads": [1], "eval_ebn0": [4], "eval_frames": 1, "eval_sampling": ["distinct"]}
                metrics = frame_metrics(message_bits([1], 6), message_bits([i], 6))
                result = {"status": "complete", "config": config, "source_sha256": {"test": "same"},
                          "initial_matrix_sha256": None, "evaluation": [{"sampling": "distinct", "K": 1,
                          "ebn0_db": 4, "frames": 1, "means": metrics, "decoder_seconds_per_frame": 1.0}]}
                (root / row["name"]).mkdir()
                (root / row["name"] / "summary.json").write_text(json.dumps(result))
                if i == 0:
                    with self.assertRaises(ValueError): load_results(manifest, root)
                    partial, audit = load_results(manifest, root, True)
                    with self.assertRaises(ValueError): select_profiles(partial, audit)
            results, audit = load_results(manifest, root)
            selected = select_profiles(results, audit)
            self.assertEqual(selected["profiles"]["B6_n32_odma_polar"]["pilot_name"], "v1")
            self.assertEqual(aggregate([results[1]])[0]["pupe"], 0)
            original = json.dumps(result)
            for key, value in (("baseline_params", {"prefix_bits": 8}), ("loads", [2]), ("eval_frames", 2),
                               ("max_epochs", 1)):
                changed = json.loads(original)
                changed["config"][key] = value
                (root / "v1" / "summary.json").write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Mismatched {key}"):
                    load_results(manifest, root)
            # A source mismatch cannot silently be pooled into one experiment.
            result["source_sha256"]["test"] = "different"
            (root / "v1" / "summary.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError): load_results(manifest, root)

    def test_native_and_learned_selection_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"name": decoder, "B": 6, "n": 32, "seed": 39, "family": "odma_polar", "decoder": decoder,
                     "mode": "native" if decoder == "native" else "fixed", "selected_profile": "B6_n32_odma_polar",
                     "loads": [1], "eval_ebn0": [4], "eval_sampling": ["distinct"], "eval_frames": 1,
                     "baseline_overrides": {"max_iterations": 30}}
                    for decoder in ("native", "d0")]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for row in rows:
                selection = {"baseline_params": {"prefix_bits": 2}, "pilot_name": "v1", "selection_seed": 3299}
                params = {**selection["baseline_params"], **row["baseline_overrides"]}
                config = {**DEFAULTS, **row, "baseline_params": params, "profile_selection": selection}
                result = {"status": "complete", "config": config, "source_sha256": {"test": "same"},
                          "initial_matrix_sha256": None if row["decoder"] == "native" else "matrix",
                          "evaluation": [{"sampling": "distinct", "K": 1, "ebn0_db": 4, "frames": 1,
                                          "means": {"pupe": 0.0}, "decoder_seconds_per_frame": 1.0}]}
                (root / row["name"]).mkdir()
                (root / row["name"] / "summary.json").write_text(json.dumps(result))
            self.assertEqual(load_results(manifest, root)[1]["status"], "complete")
            result["config"]["baseline_params"]["max_iterations"] = 10
            (root / "d0" / "summary.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "inconsistent pilot selection"):
                load_results(manifest, root)
            result["config"]["baseline_params"]["max_iterations"] = 30
            result["config"]["profile_selection"]["pilot_name"] = "v2"
            (root / "d0" / "summary.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "different selected pilot profiles"):
                load_results(manifest, root)
            result["config"]["profile_selection"]["pilot_name"] = "v1"
            rows[1]["baseline_overrides"] = {"max_iterations": 60}
            result["config"]["baseline_overrides"] = rows[1]["baseline_overrides"]
            result["config"]["baseline_params"]["max_iterations"] = 60
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "d0" / "summary.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "different selected pilot profiles"):
                load_results(manifest, root)

    def test_runner_preserves_pilot_choice_and_applies_cap_to_all_receivers(self):
        main = runpy.run_path("jobs/032_published_baselines/run_row.py")["main"]
        choice = {"baseline_params": {"prefix_bits": 6, "code_length": 32, "crc_bits": 8, "list_size": 8},
                  "pilot_name": "B12_n256_odma_polar_v7", "selection_seed": 3299}
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            original = json.dumps({"status": "complete", "profiles": {"B12_n256_odma_polar": choice}})
            selected.write_text(original)
            captured = []
            with patch.dict(main.__globals__, run_experiment=lambda config, *a, **kw: captured.append(config)):
                for index in (1, 2, 3):
                    with patch("sys.argv", ["run_row.py", "--phase", "comparison", "--index", str(index),
                                            "--selected", str(selected), "--out-root", directory]):
                        main()
            self.assertEqual({c["decoder"] for c in captured}, {"native", "d0", "d1"})
            for config in captured:
                self.assertEqual(config["baseline_params"], {**choice["baseline_params"], "max_iterations": 30})
                self.assertEqual(config["profile_selection"], choice)
            self.assertEqual(selected.read_text(), original)

    def test_paper_alignment_bracket_is_not_an_automatic_verdict(self):
        reference = {"target_pupe": 0.05, "approx_required_ebn0_db": 0.4, "reading_uncertainty_db": 0.15}
        results = [{"config": {"B": 100, "n": 30000, "family": "odma_polar", "loads": [50], "paper_reference": reference}}]
        points = [{"B": 100, "n": 30000, "family": "odma_polar", "K": 50, "ebn0_db": snr, "pupe": pupe}
                  for snr, pupe in [(0, 0.1), (0.5, 0.04), (1, 0.01)]]
        check = paper_alignment(results, points, Path("unused"), make_plots=False)[0]
        self.assertEqual(check["mean_pupe_crossing_bracket_db"], [0, 0.5])
        self.assertFalse(check["nonmonotone_mean_pupe"])
        self.assertNotIn("passed", check)
        self.assertIn("not a confidence interval", check["interpretation"])


if __name__ == "__main__": unittest.main()
