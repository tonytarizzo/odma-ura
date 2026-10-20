"""Independent posterior, proposal, baseline and paired-harness regression checks."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import runpy
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
import torch

from baselines import message_indices
from benchmarks.polar_decoders import build_encoder, fixture, merge_runs, normalized_config, run_experiment, summarize
from benchmarks.ura_bounds import list_fano_converse, polyanskiy_achievability
from benchmarks.ura_comparison import learned_batch
from framework.learned_decoders import UnrolledEffectiveChannelPGD
from framework.polar_decoders import (REGISTRY, Scenario, allocate, count_tails, guided_log_density,
                                      prepare_reference, run)
from framework.polar_mapping import PolarD2
from tests.polar_posterior_test import count_states, exact_posterior


class DecoderBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = normalized_config({"profile": "smoke", "threads": 2})
        cls.base, cls.encoder, cls.phi, _ = build_encoder(cls.config)
        cls.y = cls.phi[:, [1, 1, 5]].sum(1) + np.random.default_rng(75).normal(0, .3, cls.phi.shape[0])

    def test_reference_matches_independent_count_posterior_and_bayes_risk(self):
        for source, k in itertools.product(("iid", "distinct"), (1, 3)):
            s = Scenario(self.phi, self.y, k, .3, source)
            counts, meta = run("REF", s)
            states, probabilities = exact_posterior(self.phi, self.y, k, .3, source)
            expected = np.stack([probabilities @ (states >= j) for j in range(1, k + 1)], axis=1)
            np.testing.assert_allclose(meta["tails"], expected, atol=2e-14)
            all_actions = count_states(16, k, source)
            risk = [1 - probabilities @ np.minimum(states, a).sum(1) / k for a in all_actions]
            self.assertAlmostEqual(meta["posterior_risk"], min(risk), places=12)
            self.assertEqual(int(counts.sum()), k)

    def test_count_marks_retain_repetitions_and_aliases(self):
        tuples = np.array([[1, 1, 1], [0, 1, 2], [0, 0, 1]])
        tails = count_tails(tuples, np.array([.2, .3, .5]), 3)
        np.testing.assert_allclose(tails, [[.8, .5, 0], [1, .2, .2], [.3, 0, 0]])
        phi = np.array([[1., 1., -1.]])
        _, meta = run("REF", Scenario(phi, np.array([1.]), 2, .2))
        np.testing.assert_allclose(meta["tails"][0], meta["tails"][1])
        self.assertEqual(int(allocate(np.array([[1., 1., 1.]]), 3).sum()), 3)

    def test_product_is_exact_for_one_user_and_is_a_finite_trial_at_budget(self):
        s = Scenario(self.phi, self.y, 1, .3)
        counts, meta = run("DEC-02", s)
        ref, exact = run("REF", s)
        np.testing.assert_array_equal(counts, ref)
        np.testing.assert_allclose(meta["tails"], exact["tails"], atol=2e-14)
        self.assertAlmostEqual(meta["elbo"], exact["log_evidence"], places=12)
        counts, meta = run("DEC-02", Scenario(self.phi, self.y, 3, .3), max_iterations=1)
        self.assertEqual(counts.sum(), 3)
        self.assertTrue(np.isfinite(meta["elbo"]))

    def test_full_proposal_normalizes_and_correction_recovers_posterior(self):
        phi = self.phi[:, :4]
        for source in ("iid", "distinct"):
            s = Scenario(phi, self.y, 2, .3, source)
            tuples = np.array(list(itertools.product(range(4), repeat=2)))
            if source == "distinct":
                tuples = tuples[tuples[:, 0] != tuples[:, 1]]
            guide = np.exp(guided_log_density(s, tuples))
            self.assertAlmostEqual(guide.sum(), 1., places=13)
            prior = 1 / len(tuples)
            q = .2 * prior + .8 * guide
            likelihood = np.exp(-np.square(phi[:, tuples].sum(2).T - s.y).sum(1) / (2 * s.noise_var))
            weight = prior * likelihood / q
            self.assertTrue(np.all(weight <= 5 + 1e-14))
            tails = count_tails(tuples, q * weight / np.sum(q * weight), 4)
            _, exact = run("REF", s)
            np.testing.assert_allclose(tails, exact["tails"], atol=2e-14)

    def test_importance_reproducibility_both_sources_and_large_sample_agreement(self):
        for source in ("iid", "distinct"):
            s = Scenario(self.phi[:, :4], self.y, 2, .5, source)
            counts, a = run("DEC-05", s, samples=20_000, seed=77)
            again, b = run("DEC-05", s, samples=20_000, seed=77)
            _, exact = run("REF", s)
            np.testing.assert_array_equal(counts, again)
            np.testing.assert_array_equal(a["tails"], b["tails"])
            self.assertLess(np.max(np.abs(a["tails"] - exact["tails"])), .025)
            self.assertTrue(0 <= a["excess_risk_bound"] <= 1)
            if source == "distinct":
                self.assertLessEqual(counts.max(), 1)

    def test_d2_structured_adapter_matches_old_recurrence(self):
        old, new = UnrolledEffectiveChannelPGD(3).double(), PolarD2(3).double()
        new.load_state_dict(old.state_dict())
        y, h = torch.from_numpy(self.y)[None, :, None], torch.ones((1, 1), dtype=torch.float64)
        with torch.no_grad():
            a = old(self.encoder, y, h, 3, noise_var=.3)
            b = new(self.encoder, y, h, 3, noise_var=.3)
        torch.testing.assert_close(a.meta["soft_counts"], b.meta["soft_counts"], rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(a.counts, b.counts)

    def test_uniform_native_is_exactly_the_existing_paper_adapter(self):
        base, encoder, phi, _ = build_encoder({**self.config, "mapping": "uniform"})
        y = phi[:, [1, 5, 8]].sum(1) + .1
        s = Scenario(phi, y, 3, .2)
        counts, meta = run("ODMA", s, baseline=base, amplitudes=encoder.codebook.amplitudes().numpy())
        words, _ = base.decode(y, 3, .2)
        np.testing.assert_array_equal(counts, np.bincount(message_indices(words), minlength=16))
        self.assertEqual(meta["variant"], "unchanged_repository_paper_adaptation")
        balanced, info = run("ODMA", Scenario(self.phi, self.y, 3, .2), baseline=self.base,
                             amplitudes=self.encoder.codebook.amplitudes().numpy())
        self.assertLessEqual(balanced.sum(), 3)
        self.assertIn("balanced", info["variant"])

    def test_shared_batch_supports_double_without_changing_float_training(self):
        ids, noise, _ = fixture({**self.config, "eval_frames": 2}, self.phi, 3, "distinct")
        from baselines import message_bits
        counts, y, _ = learned_batch(self.encoder, message_bits(ids, 4), noise, 2.)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(counts.dtype, self.encoder.dtype)

    def test_frames_are_invariant_to_decoder_selection_order_and_other_loads(self):
        args = {**self.config, "eval_frames": 4}
        a = fixture(args, self.phi, 3, "iid")
        b = fixture({**args, "decoders": ["ODMA"], "loads": [3], "eval_ebn0": [10]}, self.phi, 3, "iid")
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_caps_unimplemented_sources_and_unknown_settings_fail(self):
        self.assertTrue(all(f"DEC-{i:02}" in REGISTRY for i in range(1, 14)))
        for config in ({"decoders": ["DEC-99"]}, {"eval_sampling": ["distinct"]},
                       {"B": 12, "loads": [7]}, {"decoder_params": {"DEC-05": {"samplez": 1}}},
                       {"decoder_params": {"DEC-06": {"block_size": 4}}}, {"decoder_params": {"DEC-10": {"order": 4}}}):
            with self.assertRaises(ValueError):
                normalized_config(config)
        with self.assertRaises(ValueError):
            prepare_reference(self.phi, 4, "iid", max_states=2)
        with self.assertRaises(ValueError):
            prepare_reference(self.phi, 4, "iid", max_bytes=1)

    def test_two_bounds_order_zero_information_and_monotonicity(self):
        for distinct in (False, True):
            lower = [list_fano_converse(4, 12, 2, x, distinct=distinct) for x in (-100, -8, 0, 8)]
            alphabet = 15 if distinct else 16
            self.assertAlmostEqual(lower[0], 1 - 2 / alphabet, places=4)
            self.assertTrue(np.all(np.diff(lower) <= 0))
            upper = polyanskiy_achievability(4, 12, 2, -8, distinct=distinct, grid=5, power_grid=5, order=16)
            self.assertLessEqual(lower[1], upper)

    def test_split_decoder_jobs_match_combined_and_merge_rejects_duplicates(self):
        config = {"loads": [1, 2], "eval_ebn0": [0.], "eval_frames": 3, "max_epochs": 1, "bounds": False,
                  "threads": 2, "decoders": ["REF", "DEC-05", "DEC-06", "DEC-07", "DEC-11", "DEC-13"],
                  "decoder_params": {"DEC-05": {"samples": 32}, "DEC-06": {"samples": 8, "burn_in": 4, "chains": 2},
                                     "DEC-07": {"batches": 2, "particles": 8}, "DEC-11": {"frequencies": 16},
                                     "DEC-13": {"samples": 16}}}
        with tempfile.TemporaryDirectory() as folder, patch("benchmarks.polar_decoders.plots"), contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            run_experiment(config, root / "together")
            paths = []
            for name in config["decoders"]:
                for loads in ([config["loads"]] if name == "REF" else [[2], [1]]):
                    path = root / (name if name == "REF" else f"{name}_K{loads[0]}")
                    run_experiment({**config, "decoders": [name], "loads": loads}, path)
                    paths.append(path)
            merged = merge_runs(paths, root / "merged")
            together = [json.loads(line) for line in (root / "together/frames.jsonl").read_text().splitlines()]
            separate = [json.loads(line) for line in (root / "merged/frames.jsonl").read_text().splitlines()]
            lookup = {(r["decoder"], r["K"], r["frame"]): r for r in separate}
            for row in together:
                other = lookup[(row["decoder"], row["K"], row["frame"])]
                self.assertEqual(row["counts"], other["counts"])
                self.assertEqual(row["input_sha256"], other["input_sha256"])
                self.assertEqual(row["metrics"], other["metrics"])
            self.assertEqual(len(merged["evaluation"]), 2 * len(config["decoders"]))
            with self.assertRaises(ValueError):
                merge_runs([root / "REF", root / "REF"], root / "duplicate")
            with self.assertRaisesRegex(ValueError, "nested"):
                merge_runs([root / "merged"], root / "nested")
            path = root / "REF/frames.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete frame"):
                merge_runs([root / "REF"], root / "incomplete")

    def test_multi_seed_uncertainty_uses_seed_means(self):
        rows = [{"decoder": "REF", "sampling": "iid", "K": 1, "ebn0_db": 0, "seed": seed, "frame": frame,
                 "seconds": .1, "metrics": {"pupe": float(seed)}, "meta": {}}
                for seed in (0, 1) for frame in range(4)]
        cell = summarize(rows)[0]
        self.assertEqual(cell["confidence_unit"], "seed")
        self.assertAlmostEqual(cell["pupe"]["standard_error"], .5)

    def test_d2_checkpoint_reuse_keeps_counts_and_rejects_training_drift(self):
        config = {"decoders": ["D2"], "loads": [1], "eval_ebn0": [0.], "eval_frames": 2, "bounds": False,
                  "threads": 2, "max_epochs": 1, "batches_per_epoch": 1, "validation_batches": 1, "layers": 2}
        with tempfile.TemporaryDirectory() as folder, patch("benchmarks.polar_decoders.plots"), contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            run_experiment(config, root / "trained")
            checkpoint = str(root / "trained/D2/decoder.pt")
            run_experiment({**config, "d2_checkpoint": checkpoint}, root / "loaded")
            records = [[json.loads(line) for line in (root / name / "frames.jsonl").read_text().splitlines()]
                       for name in ("trained", "loaded")]
            self.assertEqual([r["counts"] for r in records[0]], [r["counts"] for r in records[1]])
            self.assertEqual([r["input_sha256"] for r in records[0]], [r["input_sha256"] for r in records[1]])
            with self.assertRaisesRegex(ValueError, "training contract"):
                run_experiment({**config, "d2_checkpoint": checkpoint, "max_epochs": 2}, root / "bad")

    def test_hpc_manifest_matches_builder_and_excludes_infeasible_reference(self):
        folder = Path("jobs/035_polar_decoders")
        build = runpy.run_path(str(folder / "build_manifest.py"))["manifest"]
        bundles = build()
        self.assertEqual(bundles, [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()])
        self.assertEqual(len(bundles), 90)
        self.assertEqual(len({r["name"] for r in bundles}), 90)
        rows = [r for bundle in bundles for r in bundle["runs"]]
        self.assertEqual(len({r["name"] for r in rows}), 180)
        for bundle in bundles:
            self.assertEqual([r["mapping"] for r in bundle["runs"]], ["balanced", "uniform"])
            self.assertEqual(len({(r["B"], r["seed"], tuple(r["decoders"])) for r in bundle["runs"]}), 1)
        self.assertTrue(all("REF" not in r["decoders"] for r in rows))
        self.assertTrue(all(r["train_loads"] == [7, 15, 22, 26] for r in rows))
        self.assertTrue(all(r["max_epochs"] == 200 and r["layers"] == 8 for r in rows))
        self.assertEqual(len(build(sampling="distinct")), 60)
        self.assertEqual(len(build(mappings=("balanced",))), 90)
        self.assertTrue(all(r["loads"] == [7, 15, 22, 26] and r["threads"] == 2 for r in rows))
        self.assertEqual(sum(r["eval_frames"] * len(r["loads"]) * len(r["eval_ebn0"]) for r in rows), 1_474_560)
        self.assertTrue(all(r["decoder_params"] == {"DEC-07": {"guide_log_floor": -1000.}}
                            for r in rows if r["decoders"] == ["DEC-07"]))
        self.assertEqual({r["decoders"][0] for r in rows}, {f"DEC-{i:02}" for i in range(1, 14)} | {"D2", "ODMA"})
        scheduler = (folder / "035_polar_decoders.sh").read_text()
        self.assertIn("#PBS -J 1-90%4", scheduler)
        self.assertIn("select=1:ncpus=2:mem=16gb", scheduler)

    def test_bundled_job_resume_and_mapping_merge(self):
        folder = Path("jobs/035_polar_decoders")
        build = runpy.run_path(str(folder / "build_manifest.py"))["manifest"]
        execute = runpy.run_path(str(folder / "run_row.py"))["run_bundle"]
        merge = runpy.run_path(str(folder / "merge.py"))["main"]
        bundles = [b for b in build(profile="smoke") if b["runs"][0]["decoders"] in (["DEC-02"], ["DEC-07"])]
        for bundle in bundles:
            for r in bundle["runs"]:
                r.update(loads=[1, 2], eval_frames=2, eval_ebn0=[0.], bounds=False,
                         decoder_params={"DEC-07": {"batches": 2, "particles": 8}})
        with tempfile.TemporaryDirectory() as folder, patch("benchmarks.polar_decoders.plots"), contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            for bundle in bundles:
                execute(bundle, root / "rows")
                before = [(root / "rows" / r["name"] / "frames.jsonl").read_bytes() for r in bundle["runs"]]
                execute(bundle, root / "rows")
                after = [(root / "rows" / r["name"] / "frames.jsonl").read_bytes() for r in bundle["runs"]]
                self.assertEqual(before, after)
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(b) + "\n" for b in bundles))
            argv = ["merge.py", "--manifest", str(manifest), "--results", str(root / "rows"), "--out", str(root / "merged")]
            with patch("sys.argv", argv):
                merge()
            for mapping in ("balanced", "uniform"):
                result = json.loads((root / "merged" / f"B4_{mapping}" / "summary.json").read_text())
                self.assertEqual(len(result["evaluation"]), 4)
                self.assertEqual(result["config"]["mapping"], mapping)
            wrong = {**bundles[0], "runs": [{**bundles[0]["runs"][0], "eval_frames": 3}]}
            with self.assertRaisesRegex(ValueError, "differs"):
                execute(wrong, root / "rows")
            path = root / "rows" / bundles[0]["runs"][0]["name"] / "summary.json"
            saved = json.loads(path.read_text()); saved["source_sha256"] = {}; path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "differs"):
                execute(bundles[0], root / "rows")
            path.unlink()
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                execute(bundles[0], root / "rows")


if __name__ == "__main__":
    unittest.main()
