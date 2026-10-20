"""Matched follow-up manifests, procedural/explicit D2 equivalence and amplitude gradients."""

import hashlib
import json
from pathlib import Path
import runpy
import unittest

import numpy as np
import torch

from benchmarks.ura_comparison import DEFAULTS, ExplicitEncoder, forward, learned_batch, make_decoder, make_encoder, objective
from benchmarks.ura_merge import aggregate, family_label
from framework.prototype_amplitudes import build_prototype_hash_encoder


class MatchedD2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_manifest_and_contract(self):
        folder = Path("jobs/033_matched_d2")
        rows = runpy.run_path(str(folder / "build_manifest.py"))["manifest"]()
        self.assertEqual(rows, [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()])
        self.assertEqual(len(rows), 66)
        self.assertEqual(len({r["name"] for r in rows}), 66)
        self.assertIn("#PBS -J 1-66", (folder / "033_matched_d2.sh").read_text())
        originals = [json.loads(line) for line in Path("jobs/032_published_baselines/comparison.jsonl").read_text().splitlines()]
        resolve = runpy.run_path(str(folder / "run_row.py"))["resolve_config"]
        for row in rows:
            self.assertEqual(row["decoder"], "d2")
            self.assertEqual({k: row[k] for k in DEFAULTS}, DEFAULTS)
            if "selected_profile" in row:
                old = next(r for r in originals if r["name"] == row["name"].replace("_d2_", "_d1_"))
                self.assertEqual(row, {**old, "name": row["name"], "decoder": "d2"})
                choice = {"baseline_params": {"example": 1}, "pilot_name": "unchanged"}
                config = resolve(row, {"status": "complete", "profiles": {row["selected_profile"]: choice}})
                self.assertEqual(config["profile_selection"], choice)
                self.assertEqual(config["baseline_params"], {"example": 1, **row.get("baseline_overrides", {})})
                with self.assertRaises(ValueError): resolve(row, {"status": "partial"})
            else:
                self.assertIn(row["amplitude_label_bits"], (4, 8, 10, row["B"]))
                self.assertEqual(row["support"], 32)
                self.assertEqual(row["hash_search_candidates"], 128)
        self.assertEqual(sum("selected_profile" in r for r in rows), 18)
        self.assertEqual(sum(r["mode"] == "joint" for r in rows), 24)

    def test_amplitude_labels_do_not_pool_different_J(self):
        results = []
        for j in (4, 8, 10):
            config = {"family": "hash_coordinate", "amplitude_label_bits": j, "B": 14, "n": 256,
                      "mode": "fixed", "decoder": "d2", "seed": 3201}
            cell = {"sampling": "distinct", "K": 7, "ebn0_db": 0, "frames": 1,
                    "means": {"pupe": j / 100}, "decoder_seconds_per_frame": 1}
            results.append({"config": config, "evaluation": [cell]})
        self.assertEqual({p["family"] for p in aggregate(results)}, {family_label(r["config"]) for r in results})

    def test_cross_job_source_audit_is_exact(self):
        module = runpy.run_path("jobs/033_matched_d2/merge.py")
        changes, audit = module["AUDITED_CHANGES"], module["audit_source_change"]
        old = {path: pair[0] for path, pair in changes.items()}
        new = {path: pair[1] for path, pair in changes.items()}
        self.assertEqual(audit(old, new), changes)
        current = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in changes}
        # The PUPE correction must not be admitted by the historical no-metric-change audit.
        with self.assertRaises(ValueError): audit(old, current)
        self.assertEqual(audit(old, old), {})
        with self.assertRaises(ValueError): audit(old, {**new, "unreviewed.py": "new"})
        with self.assertRaises(ValueError): audit(old, {**new, "benchmarks/ura_comparison.py": "unreviewed"})

    def test_cross_job_published_control_rejects_waveform_or_profile_changes(self):
        audit = runpy.run_path("jobs/033_matched_d2/merge.py")["audit_controls"]
        base = {**DEFAULTS, "B": 12, "n": 256, "seed": 3201, "mode": "fixed"}
        selected = {"profile_selection": {"pilot_name": "v1"}, "selected_profile": "B12_n256_odma_polar",
                    "baseline_params": {"max_iterations": 30}, "baseline_overrides": {"max_iterations": 30}}
        original = {"config": {**base, **selected, "family": "odma_polar", "decoder": "d1"},
                    "initial_matrix_sha256": "same"}
        dense = {"config": {**base, "family": "dense", "decoder": "d2"}}
        control = {"config": {**original["config"], "decoder": "d2"}, "initial_matrix_sha256": "same"}
        audit([dense, original], [control])
        with self.assertRaisesRegex(ValueError, "codebook differs"):
            audit([dense, original], [{**control, "initial_matrix_sha256": "different"}])
        changed = {**control, "config": {**control["config"], "baseline_params": {"max_iterations": 10}}}
        with self.assertRaisesRegex(ValueError, "profile changed"): audit([dense, original], [changed])
        changed = {**control, "config": {**control["config"], "eval_frames": 8}}
        with self.assertRaisesRegex(ValueError, "mismatch"): audit([dense, original], [changed])

    def test_explicit_equivalence_energy_and_gradients(self):
        base = {**DEFAULTS, "B": 6, "n": 32, "seed": 3217, "loads": [2], "support": 8,
                "hash_search_candidates": 2, "decoder": "d2", "layers": 3, "mode": "joint"}
        for family, j in (("hash_full", 6), ("hash_coordinate", 2), ("hash_coordinate", 4)):
            with self.subTest(family=family, J=j):
                config = {**base, "family": family, "amplitude_label_bits": j}
                encoder = make_encoder(config)
                fixed = make_encoder({**config, "mode": "fixed"})
                matrix = encoder.explicit_matrix()
                self.assertTrue(torch.equal(matrix, fixed.explicit_matrix()))
                self.assertEqual([n for n, _ in encoder.named_parameters()], ["codebook.amplitude_bank.prototypes"])
                self.assertFalse(list(fixed.parameters()))
                self.assertTrue(torch.allclose(matrix.square().sum(0), torch.ones(64), atol=3e-7))
                self.assertEqual(encoder.codebook.amplitude_bank.prototypes.numel(), 8 * (1 << j))
                if family == "hash_full":
                    legacy = build_prototype_hash_encoder(encoder.spec, 8, 3217, 6, False, search_candidates=2)
                    self.assertTrue(torch.allclose(matrix, legacy.explicit_matrix(), atol=2e-7))
                reference = make_encoder({**base, "family": "dense", "mode": "fixed"})
                self.assertIsInstance(reference, ExplicitEncoder)
                with torch.no_grad(): reference.components[0].C.copy_(matrix)
                bits = np.array([[[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]]], dtype=np.uint8)
                counts, y, variance = learned_batch(encoder, bits, np.zeros((1, 32)), 4)
                decoder, other = make_decoder(config), make_decoder(config)
                output, _ = forward(config, decoder, encoder, y, 2, variance)
                expected, _ = forward(config, other, reference, y.detach(), 2, variance)
                self.assertTrue(torch.allclose(output.meta["soft_counts"], expected.meta["soft_counts"], atol=2e-6))
                values = encoder.codebook.amplitude_bank.prototypes
                initial = values.detach().clone()
                objective(config, output, counts, None).backward()
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertGreater(float(values.grad.norm()), 0)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()))
                torch.optim.Adam(encoder.parameters(), lr=.01).step(); encoder.apply_constraints()
                self.assertFalse(torch.equal(values, initial))
                self.assertTrue(torch.allclose(encoder.explicit_matrix().square().sum(0), torch.ones(64), atol=3e-7))
                self.assertFalse(any("_message" in k for k in encoder.state_dict()))


if __name__ == "__main__": unittest.main()
