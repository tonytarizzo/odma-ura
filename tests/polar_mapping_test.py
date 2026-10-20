"""Certify mapping identities, gradients, exact D2 geometry and the 033 contract."""

import copy
import hashlib
import json
from pathlib import Path
import runpy
import unittest

import numpy as np
import torch

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.ura_comparison import DEFAULTS, learned_batch, materialize, objective
from framework.learned_decoders import UnrolledEffectiveChannelPGD
from framework.polar_mapping import PolarD2, PolarExplicitEncoder, PolarMappedCodebook


class PolarMappingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(731)
        self.baseline = ODMAPolarBaseline(6, 64, 3217, prefix_bits=3, code_length=32, crc_bits=8, list_size=8)

    def test_manifest(self):
        folder = Path("jobs/034_polar_mapping")
        rows = runpy.run_path(str(folder / "build_manifest.py"))["manifest"]()
        self.assertEqual(rows, [json.loads(x) for x in (folder / "manifest.jsonl").read_text().splitlines()])
        self.assertEqual(len({r["name"] for r in rows}), 12)
        self.assertIn("#PBS -J 1-6", (folder / "034_polar_mapping.sh").read_text())
        old = {r["name"]: r for r in map(json.loads, Path("jobs/033_matched_d2/manifest.jsonl").read_text().splitlines())}
        for r in rows:
            self.assertEqual({k: r[k] for k in DEFAULTS}, DEFAULTS)
            expected = {**old[r["reference_033"]], "name": r["name"], "family": r["family"],
                        "mode": "joint", "reference_033": r["reference_033"]}
            self.assertEqual(r, expected)

    def test_initialization_matches_all_six_033_codebooks(self):
        for b in (12, 14):
            params = {"prefix_bits": b - 6, "code_length": 32, "crc_bits": 8, "list_size": 8}
            for seed in (3201, 3202, 3203):
                baseline = ODMAPolarBaseline(b, 256, seed, **params)
                reference = materialize(baseline)
                path = Path(f"jobs/033_matched_d2/results/B{b}_n256_odma_polar_d2_s{seed}/summary.json")
                if path.exists():
                    self.assertEqual(hashlib.sha256(reference.tobytes()).hexdigest(),
                                     json.loads(path.read_text())["initial_matrix_sha256"])
                for mapping in ("identity", "energy"):
                    encoder = PolarExplicitEncoder(baseline, [7], mapping, mapping != "identity")
                    np.testing.assert_array_equal(encoder.explicit_matrix().detach().numpy(), reference)

    def test_factorized_products_geometry_and_gradients(self):
        mapping = "energy"
        encoder = PolarExplicitEncoder(self.baseline, [2, 4], mapping, True).double()
        with torch.no_grad(): encoder.codebook.values.normal_(0, .1)
        matrix = encoder.explicit_matrix()
        counts, residual = torch.randn(3, 64, dtype=torch.float64), torch.randn(3, 64, dtype=torch.float64)
        torch.testing.assert_close(encoder.matvec(counts), counts @ matrix.T)
        torch.testing.assert_close(encoder.rmatvec(residual), residual @ matrix)
        torch.testing.assert_close(encoder.matvec(counts[0]), counts[0] @ matrix.T)
        torch.testing.assert_close(encoder.rmatvec(residual[0]), residual[0] @ matrix)
        diagonal, energy = encoder.gram_geometry()
        gram = matrix.T @ matrix
        torch.testing.assert_close(diagonal, gram.diag())
        torch.testing.assert_close(energy, gram.square().sum(1))
        actual = torch.autograd.grad(energy.sum(), encoder.codebook.values, retain_graph=True)[0]
        expected = torch.autograd.grad(gram.square().sum(), encoder.codebook.values)[0]
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    def test_d2_recurrence_and_training(self):
        mapping = "energy"
        encoder = PolarExplicitEncoder(self.baseline, [2], mapping, True)
        other = copy.deepcopy(encoder)
        bits = message_bits(np.array([[3, 17], [11, 32]]), 6)
        counts, y, variance = learned_batch(encoder, bits, np.zeros((2, 64)), 4)
        _, other_y, _ = learned_batch(other, bits, np.zeros((2, 64)), 4)
        decoder, reference = PolarD2(3), UnrolledEffectiveChannelPGD(3)
        h = torch.ones(2, 1)
        out = decoder(encoder, y[..., None], h, 2, noise_var=variance)
        expected = reference(other, other_y[..., None], h, 2, noise_var=variance)
        torch.testing.assert_close(out.meta["soft_counts"], expected.meta["soft_counts"], atol=2e-6, rtol=2e-5)
        config = {"decoder": "d2"}
        objective(config, out, counts, None).backward()
        objective(config, expected, counts, None).backward()
        torch.testing.assert_close(encoder.codebook.values.grad, other.codebook.values.grad, atol=3e-6, rtol=1e-4)
        self.assertGreater(float(encoder.codebook.values.grad.norm()), 1e-7)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()))
        optimizer = torch.optim.Adam(encoder.parameters(), lr=.01)
        optimizer.step(); encoder.apply_constraints()
        torch.testing.assert_close(encoder.explicit_matrix().square().sum(0), torch.ones(64), atol=2e-6, rtol=0)
        clone = PolarExplicitEncoder(self.baseline, [2], mapping, True)
        clone.load_state_dict(encoder.state_dict())
        torch.testing.assert_close(clone.explicit_matrix(), encoder.explicit_matrix())

    def test_retired_mixer_and_legacy_energy_checkpoint(self):
        with self.assertRaises(ValueError): PolarMappedCodebook(self.baseline, "mixer", True)
        encoder = PolarExplicitEncoder(self.baseline, [2], "energy", True)
        state = encoder.state_dict()
        state["codebook.upper"] = torch.triu_indices(32, 32, offset=1)
        encoder.load_state_dict(state, strict=True)
        state["unrecognized_weight"] = torch.tensor(1.)
        with self.assertRaises(RuntimeError): encoder.load_state_dict(state, strict=True)

    def test_large_B_procedural_energy_no_global_table(self):
        baseline = ODMAPolarBaseline(100, 1024, 17, prefix_bits=8, code_length=128, crc_bits=8, list_size=8)
        bits = torch.randint(2, (17, 100), dtype=torch.uint8)
        mapping = "energy"
        book = PolarMappedCodebook(baseline, mapping, True)
        with torch.no_grad(): book.values.normal_(0, .2)
        words = book.codewords(bits)
        self.assertEqual(words.shape, (1024, 17))
        self.assertFalse(book.global_message_axis_present)
        torch.testing.assert_close(words.square().sum(0), torch.ones(17), atol=3e-6, rtol=0)
        words[0].sum().backward()
        self.assertTrue(torch.isfinite(book.values.grad).all())
        with self.assertRaises(ValueError): PolarExplicitEncoder(baseline, [2])

    def test_merger_rejects_unreviewed_changes(self):
        merge = runpy.run_path("jobs/034_polar_mapping/merge.py")
        old = {"historical.py": "unchanged"}
        new = {**old, **{p: "new" for p in merge["ADDED_SOURCES"]}}
        merge["audit_source_extension"](old, new)
        with self.assertRaises(ValueError): merge["audit_source_extension"](old, {**new, "historical.py": "changed"})
        with self.assertRaises(ValueError): merge["audit_source_extension"](old, {**new, "surprise.py": "new"})

    def test_merger_checks_matched_controls(self):
        merge = runpy.run_path("jobs/034_polar_mapping/merge.py")
        config = {**DEFAULTS, "name": "reference", "family": "odma_polar", "mode": "fixed", "decoder": "d2",
                  "B": 12, "n": 256, "seed": 3201, "baseline_params": {"prefix_bits": 6, "code_length": 32}}
        reference = {"config": config, "initial_matrix_sha256": "same", "decoder_parameters": 16}
        for family, count in (("polar_energy", 2048), ("polar_mixer", 496)):
            result = {**reference, "config": {**config, "name": family, "family": family, "mode": "joint",
                                              "reference_033": "reference"}, "encoder_learned_parameters": count,
                      "energy_audit_initial": {"min": 1., "max": 1.}, "energy_audit_final": {"min": 1., "max": 1.},
                      "mapping_diagnostics_final": {"orthogonality_max_error": 1e-7}}
            merge["audit_controls"]([reference], [result])
            for key, value in (("max_epochs", 120), ("seed", 999), ("B", 14)):
                altered = copy.deepcopy(result)
                altered["config"][key] = value
                with self.assertRaises(ValueError): merge["audit_controls"]([reference], [altered])
            for key, value in (("initial_matrix_sha256", "different"), ("encoder_learned_parameters", 1),
                               ("decoder_parameters", 17), ("energy_audit_final", {"min": .9, "max": 1.}),
                               ("energy_audit_final", {"min": float("nan"), "max": float("nan")})):
                with self.assertRaises(ValueError): merge["audit_controls"]([reference], [{**result, key: value}])


if __name__ == "__main__": unittest.main()
