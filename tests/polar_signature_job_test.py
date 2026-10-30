"""Independent physical/channel checks for the fixed-resource Job038 comparison."""

import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
import torch

from baselines import message_bits
from benchmarks.polar_decoders import normalized_config, build_encoder as reference_encoder, fixture
from framework.polar_decoders import Scenario, run
from framework.polar_tractable import PolarModel, PolarScenario

JOB = Path(__file__).resolve().parents[1] / "jobs/038_polar_signatures"
spec = importlib.util.spec_from_file_location("signature_experiment", JOB / "experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
merge_spec = importlib.util.spec_from_file_location("signature_merge", JOB / "merge.py")
merge = importlib.util.module_from_spec(merge_spec)
with patch.dict(sys.modules, {"experiment": experiment}):
    merge_spec.loader.exec_module(merge)
    comparator_spec = importlib.util.spec_from_file_location("signature_comparators", JOB / "comparators.py")
    comparators = importlib.util.module_from_spec(comparator_spec)
    comparator_spec.loader.exec_module(comparators)


class SignatureJobTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = normalized_config({"profile": "research", "B": 12, "prefix_bits": 6, "seed": 3201,
                                       "loads": [2], "eval_frames": 2, "eval_sampling": ["iid"], "bounds": False,
                                       "decoders": ["DEC-26"]})

    def test_reference_unchanged_and_pairing(self):
        c = self.config
        _, _, old, _ = reference_encoder(c)
        _, _, original, _ = experiment.build_encoder(c, "reference")
        _, _, repaired, _ = experiment.build_encoder(c, "repair3")
        np.testing.assert_array_equal(old, original)
        left, right = fixture(c, original, 2, "iid"), fixture(c, repaired, 2, "iid")
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertFalse(np.array_equal(left[2], right[2]))

    def test_physical_factors_and_procedural_receiver(self):
        base, enc, phi, _ = experiment.build_encoder(self.config, "repair3")
        keys = (base.polar.generator @ (1 << np.arange(5, -1, -1))).tolist()
        self.assertEqual(len(set(keys)), 32)
        self.assertTrue(set(experiment.REFERENCE_KEYS) <= set(keys))
        bits = message_bits(np.arange(4096), 12)
        np.testing.assert_allclose(enc.codebook.codewords(bits).numpy(), phi, atol=1e-14)
        np.testing.assert_allclose(np.square(phi).sum(0), 1, atol=1e-12)
        rng = np.random.default_rng(31)
        counts, residual = rng.normal(size=4096), rng.normal(size=256)
        np.testing.assert_allclose(enc.matvec(torch.from_numpy(counts)).numpy(), phi @ counts, atol=1e-12)
        np.testing.assert_allclose(enc.rmatvec(torch.from_numpy(residual)).numpy(), phi.T @ residual, atol=1e-12)
        diagonal, rows = enc.gram_geometry()
        frame = phi @ phi.T
        np.testing.assert_allclose(diagonal.numpy(), np.square(phi).sum(0), atol=1e-12)
        np.testing.assert_allclose(rows.numpy(), np.sum(phi * (frame @ phi), axis=0), atol=1e-10)
        model = PolarModel(base.polar, base.positions, enc.codebook.amplitudes().numpy(), 256)
        for word in (0, 1, 721, 4095):
            np.testing.assert_array_equal(model.waveform(word), phi[:, word])
        y = phi[:, [12, 54]].sum(1)
        params = dict(width=4, patterns=2, max_refine=8, seed=8)
        with patch("baselines.polar.PolarCode.decode_list", side_effect=AssertionError("SCL is incompatible")):
            sparse, _ = run("DEC-26", PolarScenario(model, y, 2, .02), **params)
            dense, _ = run("DEC-26", Scenario(phi, y, 2, .02), model=model, **params)
        self.assertEqual(sparse, {int(w): int(dense[w]) for w in np.flatnonzero(dense)})
        with self.assertRaises(ValueError):
            base.polar.decode_list(np.zeros(32))

    def test_pair_genie_against_independent_enumeration(self):
        rng = np.random.default_rng(41)
        left = rng.normal(size=(7, 4)); left /= np.linalg.norm(left, axis=0)
        for same in (False, True):
            right = left if same else rng.normal(size=(7, 4))
            right = right / np.linalg.norm(right, axis=0)
            y, variance = rng.normal(size=7), .3
            weights, tails, counts, _ = experiment.pair_posterior(left, right, y, variance, same)
            words = [(u, v if same else v + 4) for u in range(4) for v in range(4)]
            signals = np.array([left[:, u] + right[:, v] for u in range(4) for v in range(4)])
            score = -np.square(signals - y).sum(1) / (2 * variance)
            probability = np.exp(score - logsumexp(score))
            brute = np.zeros_like(tails)
            for pair, mass in zip(words, probability):
                for w in set(pair):
                    brute[w, :pair.count(w)] += mass
            np.testing.assert_allclose(weights.ravel(), probability, atol=1e-14)
            np.testing.assert_allclose(tails, brute, atol=1e-14)
            best = max(sum(brute[w, :pair.count(w)].sum() for w in set(pair))
                       for pair in ((u, v) for u in range(len(tails)) for v in range(u, len(tails))))
            self.assertAlmostEqual(sum(tails[w, :c].sum() for w, c in enumerate(counts)), best)

    def test_wrong_fixture_rejected(self):
        c = {**self.config, "crc_bits": 4}
        with self.assertRaises(ValueError):
            experiment.build_encoder(c, "repair3")

    def test_paired_comparison_rejects_changed_noise_or_energy(self):
        cohort = {}
        for signature in ("reference", "repair3"):
            for mapping in ("uniform", "balanced"):
                result = {"config": {"decoders": ["D2"], "loads": [7], "eval_ebn0": [4]},
                          "encoder": {"positions_sha256": "positions", "amplitudes_sha256": mapping}}
                record = {"decoder": "D2", "K": 7, "ebn0_db": 4, "frame": 0, "seed": 3201,
                          "truth_sha256": "truth", "standard_noise_sha256": "noise", "seconds": .1,
                          "metrics": {"pupe": .1 if signature == "reference" else .05}, "meta": {}}
                cohort[(12, 3201, signature, mapping)] = result, {merge.key(record): record}, []
        cells, contrasts = merge.compare(cohort)
        self.assertEqual(len(cells), 4)
        self.assertAlmostEqual(contrasts[0]["repair_minus_reference_balanced"]["mean"], -.05)
        bad = copy.deepcopy(cohort)
        entry = bad[(12, 3201, "repair3", "balanced")]
        entry[1][("D2", 7, 4, 0)]["standard_noise_sha256"] = "different"
        with self.assertRaises(ValueError):
            merge.compare(bad)
        bad = copy.deepcopy(cohort)
        bad[(12, 3201, "repair3", "balanced")][0]["encoder"]["amplitudes_sha256"] = "different"
        with self.assertRaises(ValueError):
            merge.compare(bad)
        full = {(b, seed, f"polar_{signature}_{mapping}"): entry
                for (b, seed, signature, mapping), entry in cohort.items()}
        for label in ("dense_fixed", "dense_joint", "sparse_fixed", "sparse_joint", "polar_energy_joint"):
            full[(12, 3201, label)] = copy.deepcopy(full[(12, 3201, "polar_reference_uniform")])
        all_cells, all_contrasts = merge.compare_all_d2(full)
        self.assertEqual(len(all_cells), 9)
        self.assertEqual(len(all_contrasts), 16)
        full[(12, 3201, "dense_joint")][1][("D2", 7, 4, 0)]["truth_sha256"] = "unpaired"
        with self.assertRaises(ValueError):
            merge.compare_all_d2(full)

    def test_manifest_and_uncapped_submission(self):
        rows = [json.loads(line) for line in (JOB / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 54)
        expected = {(b, seed, signature, mapping) for b in (12, 14) for seed in (3201, 3202, 3203)
                    for signature in ("reference", "repair3") for mapping in ("uniform", "balanced")}
        self.assertEqual({(r["config"]["B"], r["config"]["seed"], r["signature"], r["config"]["mapping"])
                          for r in rows if "signature" in r}, expected)
        self.assertTrue(all(r["config"]["decoders"] == ["D2"] for r in rows))
        self.assertEqual(len({(r["config"]["B"], r["config"]["seed"], r["encoder_id"]) for r in rows}), 54)
        self.assertEqual(sum(len(r["config"]["decoders"]) * len(r["config"]["loads"]) * len(r["config"]["eval_ebn0"])
                             * r["config"]["eval_frames"] for r in rows), 27648)
        for script in JOB.glob("*.sh"):
            for line in script.read_text().splitlines():
                if line.startswith("#PBS -J"):
                    self.assertNotIn("%", line, str(script))

    def test_archived_matrix_normalization_and_decoder_contract(self):
        from benchmarks import ura_comparison
        from framework.polar_mapping import PolarD2
        c = normalized_config({"profile": "smoke", "B": 4, "n": 12, "loads": [2], "train_loads": [2],
                               "decoders": ["D2"], "eval_sampling": ["iid"], "bounds": False})
        old = {**c, "family": "dense", "mode": "fixed", "decoder": "d2", "loads": c["train_loads"]}
        encoder = ura_comparison.make_encoder(old)
        original = encoder.explicit_matrix().detach().numpy().copy()
        decoder = PolarD2(c["layers"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / f"B{c['B']}_n256_dense_fixed_d2_s{c['seed']}"
            path.mkdir()
            summary = {"status": "complete", "config": old, "initial_matrix_sha256": experiment.digest(original),
                       "revision": "synthetic", "source_sha256": {}, "training": {}, "encoder_learned_parameters": 0}
            (path / "summary.json").write_text(json.dumps(summary))
            torch.save({"config": old, "encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, path / "checkpoint.pt")
            row = {"config": c, "encoder_id": "dense_fixed"}
            actual, learned, phi, provenance = comparators.load_checkpoint(row, root, root)
            expected = original.astype(np.float64)
            expected /= np.linalg.norm(expected, axis=0)
            np.testing.assert_array_equal(phi, expected)
            np.testing.assert_allclose(np.square(phi).sum(0), 1, atol=1e-14)
            self.assertLess(provenance["maximum_column_rescaling"], 1e-6)
            y = phi[:, [3, 3]].sum(1)
            counts, _ = run("D2", Scenario(phi, y, 2, .02), encoder=actual, decoder=learned)
            reference = decoder.double()(actual, torch.tensor(y)[None, :, None], torch.ones(1, 1).double(), 2, noise_var=.02)
            np.testing.assert_array_equal(counts, reference.counts[0].numpy())
            bad = copy.deepcopy(row)
            bad["config"]["layers"] += 1
            with self.assertRaises(ValueError):
                comparators.load_checkpoint(bad, root, root)


if __name__ == "__main__":
    unittest.main()
