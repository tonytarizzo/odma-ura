"""Algebra, receiver and matching checks; not a published-curve reproduction."""

from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines import make_baseline, message_bits, message_indices, run
from baselines.ccs_amp import CCSAMPBaseline
from baselines.dynamic_cs import DynamicCSBaseline, _SignBank
from baselines.polar import PolarCode, _RELIABILITY, append_crc, polar_transform
from benchmarks.ura_bounds import collision_rates, polyanskiy_achievability, polyanskiy_gallager
from benchmarks.ura_comparison import (DEFAULTS, forward, frame_metrics, learned_batch, make_decoder, make_encoder,
                                       materialize, objective)


class PublishedBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rng = np.random.default_rng(3281)

    def test_polar_transform_and_reliability(self):
        self.assertEqual(len(_RELIABILITY), 512)
        np.testing.assert_array_equal(np.sort(_RELIABILITY), np.arange(512))
        data = self.rng.integers(0, 2, (5, 32), dtype=np.uint8)
        np.testing.assert_array_equal(polar_transform(polar_transform(data)), data)

    def test_crc_known_answer(self):
        # CRC-16/XMODEM, standard check value for ASCII "123456789".
        bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
        crc = append_crc(bits, 16)[-16:]
        self.assertEqual(int(message_indices(crc)), 0x31C3)

    def test_polar_full_list_is_exact_ml(self):
        for n, k in [(4, 2), (8, 3), (16, 4)]:
            code = PolarCode(k, n, crc_bits=0, list_size=1 << k)
            bits = message_bits(np.arange(1 << k), k)
            words = code.encode(bits).astype(float)
            for _ in range(4):
                llr = self.rng.normal(size=n)
                decoded, metric = code.decode_list(llr)
                exact = np.logaddexp(0, -(1 - 2 * words) * llr).sum(1)
                np.testing.assert_array_equal(decoded, bits[np.argsort(exact)])
                np.testing.assert_allclose(metric - metric[0], np.sort(exact) - exact.min(), atol=1e-12)

    def test_native_noiseless_and_high_noise(self):
        for name in ("odma_polar", "dynamic_cs", "ccs_amp", "ccs_block"):
            params = {"prefix_bits": 4, "code_length": 32, "crc_bits": 8, "list_size": 32} if name == "odma_polar" else {}
            baseline = make_baseline(name, 8, 128, 31, **params)
            single_hits = 0
            for value in (11, 23, 173):
                singleton = message_bits([value], 8)
                decoded, _ = baseline.decode(baseline.encode(singleton)[0], 1, 1e-8)
                single_hits += 1 - frame_metrics(singleton, decoded)["pupe"]
            # Finite-iteration AMP need not be ML, even noiselessly. In particular
            # the tiny dense CCS adaptation can oscillate on its shared DC atom.
            self.assertGreaterEqual(single_hits, 2 if name == "ccs_amp" else 3, name)
            high_hits = low_hits = 0
            for _ in range(4):
                bits = message_bits(self.rng.choice(256, 2, replace=False), 8)
                clean = baseline.encode(bits).sum(0)
                decoded, _ = baseline.decode(clean, 2, 1e-8)
                high_hits += 2 * (1 - frame_metrics(bits, decoded)["pupe"])
                z = self.rng.normal(size=128)
                decoded, _ = baseline.decode(clean + 10 * z, 2, 100.0)
                low_hits += 2 * (1 - frame_metrics(bits, decoded)["pupe"])
            self.assertGreater(high_hits, low_hits, name)

    def test_energy_and_explicit_signal_match(self):
        for name in ("odma_polar", "dynamic_cs", "ccs_amp", "ccs_block"):
            baseline = make_baseline(name, 8, 128, 33)
            matrix = materialize(baseline)
            bits = message_bits([0, 13, 13, 255], 8)
            np.testing.assert_allclose(matrix[:, [0, 13, 13, 255]].sum(1), baseline.encode(bits).sum(0), atol=1e-7)
            energy = np.square(matrix.astype(float)).sum(0)
            if name != "ccs_amp": np.testing.assert_allclose(energy, 1, atol=1e-6)
            else: self.assertGreater(energy.std(), 0.01)

    def test_ccs_pinned_author_encoder_exactly_matches(self):
        for b, mode, non_dc in product([8, 12, 14], ["dense", "block_diagonal"], [False, True]):
            baseline = CCSAMPBaseline(b, 256, 37, inner_mode=mode, non_dc_embedding=non_dc)
            bits = self.rng.integers(0, 2, (3, b), dtype=np.uint8)
            words = baseline.graph.encodemessages(bits)
            direct = baseline._encoder.Encode(words.sum(0)).ravel()
            np.testing.assert_allclose(baseline.encode(bits).sum(0)[:baseline.used_n], direct, atol=1e-12)
            inferred = baseline._messages(words)
            np.testing.assert_array_equal(inferred, bits)

    def test_ccs_non_dc_embedding_zero_fragment_regression(self):
        baseline = CCSAMPBaseline(8, 128, 31, non_dc_embedding=True)
        bits = message_bits([11], 8)
        decoded, _ = baseline.decode(baseline.encode(bits)[0], 1, 1e-8)
        self.assertEqual(frame_metrics(bits, decoded)["pupe"], 0)

    def test_dynamic_counter_bank_and_modulated_operator(self):
        bank = _SignBank(20, 64, 34, cache_bytes=0)
        cached = _SignBank(20, 64, 34)
        x = self.rng.normal(size=(64, 3))
        np.testing.assert_array_equal(bank.columns_at([13, 1, 63]), cached.columns_at([13, 1, 63]))
        np.testing.assert_allclose(bank.multiply(x, 17, 64), cached.multiply(x, 17, 64), atol=1e-12)
        r = self.rng.normal(size=(17, 3))
        self.assertAlmostEqual(float((bank.multiply(x, 17, 64) * r).sum()),
                               float((x * bank.multiply(r, 17, 64, transpose=True)).sum()), places=10)
        baseline = DynamicCSBaseline(8, 128, streams=2, prefix_bits=4, data_bits=[4, 4], crc_bits=4)
        tested = 0
        for bits in message_bits(np.arange(256), 8):
            header, values = baseline._parts(bits)
            if any(len(set(indices)) != 2 for indices in values):
                continue  # Source receiver is Bernoulli; it does not resolve within-slot multiplicities.
            llrs = []
            for indices in values:
                score = np.full(4, -100.0)
                score[indices] = 100.0
                llrs.append(score)
            recovered, _, _ = baseline._assemble(header, llrs)
            np.testing.assert_array_equal(recovered, bits)
            tested += 1
        self.assertGreater(tested, 100)

    def test_dynamic_double_cache_preserves_operator_and_receiver(self):
        a = _SignBank(37, 300, 34, cache_bytes=1_000_000)
        b = _SignBank(37, 300, 34, cache_bytes=1_000_000, cache_dtype="float64")
        np.testing.assert_array_equal(a._cache, b._cache)
        for transpose in (False, True):
            x = self.rng.normal(size=(29 if transpose else 290, 5))
            np.testing.assert_allclose(a.multiply(x, 29, 290, transpose), b.multiply(x, 29, 290, transpose), atol=1e-12)
        self.assertIsNone(_SignBank(37, 300, 34, cache_bytes=50_000, cache_dtype="float64")._cache)
        old = DynamicCSBaseline(8, 128, 31)
        new = DynamicCSBaseline(8, 128, 31, cache_dtype="float64")
        for _ in range(4):
            truth = message_bits(self.rng.choice(256, 3, replace=False), 8)
            y = old.encode(truth).sum(0) + .05*self.rng.normal(size=128)
            da, ma = old.decode(y, 3, .05**2)
            db, mb = new.decode(y, 3, .05**2)
            np.testing.assert_array_equal(da, db)
            self.assertEqual(ma, mb)

    def test_run_contract_ignores_truth(self):
        baseline = make_baseline("odma_polar", 8, 128, 15)
        bits = message_bits([23], 8)
        scenario = SimpleNamespace(Y=baseline.encode(bits).sum(0)[:, None], noise_var=1e-8, num_devices_active=1)
        counts, meta = run(scenario, baseline=baseline)
        self.assertEqual(counts[23], 1)
        self.assertIn("presence", meta["count_semantics"])

    def test_dynamic_false_alarm_removal_restarts_amp_without_crc_success(self):
        baseline = DynamicCSBaseline(8, 128, 31, prefix_bits=4, global_iterations=2)
        message = message_bits([16], 8)[0]

        def amp(y, bank, columns, amplitude, prior, noise_var, signatures=None):
            if signatures is None:
                evidence = np.full(columns, -10.0)
                evidence[1:3] = 10.0
                return evidence
            evidence = np.full((columns, signatures.shape[1]), -10.0)
            evidence[:, 0] = 1.0
            return evidence

        with patch.object(baseline, "_amp", side_effect=amp), patch.object(
                baseline, "_assemble", side_effect=[(None, -np.inf, 1), (message, 1.0, 1)]):
            decoded, meta = baseline.decode(np.zeros(128), 1, 0.01)
        np.testing.assert_array_equal(decoded, message[None, :])
        self.assertEqual(meta["global_iterations"], 2)
        self.assertEqual(meta["fa_removed"], 1)

    def test_metric_duplicates_and_missed_proposals(self):
        truth = message_bits([1, 1, 3], 3)
        result = frame_metrics(truth, message_bits([1], 3), message_bits([1, 2], 3))
        self.assertAlmostEqual(result["pupe"], 1 / 3)
        self.assertAlmostEqual(result["strict_pupe"], 1)
        self.assertAlmostEqual(result["candidate_recall"], 2 / 3)
        with self.assertRaises(ValueError): frame_metrics(truth, message_bits([1, 1], 3))

    def test_all_decoders_gradients_and_joint_constraints(self):
        for name, family, mode in product(["d0", "d1", "d2", "d3", "d4"], ["dense", "sparse"], ["fixed", "joint"]):
            cfg = {**DEFAULTS, "B": 5, "n": 24, "seed": 31, "family": family, "mode": mode,
                   "decoder": name, "loads": [2], "layers": 2, "hidden_dim": 8, "candidate_size": 12}
            encoder, decoder = make_encoder(cfg), make_decoder(cfg)
            original = encoder.explicit_matrix().detach().clone()
            bits = message_bits([[1, 4], [3, 7]], 5)
            counts, y, variance = learned_batch(encoder, bits, self.rng.normal(size=(2, 24)), 4)
            output, candidates = forward(cfg, decoder, encoder, y, 2, variance)
            loss = objective(cfg, output, counts, candidates)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in decoder.parameters()))
            if mode == "joint":
                parameter = encoder.components[0].C
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
                with torch.no_grad(): parameter.add_(parameter.grad, alpha=-0.001)
                encoder.apply_constraints()
                torch.testing.assert_close(parameter.square().sum(0), torch.ones(32))
                self.assertTrue(torch.equal(parameter[original == 0], torch.zeros_like(parameter[original == 0])))

    def test_d3_full_list_matches_d2(self):
        cfg = {**DEFAULTS, "B": 5, "n": 24, "seed": 31, "family": "dense", "mode": "fixed",
               "decoder": "d2", "loads": [2], "layers": 3, "candidate_size": 32}
        encoder = make_encoder(cfg)
        bits = message_bits([[1, 4], [3, 7]], 5)
        _, y, var = learned_batch(encoder, bits, self.rng.normal(size=(2, 24)), 4)
        full, _ = forward(cfg, make_decoder(cfg), encoder, y, 2, var)
        cfg["decoder"] = "d3"
        restricted, indices = forward(cfg, make_decoder(cfg), encoder, y, 2, var)
        restored = torch.zeros_like(full.meta["soft_counts"]).scatter(1, indices, restricted.meta["soft_counts"])
        torch.testing.assert_close(full.meta["soft_counts"], restored, atol=1e-5, rtol=1e-5)

    def test_candidate_loss_includes_rejected_true_messages(self):
        indices = torch.tensor([[0, 1]])
        logits = torch.tensor([[10.0, -10.0]], requires_grad=True)
        output = SimpleNamespace(meta={"layer_evidence_logits": [logits], "soft_counts": torch.tensor([[1.0, 0.0]])})
        retained = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        missed = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        good = objective({"decoder": "d3"}, output, retained, indices)
        bad = objective({"decoder": "d3"}, output, missed, indices)
        self.assertGreater(float(bad.detach()), float(good.detach()) + 10)
        bad.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bounds(self):
        collisions = collision_rates(14, 26)
        self.assertAlmostEqual(collisions["per_user_duplicate_probability"], 1 - (1 - 2 ** -14) ** 25)
        for function in (polyanskiy_gallager, polyanskiy_achievability):
            values = [function(12, 256, 7, snr) for snr in [-6, 0, 6, 12]]
            self.assertTrue(all(0 <= x <= 1 for x in values))
            self.assertTrue(all(a >= b - 1e-10 for a, b in zip(values, values[1:])), values)
        self.assertGreaterEqual(polyanskiy_gallager(8, 128, 4, 2), polyanskiy_gallager(8, 128, 4, 2, distinct=True))


if __name__ == "__main__": unittest.main()
