"""Physical one-copy identities and finite complete count-cavity integration, including repeated messages."""

from collections import Counter
import hashlib
import unittest

import numpy as np

from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode
from benchmarks.polar_energy_geometry import balanced_energy
from framework.polar_count_cavity import MODES, action_hash, cavity_query, continue_cavity, run
from framework.polar_full_pattern import CountBank
from framework.polar_tractable import PolarModel, PolarScenario


def small_model():
    code = PolarCode(2, 4, 0)
    positions = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [0, 2, 4, 6]])
    return PolarModel(code, positions, np.full((4, 4), .5), 8)


def smoke_scenario():
    block = ODMAPolarBaseline(11, 128, 982051, prefix_bits=5, code_length=32, crc_bits=8, list_size=128)
    energy, _ = balanced_energy(block.positions, block.n)
    model = PolarModel(block.polar, block.positions, np.sqrt(energy), block.n)
    truth = Counter({209: 2, 1258: 1})
    variance = 1 / (2 * 11 * 10**1.5)
    y = np.random.default_rng(982052).normal(0, np.sqrt(variance), model.n)
    for word, count in truth.items():
        y += count * model.waveform(word)
    return PolarScenario(model, y, 3, variance), truth


class CountCavityTest(unittest.TestCase):
    def test_one_copy_restoration_removes_a_wrong_payload_ghost_exactly(self):
        model = small_model()
        u, w, companion = 0, 1, 8
        noise = np.linspace(-.07, .09, model.n)
        truth = Counter({u: 1, w: 1, companion: 1})
        action = Counter({u: 2, companion: 1})
        y = sum((count * model.waveform(word) for word, count in truth.items()), noise.copy())
        scenario = PolarScenario(model, y, 3, .08)
        residual = y - sum(count * model.waveform(word) for word, count in action.items())
        original = residual.copy()
        llr, meta = cavity_query(scenario, action, residual, 0, word=u)
        recovered = llr * meta["variance"] / (2 * model.amplitudes[0])
        np.testing.assert_allclose(recovered, (model.waveform(w) + noise)[model.positions[0]], atol=1e-15)
        self.assertEqual(meta["restored"], [(u, 1)])
        self.assertEqual(meta["variance"], scenario.noise_var)
        both_llr, both = cavity_query(scenario, action, residual, 0, restore_pattern=True)
        recovered_both = both_llr * both["variance"] / (2 * model.amplitudes[0])
        np.testing.assert_allclose(recovered_both, (model.waveform(u) + model.waveform(w) + noise)[model.positions[0]])
        self.assertEqual(both["restored_count"], 2)
        self.assertEqual(both["variance"], scenario.noise_var + 1 / model.n)
        np.testing.assert_array_equal(residual, original)
        self.assertEqual(action, Counter({u: 2, companion: 1}))

    def test_pending_count_and_exact_full_action_swap_ratio(self):
        model = small_model()
        scenario = PolarScenario(model, np.linspace(-.4, .8, model.n), 3, .12)
        action = Counter({0: 2})
        raw = scenario.y - 2 * model.waveform(0)
        for kwargs, pending in [({}, 1), ({"word": 0}, 2), ({"restore_pattern": True}, 3)]:
            _, meta = cavity_query(scenario, action, raw, 0, **kwargs)
            self.assertEqual(meta["pending"], pending)
            self.assertEqual(meta["variance"], scenario.noise_var + (pending - 1) / model.n)
        bank = CountBank(scenario, [0, 1, 8])
        initial = Counter({0: 2, 8: 1})
        after = Counter({0: 1, 1: 1, 8: 1})
        residual = scenario.y - 2 * model.waveform(0) - model.waveform(8)
        delta = model.waveform(1) - model.waveform(0)
        expected = (residual @ delta - delta @ delta / 2) / scenario.noise_var + np.log(2.)
        self.assertAlmostEqual(bank.objective(bank.vector(after)) - bank.objective(bank.vector(initial)), expected, places=12)
        with self.assertRaises(ValueError):
            cavity_query(scenario, action, raw, 0, word=1)
        with self.assertRaises(ValueError):
            cavity_query(scenario, action, raw, 0, word=0, restore_pattern=True)

    def test_physical_replacement_changes_later_queries_and_next_frozen_queue(self):
        code = PolarCode(2, 4, 0)
        positions = (2 * np.arange(8)[:, None] + np.arange(4)) % 16
        amplitudes = np.tile([.1, .7, .1, .7], (8, 1))
        model = PolarModel(code, positions, amplitudes, 16)
        wrong, target = 0, 2
        scenario = PolarScenario(model, model.waveform(target), 1, .01)
        initial = Counter({wrong: 1})
        action, bank, meta = continue_cavity(scenario, initial, [wrong])
        self.assertEqual(Counter(dict(meta["initial_refit_action"])), initial)
        self.assertEqual(action, Counter({target: 1}))
        self.assertTrue(set(range(4)).issubset(bank))
        first, second = meta["queries"][:2]
        self.assertEqual(first["restored"], [(wrong, 1)])
        llr, _ = cavity_query(scenario, initial, scenario.y - model.waveform(wrong), 0, word=wrong)
        self.assertEqual(first["llr_sha256"], hashlib.sha256(llr.tobytes()).hexdigest())
        self.assertEqual(second["pre_action_sha256"], action_hash(Counter({target: 1})))
        self.assertEqual(meta["sweeps"][0]["id_queue"], [wrong])
        self.assertEqual(meta["sweeps"][1]["id_queue"], [target])
        self.assertEqual(meta["stop"], "full_sweep_no_new_ids")

    def test_complete_paired_smoke_all_modes_and_query_replay(self):
        scenario, truth = smoke_scenario()
        base_action = base_bank = None
        for mode in MODES:
            counts, meta = run(scenario, mode=mode)
            self.assertEqual(Counter(counts), truth)
            continuation = meta["continuation"]
            self.assertTrue(continuation["complete"])
            if base_action is None:
                base_action, base_bank = meta["rollback_action"], meta["base_bank"]
            self.assertEqual(meta["rollback_action"], base_action)
            self.assertEqual(meta["base_bank"], base_bank)
            self.assertLessEqual(len(continuation["sweeps"]), 3)
            action = Counter(dict(continuation["initial_refit_action"]))
            for row in continuation["queries"]:
                self.assertEqual(row["pre_action_sha256"], action_hash(action))
                raw = scenario.y.copy()
                for word, count in action.items():
                    raw -= count * scenario.model.waveform(word)
                kwargs = {"word": row["restored"][0][0]} if row["context"] == "restore_one" else {}
                kwargs["restore_pattern"] = row["context"] == "restore_pattern"
                llr, context = cavity_query(scenario, action, raw, row["pattern"], **kwargs)
                for key in ("restored", "pending", "variance"):
                    self.assertEqual(context[key], row[key])
                # Sparse residual construction may round differently from dense independent replay.
                query_code = PolarCode(6, 32, 8, row["list_size"])
                payloads, _ = query_code.decode_list(llr)
                self.assertEqual([scenario.model.word(row["pattern"], p) for p in payloads], row["ids"])
                if row["refit_index"] is not None:
                    for move in continuation["refits"][row["refit_index"]]["meta"]["moves"]:
                        if move["old"] is not None:
                            action[move["old"]] -= 1
                            if not action[move["old"]]:
                                del action[move["old"]]
                        if move["new"] is not None:
                            action[move["new"]] += 1
            self.assertEqual(action, Counter(counts))
            for sweep in continuation["sweeps"]:
                rows = [r for r in continuation["queries"] if r["sweep"] == sweep["sweep"]]
                self.assertEqual(len(rows), sweep["queries"])
                self.assertLessEqual(len(rows), continuation["query_budget_per_sweep"])
                single = [r["restored"][0][0] for r in rows if r["context"] == "restore_one"]
                self.assertEqual(len(single), len(set(single)))
                if mode == "all_ids":
                    self.assertEqual(set(single) | set(sweep["withdrawn_ids_skipped"]), set(sweep["id_queue"]))
            self.assertGreaterEqual(continuation["other_seconds"], 0)
        self.assertEqual(Counter(dict(base_action)), truth)

    def test_invalid_phase_parameters(self):
        scenario, _ = smoke_scenario()
        for kwargs in ({"mode": "unknown"}, {"max_sweeps": 4}, {"max_moves": 0}):
            with self.assertRaises(ValueError):
                continue_cavity(scenario, Counter(), [], **kwargs)


if __name__ == "__main__":
    unittest.main()
