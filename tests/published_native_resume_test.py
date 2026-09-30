"""Native frame checkpoints must reproduce the uninterrupted experiment and reject drift."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.ura_comparison import run_experiment
from baselines.odma_polar import ODMAPolarBaseline


class NativeResumeTests(unittest.TestCase):
    def test_interrupt_resume_and_provenance_guard(self):
        config = {"B": 6, "n": 64, "seed": 73, "family": "odma_polar", "mode": "native", "decoder": "native",
                  "loads": [1], "eval_ebn0": [0, 4], "eval_sampling": ["iid"], "eval_frames": 3, "batch_size": 2,
                  "baseline_params": {"prefix_bits": 3, "code_length": 32, "crc_bits": 8, "list_size": 8}}
        original = ODMAPolarBaseline.decode
        calls = 0

        def interrupted(self, *args):
            nonlocal calls
            calls += 1
            if calls == 5: raise RuntimeError("test interruption")
            return original(self, *args)

        with TemporaryDirectory() as directory:
            target, reference = Path(directory)/"resumed", Path(directory)/"reference"
            with patch.object(ODMAPolarBaseline, "decode", interrupted):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    run_experiment(config, target, native_checkpoint=True)
            self.assertEqual(len(json.loads((target/"native_progress.json").read_text())), 4)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                run_experiment({**config, "seed": 74}, target, native_checkpoint=True, resume=True)
            run_experiment(config, target, native_checkpoint=True, resume=True)
            run_experiment(config, reference, native_checkpoint=True)
            resumed = json.loads((target/"summary.json").read_text())
            full = json.loads((reference/"summary.json").read_text())
            for a, b in zip(resumed["evaluation"], full["evaluation"]):
                self.assertEqual(a["frame_metrics"], b["frame_metrics"])
                self.assertEqual(a["means"], b["means"])
            self.assertEqual(len((target/"evaluation.jsonl").read_text().splitlines()), 2)
            run_experiment(config, target, native_checkpoint=True, resume=True)
            state = json.loads((target/"run_state.json").read_text())
            state["source_sha256"]["test"] = "changed"
            (target/"run_state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "different configuration or source"):
                run_experiment(config, target, native_checkpoint=True, resume=True)


if __name__ == "__main__": unittest.main()
