"""Local execution and initial-learning checks; no production performance claims."""

import argparse
import json
from pathlib import Path

from benchmarks.polar_mapping import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--production-shape", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    b = 14 if args.production_shape else (8 if args.mini else 6)
    shared = {"B": b, "n": 256 if args.production_shape else 64, "seed": 3217,
              "loads": [7, 15, 22, 26] if args.production_shape else [2, 4, 7], "decoder": "d2", "threads": 2,
              "layers": 8 if args.production_shape else (4 if args.mini else 2),
              "max_epochs": 40 if args.mini else 2, "patience": 10, "batches_per_epoch": 12 if args.mini else 2,
              "batch_size": 8, "validation_batches": 8 if args.mini else 2, "eval_frames": 32 if args.mini else 4,
              "train_ebn0": [-4, 10], "eval_ebn0": [0, 8], "eval_sampling": ["distinct", "iid"],
              "baseline_params": {"prefix_bits": 8 if args.production_shape else b // 2,
                                  "code_length": 32, "crc_bits": 8, "list_size": 8}}
    checks = []
    for family in ("odma_polar", "polar_energy"):
        config = {**shared, "family": family, "mode": "fixed" if family == "odma_polar" else "joint", "name": family}
        result = run_experiment(config, args.out_dir / family)
        training = result["training"]
        initial, final = training["initial"]["loss"], training["final"]["loss"]
        assert final <= initial + 1e-6 and training["early_stopping"]["restored_best"]
        if args.mini: assert final < initial - 1e-5, f"No initial learning in {family}"
        checks.append({"family": family, "initial_loss": initial, "final_loss": final,
                       "initial_pupe": training["initial"]["pupe"], "final_pupe": training["final"]["pupe"],
                       "epochs": training["early_stopping"]["epochs_run"], "energy": result["energy_audit_final"],
                       "mapping": result["mapping_diagnostics_final"]})
    (args.out_dir / "learning_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__": main()
