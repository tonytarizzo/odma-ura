"""Small local learning checks for Job 033; not evidence about comparative performance."""

import argparse
import json
from pathlib import Path

from benchmarks.ura_comparison import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    b = 8 if args.mini else 6
    shared = {"B": b, "n": 64, "seed": 3217, "loads": [2, 4, 7], "support": 8, "decoder": "d2",
              "hash_search_candidates": 2, "threads": 2, "layers": 4 if args.mini else 2,
              "max_epochs": 40 if args.mini else 2, "patience": 10, "batches_per_epoch": 12 if args.mini else 2,
              "batch_size": 8, "validation_batches": 8 if args.mini else 2, "eval_frames": 32 if args.mini else 4,
              "train_ebn0": [-4, 8], "eval_ebn0": [0, 8], "eval_sampling": ["distinct", "iid"]}
    variants = [("hash_full", b), ("hash_coordinate", 4)]
    if not args.mini: variants.append(("hash_coordinate", 2))
    configs = [{**shared, "family": family, "amplitude_label_bits": j, "mode": mode,
                "name": f"{family}_J{j}_{mode}"} for family, j in variants for mode in ("fixed", "joint")]
    if not args.mini:
        for family in ("odma_polar", "ccs_amp", "ccs_block"):
            params = {"prefix_bits": 3, "code_length": 32, "crc_bits": 8, "list_size": 32} if family == "odma_polar" else {}
            configs.append({**shared, "family": family, "mode": "fixed", "name": family, "baseline_params": params})
    checks = []
    for config in configs:
        result = run_experiment(config, args.out_dir / config["name"])
        training = result["training"]
        initial, final = training["initial"]["loss"], training["final"]["loss"]
        assert final <= initial + 1e-6 and training["early_stopping"]["restored_best"]
        if args.mini: assert final < initial - 1e-5, f"No learning in {config['name']}"
        energy = result["energy_audit_final"]
        if config["family"] != "ccs_amp":
            assert max(abs(energy[k] - 1) for k in ("min", "max")) < 1e-5
        checks.append({"name": config["name"], "initial_loss": initial, "final_loss": final,
                       "initial_validation_pupe": training["initial"]["pupe"],
                       "final_validation_pupe": training["final"]["pupe"],
                       "epochs": training["early_stopping"]["epochs_run"]})
    (args.out_dir / "learning_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__": main()
