"""Laptop end-to-end smoke/mini suite; writes only to the requested result folder."""

import argparse
import json
from pathlib import Path

from benchmarks.ura_comparison import run_experiment
from benchmarks.ura_merge import aggregate, plot_learning, plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "mini"), default="smoke")
    parser.add_argument("--joint-ladder-only", action="store_true", help="Check the six newly added joint D2/D3/D4 paths")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    mini = args.preset == "mini"
    shared = {"B": 6, "n": 64, "seed": 3217, "loads": [1, 2, 3], "layers": 4 if mini else 2,
              "hidden_dim": 16 if mini else 8, "power_iters": 8, "candidate_size": 32,
              "max_epochs": 200 if mini else 2, "batches_per_epoch": 12 if mini else 2, "batch_size": 8,
              "validation_batches": 8 if mini else 2, "eval_frames": 32 if mini else 4,
              "eval_ebn0": [-4, 0, 4, 8] if mini else [0, 8], "support": 8, "threads": 2}
    runs = []
    for family in ("odma_polar", "dynamic_cs", "ccs_amp", "ccs_block"):
        params = {"prefix_bits": 3, "code_length": 32, "crc_bits": 8, "list_size": 32} if family == "odma_polar" else {}
        for decoder in ("native", "d0"):
            runs.append({**shared, "family": family, "decoder": decoder, "baseline_params": params,
                         "mode": "native" if decoder == "native" else "fixed"})
    for family in ("dense", "sparse"):
        for decoder in ("d0", "d1", "d2", "d3", "d4"):
            runs.append({**shared, "family": family, "mode": "fixed", "decoder": decoder})
        for decoder in ("d0", "d1", "d2", "d3", "d4"):
            runs.append({**shared, "family": family, "mode": "joint", "decoder": decoder})
    if args.joint_ladder_only:
        runs = [c for c in runs if c["mode"] == "joint" and c["decoder"] in {"d2", "d3", "d4"}]
    results = []
    for config in runs:
        config["name"] = f"{config['family']}_{config['mode']}_{config['decoder']}"
        results.append(run_experiment(config, args.out_dir / config["name"]))
    outcomes = []
    for result in results:
        training = result["training"]
        if training is not None:
            initial, final = training["initial"]["loss"], training["final"]["loss"]
            if final > initial + 1e-6:
                raise AssertionError("Best-state restoration made deterministic validation worse")
            outcomes.append({"name": result["config"]["name"], "initial_loss": initial, "best_loss": final,
                             "epochs": training["early_stopping"]["epochs_run"],
                             "initial_validation_pupe": training["initial"]["pupe"],
                             "best_validation_pupe": training["final"]["pupe"]})
    (args.out_dir / "learning_check.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    (args.out_dir / "manifest.jsonl").write_text("".join(json.dumps(r["config"]) + "\n" for r in results))
    plots(aggregate(results), args.out_dir)
    plot_learning(results, args.out_dir)
    print(json.dumps(outcomes, indent=2))
    print("Execution/learning check only; no published-performance claim.")


if __name__ == "__main__": main()
