"""Local smoke or small learning grid for all new amplitude paths."""

import argparse
import json
import math
from pathlib import Path
import tempfile

import torch

from tests.framework_product_experiment import main as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", action="store_true", help="20 epochs instead of a one-batch execution check")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.out_dir or Path(tempfile.mkdtemp(prefix="ura-coordinate-mini-" if args.mini else "ura-coordinate-smoke-"))
    torch.set_num_threads(2)
    results = []
    for decoder in ("d0", "d1"):
        cases = [("coordinate", 4, True, False), ("coordinate", 4, True, True)]
        if not args.mini:
            cases += [("coordinate", 4, False, True), ("shared", 4, False, False),
                      ("shared", 4, True, False), ("coordinate", 8, False, True)]
        for mapping, J, balanced, learned in cases:
            name = f"{decoder}_{mapping}_J{J}_{'balanced' if balanced else 'raw'}_{'learned' if learned else 'fixed'}"
            out = root / name
            command = ["--encoder", "hash_coordinate", "--decoder", decoder, "-B", "8", "--n", "64",
                       "--sparse-support", "8", "--amplitude-label-bits", str(J), "--amplitude-map", mapping,
                       "--hash-search-candidates", "4", "--num-layers", "4", "--power-iters", "4",
                       "--k-min", "3", "--k-max", "6", "--eval-k", "3,6", "--eval-ebn0", "4,8",
                       "--encoder-epochs", "0", "--decoder-epochs", "20" if args.mini else "1",
                       "--batches-per-epoch", "10" if args.mini else "1", "--batch-size", "8",
                       "--validation-batches", "4", "--eval-batches", "4" if args.mini else "1",
                       "--seed", "31", "--train-seed", "100031", "--validation-seed", "200031",
                       "--eval-seed", "300031", "--out-dir", str(out)]
            if balanced: command += ["--amplitude-balance-init"]
            if learned: command += ["--learn-encoder", "--joint-train"]
            experiment(command)
            data = json.loads((out / "summary.json").read_text())
            losses = [p["validation_total"] for p in data["progress"]]
            assert losses and all(math.isfinite(x) for x in losses)
            assert data["metadata"]["amplitude_prototypes"]["max_energy_deviation"] < 1e-5
            stopping = data["metadata"]["early_stopping"][0]
            assert stopping["restored_best"]
            initial_loss = stopping["initial_validation_loss"]
            if args.mini:
                assert min(losses) < initial_loss - 1e-5, f"no learning in {name}"
            results.append({"name": name, "initial_validation": initial_loss, "best_validation": min(losses),
                            "mean_pupe": sum(r["pupe"] for r in data["learned"]) / len(data["learned"]),
                            "epochs": data["metadata"]["early_stopping"][0]["epochs_run"]})
    (root / "local_checks.json").write_text(json.dumps(results, indent=2))
    print(f"Passed {len(results)} local checks: {root}")


if __name__ == "__main__":
    main()
