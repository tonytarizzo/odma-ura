"""Run one manifest row with the common product-experiment training contract."""

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.framework_product_experiment import main as experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "1")))
    parser.add_argument("--out-root", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--diagnostic-scale", type=float, default=1.0, help="smoke override; production uses 1")
    args = parser.parse_args()
    with Path(__file__).with_name("manifest.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not 1 <= args.index <= len(rows):
        parser.error(f"index must lie in 1..{len(rows)}")
    row = rows[args.index - 1]; seed = int(row["seed"])
    command = ["--encoder", "hash_coordinate", "--decoder", row["decoder"], "-B", row["B"], "--n", row["n"],
               "--Q", "1", "--sparse-support", row["support"], "--amplitude-label-bits", row["J"],
               "--amplitude-map", row["mapping"], "--hash-search-candidates", row["search_candidates"],
               "--num-antennas", "1", "--num-layers", "8", "--power-iters", "12", "--encoder-epochs", "0",
               "--decoder-epochs", str(args.epochs), "--batches-per-epoch", str(args.batches_per_epoch),
               "--batch-size", "8", "--validation-batches", "8", "--early-stopping-patience", "5",
               "--early-stopping-min-delta", "0", "--train-ebn0-min", "-4", "--train-ebn0-max", "12",
               "--eval-ebn0=-4,0,4,8,12", "--eval-batches", str(args.eval_batches), "--extrapolate-k",
               "--diagnose-before-training", "--seed", str(seed), "--train-seed", str(seed + 100000),
               "--validation-seed", str(seed + 200000), "--eval-seed", str(seed + 300000),
               "--out-dir", str(args.out_root / row["name"])]
    for flag, count in (("pairs", 20000), ("active-samples", 256), ("active-gram-samples", 64), ("sum-pairs", 64)):
        command += [f"--diagnostic-{flag}", str(round(count * args.diagnostic_scale))]
    if row["initialization"] == "balanced":
        command += ["--amplitude-balance-init"]
    if row["mode"] == "learned":
        command += ["--learn-encoder", "--joint-train"]
    experiment(command)


if __name__ == "__main__":
    main()
