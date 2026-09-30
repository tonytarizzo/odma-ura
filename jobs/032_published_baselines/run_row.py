"""Run exactly one array row; tuned profiles must exist before the matched stage."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.ura_comparison import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "comparison", "native", "checks"), default=os.environ.get("URA_PHASE", "pilot"))
    parser.add_argument("--index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "1")))
    parser.add_argument("--selected", type=Path, default=Path(__file__).with_name("results") / "pilot" / "selected.json")
    parser.add_argument("--out-root", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--smoke", action="store_true", help="B=6 execution check, never production data")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(__file__).with_name(f"{args.phase}.jsonl").read_text().splitlines()]
    if not 1 <= args.index <= len(rows): parser.error(f"index must be in 1..{len(rows)}")
    config = rows[args.index - 1].copy()
    if args.smoke:
        config.pop("selected_profile", None)
        config.pop("paper_reference", None)
        config["purpose"] = "small-B execution smoke only; not production or paper-alignment evidence"
        config.update(B=6, n=64, loads=[1, 2], layers=2, hidden_dim=8, candidate_size=16, max_epochs=2,
                      batches_per_epoch=2, batch_size=4, validation_batches=2, eval_frames=4,
                      eval_ebn0=[0, 8], eval_sampling=["distinct", "iid"], support=8, threads=2)
        config["baseline_params"] = ({"prefix_bits": 3, "code_length": 32, "crc_bits": 8, "list_size": 32}
                                     if config["family"] == "odma_polar" else {})
        args.out_root = args.out_root / "smoke"
    elif "selected_profile" in config:
        if not args.selected.is_file():
            parser.error(f"Missing {args.selected}; finish/merge the pilot first")
        selected = json.loads(args.selected.read_text())
        if selected.get("status") != "complete": parser.error("Pilot selection is incomplete")
        choice = selected["profiles"][config["selected_profile"]]
        config["baseline_params"] = choice["baseline_params"]
        config["profile_selection"] = choice
    # The follow-up array is restartable; old pilot/native/comparison outputs remain untouched.
    run_experiment(config, args.out_root / args.phase / config["name"],
                   native_checkpoint=args.phase == "checks", resume=args.phase == "checks")


if __name__ == "__main__": main()
