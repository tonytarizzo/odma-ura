"""Job 032-matched D2 follow-up; never overwrite or mix with its historical results."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.ura_comparison import run_experiment


def resolve_config(row, selected):
    config = row.copy()
    if "selected_profile" in config:
        if selected.get("status") != "complete":
            raise ValueError("The completed Job 032 pilot selection is required")
        choice = selected["profiles"][config["selected_profile"]]
        config["profile_selection"] = choice
        config["baseline_params"] = {**choice["baseline_params"], **config.get("baseline_overrides", {})}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "1")))
    parser.add_argument("--out-root", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--selected", type=Path, default=ROOT / "jobs/032_published_baselines/results/pilot/selected.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(__file__).with_name("manifest.jsonl").read_text().splitlines()]
    if not 1 <= args.index <= len(rows): parser.error(f"index must be in 1..{len(rows)}")
    row = rows[args.index - 1]
    if "selected_profile" in row and not args.selected.is_file():
        parser.error(f"Missing completed pilot selection: {args.selected}")
    selected = json.loads(args.selected.read_text()) if "selected_profile" in row else {}
    config = resolve_config(row, selected)
    run_experiment(config, args.out_root / config["name"])


if __name__ == "__main__": main()
