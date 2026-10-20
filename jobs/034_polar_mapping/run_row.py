"""Run one 034 row using the unchanged 032 pilot selection."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_mapping import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "1")))
    parser.add_argument("--out-root", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--selected", type=Path, default=ROOT / "jobs/032_published_baselines/results/pilot/selected.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(__file__).with_name("manifest.jsonl").read_text().splitlines()]
    if not 1 <= args.index <= len(rows): parser.error(f"index must be in 1..{len(rows)}")
    if rows[args.index - 1]["family"] == "polar_mixer":
        parser.error("Mixer training is retired. Historical rows 7-12 remain auditable; replay at revision 9d51cd8.")
    if not args.selected.is_file(): parser.error(f"Missing original pilot selection: {args.selected}")
    resolve = runpy.run_path(str(ROOT / "jobs/033_matched_d2/run_row.py"))["resolve_config"]
    config = resolve(rows[args.index - 1], json.loads(args.selected.read_text()))
    run_experiment(config, args.out_root / config["name"])


if __name__ == "__main__": main()
