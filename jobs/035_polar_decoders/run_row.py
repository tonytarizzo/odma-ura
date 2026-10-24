"""Execute a bundle sequentially; resume matching complete runs without overwriting partial work."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import normalized_config, run_experiment, source_hashes


def run_bundle(row, out_root, *, mapping="all"):
    runs = [config for config in row["runs"] if mapping == "all" or config["mapping"] == mapping]
    if not runs:
        raise ValueError(f"No {mapping} runs in bundle {row['name']}")
    for config in runs:
        out = Path(out_root) / config["name"]
        summary = out / "summary.json"
        if summary.is_file():
            saved = json.loads(summary.read_text())
            if (saved["status"] != "complete" or saved["config"] != normalized_config(config)
                    or saved["source_sha256"] != source_hashes()):
                raise ValueError(f"Existing run differs from this bundle/config/source: {out}")
            print(f"Already complete, skipping: {out}", flush=True)
            continue
        print(f"Running {row['name']}: {config['name']}", flush=True)
        run_experiment(config, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "1")))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    parser.add_argument("--out-root", type=Path, default=Path(__file__).with_name("results_all_decoders"))
    parser.add_argument("--mapping", choices=["all", "balanced", "uniform"], default="all")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if not 1 <= args.index <= len(rows):
        parser.error(f"index must be in 1..{len(rows)}")
    run_bundle(rows[args.index - 1], args.out_root, mapping=args.mapping)


if __name__ == "__main__":
    main()
