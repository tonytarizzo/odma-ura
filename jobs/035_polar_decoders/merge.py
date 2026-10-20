"""Merge complete comparable B/mapping panels; reject missing or duplicate rows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import merge_runs, normalized_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results_all_decoders"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    bundles = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    rows = [run for bundle in bundles for run in bundle["runs"]]
    missing = [r["name"] for r in rows if not (args.results / r["name"] / "summary.json").is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} rows: {missing}")
    for row in rows:
        summary = json.loads((args.results / row["name"] / "summary.json").read_text())
        if summary["config"] != normalized_config(row):
            raise ValueError(f"Saved configuration differs from the manifest: {row['name']}")
    for b, mapping in sorted({(r["B"], r["mapping"]) for r in rows}):
        paths = [args.results / r["name"] for r in rows if (r["B"], r["mapping"]) == (b, mapping)]
        out = (args.out or args.results / "analysis") / f"B{b}_{mapping}"
        merge_runs(paths, out)
        print(out)


if __name__ == "__main__":
    main()
