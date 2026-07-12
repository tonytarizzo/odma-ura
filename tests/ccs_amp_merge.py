"""Merge per-load CCS-AMP author-code jobs into one validation graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.ccs_amp_author_curve import parse_args as curve_args, plot
from tests.ccs_amp_author import preset_parameters


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--preset", choices=["paper_b128", "adapted_b100"], default="paper_b128")
    args = p.parse_args(argv)
    summaries = []
    for path in sorted(args.input_dir.glob("K*/ccs_amp_summary.json")):
        summaries.append(json.loads(path.read_text()))
    if not summaries:
        raise SystemExit(f"no K*/ccs_amp_summary.json files below {args.input_dir}")
    rows = [row for summary in summaries for row in summary["rows"]]
    points = [row for summary in summaries for row in summary["points"]]
    required = [row for summary in summaries for row in summary["required"]]
    bounds_by_key = {}
    for summary in summaries:
        for row in summary["polyanskiy"]:
            bounds_by_key[(row["K"], row["variant"])] = row
    bounds = list(bounds_by_key.values())
    first_args = summaries[0]["args"]
    plotting_args = curve_args([
        "--preset", args.preset, "--K-values", *[str(k) for k in sorted({r["K"] for r in points})],
        "--ebn0-grid", *[str(e) for e in sorted({r["ebn0_db"] for r in points})],
        "--num-seeds", str(first_args["num_seeds"]), "--schemes", *sorted({r["scheme"] for r in points}),
        "--out-dir", str(args.out_dir),
    ])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot(points, required, bounds, plotting_args, preset_parameters(args.preset))
    merged = {"preset": args.preset, "sources": len(summaries), "rows": rows, "points": points,
              "required": required, "polyanskiy": bounds, "paper_enhanced": summaries[0]["paper_enhanced"],
              "paper_original": summaries[0]["paper_original"], "author_commit": summaries[0]["author_commit"]}
    (args.out_dir / "ccs_amp_merged_summary.json").write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(summaries)} load jobs into {args.out_dir}")


if __name__ == "__main__":
    main()

