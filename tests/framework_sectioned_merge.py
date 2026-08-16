"""Merge a sectioned-learning manifest result tree into compact comparison tables and plots."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", required=True)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def mean(values) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "merged"
    summaries = sorted(root.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"no summary.json files found below {root}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in summaries:
        payload = json.loads(path.read_text())
        name = re.sub(r"_s\d+$", "", path.parent.name)
        grouped[name].append(payload)
    rows = []
    for name, runs in sorted(grouped.items()):
        initial_bp = [mean(row["pupe"] for row in run["initial"]["rows"]) for run in runs]
        trained_bp = [mean(row["pupe"] for row in run["trained"]["rows"]) for run in runs]
        trained_d0 = [mean(row["d0_without_bp_pupe"] for row in run["trained"]["rows"]) for run in runs]
        rows.append({"name": name, "num_runs": len(runs), "initial_bp_pupe": mean(initial_bp),
                     "trained_bp_pupe": mean(trained_bp), "trained_d0_pupe": mean(trained_d0),
                     "bp_minus_d0": mean(trained_bp) - mean(trained_d0),
                     "trained_bp_seed_standard_error": float(np.std(trained_bp) / np.sqrt(len(trained_bp))),
                     "max_energy_deviation": max(run["metadata"]["sampled_energy_final"]["max_abs_unit_deviation"]
                                                 for run in runs)})
    rows.sort(key=lambda row: row["trained_bp_pupe"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "merged_summary.json").write_text(json.dumps({"result_root": str(root), "rows": rows}, indent=2))
    header = list(rows[0])
    lines = ["\t".join(header)] + ["\t".join(str(row[key]) for key in header) for row in rows]
    (out_dir / "merged_summary.tsv").write_text("\n".join(lines) + "\n")

    height = max(5.0, 0.42 * len(rows) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, height), sharey=True)
    y = np.arange(len(rows))
    labels = [row["name"] for row in rows]
    for axis, left_key, right_key, left_label, right_label, title in [
        (axes[0], "trained_d0_pupe", "trained_bp_pupe", "D0 + valid-path beam", "D0 + BP + beam",
         "Does outer BP improve complete-message recovery?"),
        (axes[1], "initial_bp_pupe", "trained_bp_pupe", "initial", "trained",
         "Does learning improve the complete decoder?")]:
        left = np.asarray([row[left_key] for row in rows])
        right = np.asarray([row[right_key] for row in rows])
        axis.hlines(y, np.minimum(left, right), np.maximum(left, right), color="0.65", linewidth=1.2)
        axis.scatter(left, y, label=left_label, color="#7dbbf2", s=42, zorder=3)
        axis.scatter(right, y, label=right_label, color="#ff9b52", s=42, zorder=3)
        axis.set(xlabel="mean PUPE over evaluated Eb/N0 points", title=title, xlim=(0.0, 1.0))
        axis.grid(axis="x", alpha=0.25); axis.legend(loc="lower right")
    axes[0].set_yticks(y, labels=labels); axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(out_dir / "decoder_comparison.png", dpi=180); plt.close(fig)
    print(f"merged {len(summaries)} runs into {len(rows)} configurations under {out_dir}")


if __name__ == "__main__":
    main()
