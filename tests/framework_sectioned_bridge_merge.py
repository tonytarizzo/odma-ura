"""Merge jobs 025/026 into paired tables and a common PUPE landscape."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l1-root", required=True)
    p.add_argument("--lgt-root", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def label_for(metadata: dict, decoder: str) -> str | None:
    args = metadata["args"]
    if args["bridge"] == "l1":
        family = {"dense_fixed": "dense", "sparse_global_fixed": "sparse global", "odma_fixed": "ODMA"}[args["encoder"]]
        if decoder == "global_bernoulli": return f"{family}: global D0"
        if decoder == "section_binomial": return f"{family}: L1 Binomial D0"
        return None
    family = f"{args['outer_code']} {'learned' if args['learn_encoder'] else 'fixed'}"
    suffix = {"materialised_global_d0": "induced global D0", "local_d0_association": "local D0 + association",
              "local_d0_bp_association": "local D0 + BP + association"}[decoder]
    return f"{family}: {suffix}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = sorted(Path(args.l1_root).glob("*/summary.json")) + sorted(Path(args.lgt_root).glob("*/summary.json"))
    if not paths:
        raise SystemExit("no bridge summary.json files found")
    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    equivalence = []
    for path in paths:
        payload = json.loads(path.read_text()); metadata = payload["metadata"]
        if metadata["args"]["bridge"] == "l1":
            equivalence.append({"run": path.parent.name, **metadata["compatibility_max_abs_difference"]})
        for row in payload["rows"]:
            label = label_for(metadata, row["decoder"])
            if label is not None: grouped[(label, int(row["K"]), float(row["ebn0_db"]))].append(float(row["pupe"]))
    rows = []
    for (label, K, ebn0), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        rows.append({"label": label, "K": K, "ebn0_db": ebn0, "num_seeds": len(values),
                     "mean_pupe": float(np.mean(values)),
                     "seed_standard_error": float(np.std(values) / math.sqrt(len(values)))})
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bridge_summary.json").write_text(json.dumps({"num_runs": len(paths), "rows": rows,
                                                               "l1_equivalence": equivalence}, indent=2))
    with (out_dir / "bridge_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t"); writer.writeheader(); writer.writerows(rows)

    loads = sorted({row["K"] for row in rows})
    fig, axes = plt.subplots(2, math.ceil(len(loads) / 2), figsize=(15, 9), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    labels = sorted({row["label"] for row in rows})
    for axis, K in zip(axes, loads):
        for label in labels:
            selected = [row for row in rows if row["K"] == K and row["label"] == label]
            if selected:
                axis.plot([row["ebn0_db"] for row in selected], [row["mean_pupe"] for row in selected], marker="o", label=label)
        axis.set(title=f"B=12, n=256, K={K}", xlabel="$E_b/N_0$ (dB)", ylabel="PUPE", ylim=(0.0, 1.0))
        axis.grid(alpha=0.25)
    for axis in axes[len(loads):]: axis.set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(out_dir / "bridge_pupe_landscape.png", dpi=180); plt.close(fig)
    print(f"merged {len(paths)} bridge runs into {out_dir}")


if __name__ == "__main__":
    main()
