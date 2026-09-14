"""Validate and merge job-030 fixed-versus-learned amplitude results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAUSSIAN_LABEL_BITS = (0, 2, 4, 8, 10, 12, 14)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def mean_pupe(rows: list[dict], high_snr_only: bool) -> float:
    selected = [float(row["pupe"]) for row in rows if not high_snr_only or float(row["ebn0_db"]) >= 8.0]
    if not selected:
        raise ValueError("no evaluation rows selected")
    return float(np.mean(selected))


def load_runs(manifest: Path, results_root: Path) -> list[dict]:
    with manifest.open(newline="") as handle:
        expected = list(csv.DictReader(handle, delimiter="\t"))
    runs, missing = [], []
    for row in expected:
        path = results_root / row["name"] / "summary.json"
        if not path.exists():
            missing.append(row["name"]); continue
        payload = json.loads(path.read_text())
        args = payload["metadata"]["args"]
        for key, expected_value in (("decoder", row["decoder"]), ("payload_bits", row["B"]), ("n", row["n"]),
                                    ("sparse_support", row["support"]), ("amplitude_label_bits", row["J"]),
                                    ("amplitude_init", row["amplitude_init"]), ("seed", row["seed"])):
            if str(args[key]) != str(expected_value):
                raise ValueError(f"{row['name']}: {key}={args[key]} does not match manifest {expected_value}")
        learned = row["mode"] == "learned"
        if bool(args["joint_train"]) != learned or bool(args["learn_encoder"]) != learned:
            raise ValueError(f"{row['name']}: amplitude mode flags disagree with manifest")
        stopping = payload["metadata"].get("early_stopping", [])
        if len(stopping) != 1 or stopping[0].get("patience") != 5 or not stopping[0].get("restored_best"):
            raise ValueError(f"{row['name']}: missing the required patience-5 best-checkpoint contract")
        diagnostic = payload["metadata"].get("amplitude_prototypes", {})
        if diagnostic.get("max_energy_deviation", math.inf) > 1e-5:
            raise ValueError(f"{row['name']}: unit-energy amplitude constraint failed")
        runs.append({**row, "J": int(row["J"]), "seed": int(row["seed"]),
                     "mean_pupe": mean_pupe(payload["learned"], False),
                     "mean_pupe_8_12db": mean_pupe(payload["learned"], True),
                     "epochs_run": int(stopping[0]["epochs_run"]),
                     "best_epoch": int(stopping[0]["best_epoch"]),
                     "stopped_early": bool(stopping[0]["stopped_early"]),
                     "prototype_parameters": int(diagnostic["prototype_parameters"])})
    if missing:
        raise SystemExit(f"missing {len(missing)}/{len(expected)} runs: {', '.join(missing[:8])}")
    return runs


def aggregate(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        groups[(row["decoder"], row["mode"], row["amplitude_init"], row["J"])].append(row)
    output = []
    for (decoder, mode, init, J), values in sorted(groups.items()):
        scores = np.array([row["mean_pupe_8_12db"] for row in values])
        epochs = np.array([row["epochs_run"] for row in values])
        output.append({"decoder": decoder, "mode": mode, "amplitude_init": init, "J": J,
                       "prototype_parameters": values[0]["prototype_parameters"], "num_seeds": len(values),
                       "mean_pupe_8_12db": float(scores.mean()),
                       "seed_standard_error": float(scores.std(ddof=0) / math.sqrt(len(scores))),
                       "mean_epochs_run": float(epochs.mean()),
                       "early_stop_fraction": float(np.mean([row["stopped_early"] for row in values]))})
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_frontier(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, decoder in zip(axes, ("d0", "d1")):
        for mode, marker in (("fixed", "o"), ("learned", "s")):
            values = sorted((row for row in rows if row["decoder"] == decoder and row["mode"] == mode
                             and row["amplitude_init"] == "gaussian"), key=lambda row: row["J"])
            ax.errorbar([row["J"] for row in values], [row["mean_pupe_8_12db"] for row in values],
                        yerr=[row["seed_standard_error"] for row in values], marker=marker, capsize=3, label=mode)
        for init, marker in (("equal", "X"), ("rademacher", "D")):
            value = next(row for row in rows if row["decoder"] == decoder and row["amplitude_init"] == init)
            ax.errorbar(value["J"], value["mean_pupe_8_12db"], yerr=value["seed_standard_error"],
                        marker=marker, capsize=3, linestyle="none", label=f"fixed {init}")
        ax.set(title=decoder.upper(), xlabel="amplitude-label bits J", xticks=GAUSSIAN_LABEL_BITS)
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    axes[0].set_ylabel("mean PUPE at 8 and 12 dB (lower is better)")
    fig.suptitle("Selected-hash amplitude model-class frontier, B=14, n=256, T=32")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_epochs(rows: list[dict], path: Path) -> None:
    values = [row for row in rows if row["amplitude_init"] == "gaussian"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for decoder, linestyle in (("d0", "-"), ("d1", "--")):
        for mode, marker in (("fixed", "o"), ("learned", "s")):
            selected = sorted((row for row in values if row["decoder"] == decoder and row["mode"] == mode),
                              key=lambda row: row["J"])
            ax.plot([row["J"] for row in selected], [row["mean_epochs_run"] for row in selected],
                    linestyle=linestyle, marker=marker, label=f"{decoder.upper()} {mode}")
    ax.set(xlabel="amplitude-label bits J", ylabel="mean epochs run", xticks=GAUSSIAN_LABEL_BITS, ylim=(0, 125))
    ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    args = parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.manifest, args.results_root)
    merged = aggregate(runs)
    write_csv(runs, args.out_dir / "runs.csv"); write_csv(merged, args.out_dir / "aggregate.csv")
    plot_frontier(merged, args.out_dir / "amplitude_frontier.png")
    plot_epochs(merged, args.out_dir / "early_stopping_epochs.png")
    comparisons = []
    for decoder in ("d0", "d1"):
        for J in GAUSSIAN_LABEL_BITS:
            fixed = next(row for row in merged if row["decoder"] == decoder and row["mode"] == "fixed"
                         and row["amplitude_init"] == "gaussian" and row["J"] == J)
            learned = next(row for row in merged if row["decoder"] == decoder and row["mode"] == "learned"
                           and row["amplitude_init"] == "gaussian" and row["J"] == J)
            comparisons.append({"decoder": decoder, "J": J,
                                "learned_minus_fixed_pupe": learned["mean_pupe_8_12db"] - fixed["mean_pupe_8_12db"]})
    (args.out_dir / "summary.json").write_text(json.dumps({"num_runs": len(runs), "comparisons": comparisons}, indent=2))
    print(f"merged {len(runs)} complete runs into {args.out_dir}")


if __name__ == "__main__":
    main()
