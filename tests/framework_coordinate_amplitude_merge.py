"""Strict job-031 audit, paired parent gaps, and amplitude-frontier plots."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(manifest: Path, results_root: Path) -> list[dict]:
    with manifest.open() as handle:
        expected = list(csv.DictReader(handle, delimiter="\t"))
    runs, missing, supports = [], [], {}
    for row in expected:
        path = results_root / row["name"]
        if not (path / "summary.json").exists() or not (path / "checkpoint.pt").exists():
            missing.append(row["name"]); continue
        data = json.loads((path / "summary.json").read_text())
        meta = data["metadata"]; args = meta["args"]
        learned, balanced = row["mode"] == "learned", row["initialization"] == "balanced"
        contract = {"encoder": "hash_coordinate", "decoder": row["decoder"], "payload_bits": int(row["B"]),
                    "n": int(row["n"]), "sparse_support": int(row["support"]), "amplitude_label_bits": int(row["J"]),
                    "amplitude_map": row["mapping"], "amplitude_balance_init": balanced, "amplitude_init": "gaussian",
                    "seed": int(row["seed"]), "hash_search_candidates": int(row["search_candidates"]),
                    "joint_train": learned, "learn_encoder": learned, "decoder_epochs": 120, "encoder_epochs": 0,
                    "num_layers": 8, "power_iters": 12, "num_antennas": 1, "batch_size": 8,
                    "batches_per_epoch": 100, "validation_batches": 8, "eval_batches": 16,
                    "early_stopping_patience": 5, "early_stopping_min_delta": 0, "lr": .001,
                    "train_ebn0_min": -4, "train_ebn0_max": 12, "lambda_count": .1, "lambda_symmetry": .01,
                    "diagnostic_pairs": 20000, "diagnostic_active_samples": 256,
                    "diagnostic_active_gram_samples": 64, "diagnostic_sum_pairs": 64, "diagnose_before_training": True,
                    "train_seed": int(row["seed"]) + 100000, "validation_seed": int(row["seed"]) + 200000,
                    "eval_seed": int(row["seed"]) + 300000}
        for key, value in contract.items():
            if args.get(key) != value:
                raise ValueError(f"{row['name']}: {key}={args.get(key)}; expected {value}")
        stopping = meta["early_stopping"]
        if len(stopping) != 1 or stopping[0]["patience"] != 5 or not stopping[0]["restored_best"]:
            raise ValueError(f"{row['name']}: invalid best-checkpoint contract")
        stop = stopping[0]
        if (len(data["progress"]) != stop["epochs_run"] or not 0 <= stop["best_epoch"] <= stop["epochs_run"]
                or (not stop["stopped_early"] and stop["epochs_run"] != 120)
                or (stop["stopped_early"] and not 5 <= stop["epochs_run"] <= 120)):
            raise ValueError(f"{row['name']}: training did not finish its declared stopping contract")
        if meta["K_train"] != [7, 22]:
            raise ValueError(f"{row['name']}: unexpected training loads")
        geometry = meta["amplitude_prototypes"]
        if not math.isfinite(geometry["max_energy_deviation"]) or geometry["max_energy_deviation"] > 1e-5:
            raise ValueError(f"{row['name']}: energy constraint failed")
        cells = {(r["K"], r["ebn0_db"]) for r in data["learned"]}
        if len(data["learned"]) != 20 or cells != {(k, snr) for k in (7, 15, 22, 26) for snr in (-4, 0, 4, 8, 12)}:
            raise ValueError(f"{row['name']}: incomplete or unexpected evaluation grid")
        if any(not math.isfinite(r["pupe"]) or not 0 <= r["pupe"] <= 1 for r in data["learned"]):
            raise ValueError(f"{row['name']}: nonfinite or invalid PUPE")
        if any(not math.isfinite(p["validation_total"]) for p in data["progress"]):
            raise ValueError(f"{row['name']}: nonfinite validation loss")
        initial = meta["amplitude_prototypes_initial"]
        if not learned and geometry != initial:
            raise ValueError(f"{row['name']}: fixed amplitude bank changed")
        construction = meta["codebook_construction"]
        support = (construction["A"], construction["b"])
        if supports.setdefault(row["seed"], support) != support:
            raise ValueError(f"{row['name']}: support differs between paired runs")
        if row["mapping"] == "coordinate":
            J, B = int(row["J"]), int(row["B"])
            r = (int(row["n"]) // int(row["support"])).bit_length() - 1
            if construction["stacked_projection_rank"] != B or any(x != min(B, r + J)
                                                                    for x in construction["local_joint_ranks"]):
                raise ValueError(f"{row['name']}: rank contract failed")
        runs.append({**row, "J": int(row["J"]), "seed": int(row["seed"]),
                     "high_snr_pupe": float(np.mean([r["pupe"] for r in data["learned"] if r["ebn0_db"] >= 8])),
                     "epochs": stopping[0]["epochs_run"], "best_epoch": stopping[0]["best_epoch"],
                     "amplitudes_initial": initial, "amplitudes_final": geometry,
                     "evaluation": data["learned"], "progress": data["progress"]})
    if missing:
        raise ValueError(f"missing {len(missing)}/{len(expected)} rows: {', '.join(missing[:8])}")
    return runs


def aggregate(runs):
    parents = {(r["decoder"], r["mode"], r["seed"]): r for r in runs if r["J"] == int(r["B"])}
    groups = defaultdict(list)
    for row in runs:
        parent = parents[(row["decoder"], row["mode"], row["seed"])]
        row["paired_parent_gap"] = row["high_snr_pupe"] - parent["high_snr_pupe"]
        groups[(row["decoder"], row["mode"], row["mapping"], row["initialization"], row["J"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        scores = np.array([r["high_snr_pupe"] for r in rows])
        gaps = np.array([r["paired_parent_gap"] for r in rows])
        output.append(dict(zip(("decoder", "mode", "mapping", "initialization", "J"), key)) | {
            "seeds": [r["seed"] for r in rows], "mean_pupe": float(scores.mean()),
            "seed_min": float(scores.min()), "seed_max": float(scores.max()),
            "paired_parent_gap": float(gaps.mean()), "paired_seed_gaps": gaps.tolist(),
            "gap_seed_standard_error": float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else None,
            "mean_epochs": float(np.mean([r["epochs"] for r in rows]))})
    return output


def plot(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, decoder in enumerate(("d0", "d1")):
        for mode, marker in (("fixed", "o"), ("learned", "s")):
            values = sorted([r for r in rows if r["decoder"] == decoder and r["mode"] == mode
                             and r["mapping"] == "coordinate" and (r["initialization"] == "balanced" or r["J"] == 14)],
                            key=lambda r: r["J"])
            x = [r["J"] for r in values]; y = np.array([r["mean_pupe"] for r in values])
            ranges = np.array([[r["seed_min"] for r in values], [r["seed_max"] for r in values]])
            axes[0, col].errorbar(x, y, yerr=np.stack((y - ranges[0], ranges[1] - y)), marker=marker,
                                  capsize=3, label=mode)
            axes[1, col].plot(x, [r["paired_parent_gap"] for r in values], marker=marker, label=mode)
        axes[0, col].set_title(decoder.upper()); axes[0, col].legend()
        axes[1, col].axhline(0, color="black", linewidth=.7)
        axes[1, col].set_xlabel("amplitude label bits J")
        for ax in axes[:, col]:
            ax.set_xticks([4, 8, 10, 14]); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Mean PUPE (8/12 dB); bars = seed range")
    axes[1, 0].set_ylabel("Paired PUPE gap to unrestricted J=14")
    fig.suptitle("Coordinate banks: balanced J<14; raw Gaussian unrestricted endpoint")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.manifest, args.results_root); rows = aggregate(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps({"runs": runs, "aggregate": rows}, indent=2))
    plot(rows, args.out_dir / "coordinate_frontier.png")
    print(f"Validated and merged {len(runs)} runs; inspect paired seed gaps and per-load results before conclusions.")


if __name__ == "__main__":
    main()
