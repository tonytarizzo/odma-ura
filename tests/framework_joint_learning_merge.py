"""Validate and compare the focused joint-learning batch with optional fixed-encoder job-028 controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_LABEL = {"dense_fixed": "dense", "sparse_iid_fixed": "iid sparse", "hash_linear_selected_fixed": "selected hash"}
FAMILY_ORDER = list(FAMILY_LABEL)
DECODER_COLOUR = {"d0": "#76b7f2", "d1": "#f59d56"}


def parse_float_grid(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-root", required=True)
    parser.add_argument("--joint-manifest", required=True)
    parser.add_argument("--fixed-root")
    parser.add_argument("--fixed-manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--high-snr", type=parse_float_grid, default=parse_float_grid("8,12"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def manifest_names(path: str | None, allowed: set[str] | None = None) -> set[str]:
    if path is None:
        return set()
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["name"] for row in rows if allowed is None or row["encoder"] in allowed}


def standard_error(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_mode(root: str, mode: str, expected: set[str], high_snr: list[float], allow_incomplete: bool) -> tuple[list[dict], list[str]]:
    paths = sorted(Path(root).glob("*/summary.json")); seen, rows = set(), []
    for path in paths:
        name = path.parent.name
        payload = json.loads(path.read_text()); metadata = payload["metadata"]; run_args = metadata["args"]
        encoder = run_args["encoder"]
        if encoder not in FAMILY_LABEL:
            continue
        if name in seen:
            raise ValueError(f"duplicate result {name}")
        seen.add(name)
        if mode == "joint" and not (run_args.get("learn_encoder") and run_args.get("joint_train")):
            raise ValueError(f"{path} is not a joint encoder/decoder run")
        final_geometry = metadata.get("codebook_sparsity")
        if final_geometry is None:
            raise ValueError(f"{path} has no final codebook diagnostics")
        initial_geometry = metadata.get("codebook_sparsity_initial")
        if mode == "joint" and initial_geometry is None:
            raise ValueError(f"{path} has no pre-training codebook diagnostics")
        high = [row for row in payload["learned"]
                if any(abs(float(row["ebn0_db"]) - target) < 1e-8 for target in high_snr)]
        expected_cells = len(metadata["K_eval"]) * len(high_snr)
        if len(high) != expected_cells:
            raise ValueError(f"{path} has {len(high)}/{expected_cells} requested high-SNR cells")
        losses = [float(item["total"]) for item in payload["progress"]]
        tail = losses[-min(10, len(losses)):]
        common = {
            "run": name, "mode": mode, "encoder": encoder, "family": FAMILY_LABEL[encoder],
            "decoder": run_args["decoder"], "B": int(run_args["payload_bits"]), "n": int(run_args["n"]),
            "support_size": int(final_geometry["support_size"]), "seed": int(run_args["seed"]),
            "mean_high_snr_pupe": float(np.mean([float(row["pupe"]) for row in high])),
            "epochs": len(losses), "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
            "tail_loss_change": tail[-1] - tail[0] if len(tail) > 1 else 0.0,
            "loss_curve": losses,
        }
        for metric in ("max_unit_energy_deviation", "row_energy_cv", "sampled_abs_correlation_q999",
                       "active_gram_min_eigenvalue_q05", "normalised_disjoint_K_sum_distance_q05"):
            common[f"final_{metric}"] = final_geometry.get(metric)
            common[f"initial_{metric}"] = initial_geometry.get(metric) if initial_geometry else None
        rows.append(common)
    missing = sorted(expected - seen); unexpected = sorted(seen - expected) if expected else []
    if (missing or unexpected) and not allow_incomplete:
        raise SystemExit(f"{mode} manifest/result mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    return rows, missing + [f"unexpected:{name}" for name in unexpected]


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["encoder"], row["decoder"], row["support_size"])].append(row)
    output = []
    for (mode, encoder, decoder, support), values in grouped.items():
        pupes = [row["mean_high_snr_pupe"] for row in values]
        output.append({
            "mode": mode, "encoder": encoder, "family": FAMILY_LABEL[encoder], "decoder": decoder,
            "support_size": support, "num_seeds": len(values), "mean_high_snr_pupe": float(np.mean(pupes)),
            "seed_standard_error": standard_error(pupes),
            "mean_tail_loss_change": float(np.mean([row["tail_loss_change"] for row in values])),
        })
    return sorted(output, key=lambda row: (row["support_size"], FAMILY_ORDER.index(row["encoder"]), row["decoder"], row["mode"]))


def paired_deltas(rows: list[dict]) -> list[dict]:
    fixed = {(row["encoder"], row["decoder"], row["support_size"], row["seed"]): row
             for row in rows if row["mode"] == "fixed"}
    output = []
    for joint in (row for row in rows if row["mode"] == "joint"):
        key = (joint["encoder"], joint["decoder"], joint["support_size"], joint["seed"])
        if key not in fixed:
            continue
        control = fixed[key]
        for metric in ("max_unit_energy_deviation", "row_energy_cv", "sampled_abs_correlation_q999",
                       "active_gram_min_eigenvalue_q05", "normalised_disjoint_K_sum_distance_q05"):
            initial, fixed_value = joint[f"initial_{metric}"], control[f"final_{metric}"]
            if initial is not None and fixed_value is not None and not np.isclose(initial, fixed_value, rtol=1e-7, atol=1e-9):
                raise ValueError(f"joint initial geometry does not match fixed control for {key}, metric={metric}")
        output.append({
            "encoder": joint["encoder"], "family": joint["family"], "decoder": joint["decoder"],
            "support_size": joint["support_size"], "seed": joint["seed"],
            "fixed_pupe": control["mean_high_snr_pupe"], "joint_pupe": joint["mean_high_snr_pupe"],
            "joint_minus_fixed_pupe": joint["mean_high_snr_pupe"] - control["mean_high_snr_pupe"],
        })
    return output


def plot_performance(rows: list[dict], out_dir: Path, high_snr: list[float]) -> None:
    sparse_supports = sorted({row["support_size"] for row in rows if row["encoder"] != "dense_fixed"})
    fig, axes = plt.subplots(1, len(sparse_supports), figsize=(6.7 * len(sparse_supports), 5.2), sharey=True, squeeze=False)
    for axis, support in zip(axes.reshape(-1), sparse_supports):
        selected = [row for row in rows if row["support_size"] == support or row["encoder"] == "dense_fixed"]
        for decoder_index, decoder in enumerate(("d0", "d1")):
            for mode_index, mode in enumerate(("fixed", "joint")):
                available = {row["encoder"]: row for row in selected if row["decoder"] == decoder and row["mode"] == mode}
                points = [(FAMILY_ORDER.index(family), available[family]) for family in FAMILY_ORDER if family in available]
                if not points:
                    continue
                offset = (-0.12 if decoder_index == 0 else 0.12) + (-0.035 if mode_index == 0 else 0.035)
                axis.errorbar([x + offset for x, _ in points], [row["mean_high_snr_pupe"] for _, row in points],
                              yerr=[row["seed_standard_error"] for _, row in points], color=DECODER_COLOUR[decoder],
                              marker="o", markerfacecolor="none" if mode == "fixed" else DECODER_COLOUR[decoder],
                              linestyle="none", capsize=3, label=f"{decoder.upper()} {mode}")
        axis.set_xticks(range(len(FAMILY_ORDER)), [FAMILY_LABEL[name] for name in FAMILY_ORDER], rotation=18, ha="right")
        axis.set(title=f"support T={support}", ylabel="mean PUPE", ylim=(0, 1)); axis.grid(axis="y", alpha=0.22)
    axes.reshape(-1)[0].legend()
    snr = " and ".join(f"{value:g}" for value in high_snr)
    fig.suptitle(f"Fixed versus jointly learned encoding: mean PUPE over {snr} dB and all loads")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out_dir / "joint_vs_fixed_pupe.png", dpi=200); plt.close(fig)


def plot_training(rows: list[dict], out_dir: Path) -> None:
    joint = [row for row in rows if row["mode"] == "joint"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, decoder in zip(axes, ("d0", "d1")):
        grouped = defaultdict(list)
        for row in joint:
            if row["decoder"] == decoder:
                grouped[(row["encoder"], row["support_size"])].append(row["loss_curve"])
        for (encoder, support), curves in grouped.items():
            values = np.asarray(curves, dtype=float)
            label = FAMILY_LABEL[encoder] if encoder == "dense_fixed" else f"{FAMILY_LABEL[encoder]}, T={support}"
            axis.plot(np.arange(1, values.shape[1] + 1), values.mean(axis=0), label=label)
        axis.set(title=decoder.upper(), xlabel="joint-training epoch", ylabel="mean training loss"); axis.grid(alpha=0.22)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle("Joint encoder-decoder optimisation traces (mean over seeds)")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out_dir / "joint_training_curves.png", dpi=200); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    allowed = set(FAMILY_LABEL)
    joint, notes = load_mode(args.joint_root, "joint", manifest_names(args.joint_manifest), args.high_snr, args.allow_incomplete)
    rows = list(joint)
    if args.fixed_root:
        fixed_expected = manifest_names(args.fixed_manifest, allowed) if args.fixed_manifest else set()
        fixed, fixed_notes = load_mode(args.fixed_root, "fixed", fixed_expected, args.high_snr, args.allow_incomplete)
        rows.extend(fixed); notes.extend(fixed_notes)
    aggregate_rows = aggregate(rows)
    delta_rows = paired_deltas(rows)
    serialisable = [{key: value for key, value in row.items() if key != "loss_curve"} for row in rows]
    (out_dir / "joint_learning_summary.json").write_text(json.dumps({
        "high_snr_db": args.high_snr, "completeness_notes": notes, "aggregate": aggregate_rows,
        "joint_minus_fixed": delta_rows, "runs": rows,
    }, indent=2))
    write_tsv(out_dir / "joint_learning_aggregate.tsv", aggregate_rows)
    write_tsv(out_dir / "joint_minus_fixed.tsv", delta_rows)
    write_tsv(out_dir / "joint_learning_runs.tsv", serialisable)
    plot_performance(aggregate_rows, out_dir, args.high_snr); plot_training(rows, out_dir)
    print(f"merged {len(rows)} fixed/joint runs into {out_dir}; completeness notes={len(notes)}")


if __name__ == "__main__":
    main()
