"""Validate, merge, and plot the controlled B=14 hash-skeleton experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_LABEL = {
    "dense_fixed": "dense",
    "sparse_iid_fixed": "iid sparse",
    "hash_table_random_fixed": "balanced tables",
    "hash_linear_random_fixed": "linear hash",
    "hash_linear_selected_fixed": "selected linear hash",
}
FAMILY_ORDER = list(FAMILY_LABEL)
DECODER_STYLE = {"d0": ("D0", "#76b7f2"), "d1": ("D1", "#f59d56")}


def parse_float_grid(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", action="append", required=True)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--high-snr", type=parse_float_grid, default=parse_float_grid("8,12"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def expected_names(paths: list[str]) -> set[str]:
    names = set()
    for path in paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["name"] in names:
                    raise ValueError(f"duplicate manifest name {row['name']}")
                names.add(row["name"])
    return names


def standard_error(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_runs(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    paths = [path for root in args.result_root for path in sorted(Path(root).glob("*/summary.json"))]
    if not paths:
        raise SystemExit("no */summary.json files found")
    seen, runs, diagnostics, constructions, paired_matched = set(), [], [], [], {}
    for path in paths:
        name = path.parent.name
        if name in seen:
            raise ValueError(f"duplicate result run {name}")
        seen.add(name)
        payload = json.loads(path.read_text()); metadata = payload["metadata"]; run_args = metadata["args"]
        encoder = run_args["encoder"]
        if encoder not in FAMILY_LABEL:
            raise ValueError(f"unexpected encoder {encoder} in {path}")
        geometry = metadata.get("codebook_sparsity")
        if geometry is None:
            raise ValueError(f"{path} has no codebook_sparsity diagnostics")
        high = [row for row in payload["learned"]
                if any(abs(float(row["ebn0_db"]) - target) < 1e-8 for target in args.high_snr)]
        high_matched = [row for row in payload["matched_filter"]
                        if any(abs(float(row["ebn0_db"]) - target) < 1e-8 for target in args.high_snr)]
        expected_high = len(metadata["K_eval"]) * len(args.high_snr)
        if len(high) != expected_high or len(high_matched) != expected_high:
            raise ValueError(f"{path} has incomplete requested high-SNR cells")
        common = {
            "run": name, "encoder": encoder, "family": FAMILY_LABEL[encoder], "decoder": run_args["decoder"],
            "B": int(run_args["payload_bits"]), "n": int(run_args["n"]), "support_size": int(geometry["support_size"]),
            "seed": int(run_args["seed"]), "num_high_snr_cells": len(high),
        }
        runs.append({**common, "mean_high_snr_pupe": float(np.mean([float(row["pupe"]) for row in high]))})
        paired_key = (encoder, common["support_size"], common["seed"])
        if paired_key in paired_matched and paired_matched[paired_key] != payload["matched_filter"]:
            raise ValueError(f"paired D0/D1 matched-filter streams differ for {paired_key}")
        paired_matched[paired_key] = payload["matched_filter"]
        diagnostics.append({**common, **geometry})
        construction = metadata.get("codebook_construction")
        if encoder != "dense_fixed" and construction is None:
            raise ValueError(f"{path} has no hash-skeleton construction metadata")
        if construction is not None:
            constructions.append({**common, **construction})
    expected = expected_names(args.manifest)
    missing = sorted(expected - seen); unexpected = sorted(seen - expected) if expected else []
    if (missing or unexpected) and not args.allow_incomplete:
        raise SystemExit(f"manifest/result mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    notes = missing + [f"unexpected:{name}" for name in unexpected]
    return runs, diagnostics, constructions, notes


def aggregate_runs(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["B"], row["n"], row["encoder"], row["decoder"], row["support_size"])].append(row)
    aggregate = []
    for (B, n, encoder, decoder, support), values in grouped.items():
        pupe = [row["mean_high_snr_pupe"] for row in values]
        aggregate.append({
            "B": B, "n": n, "encoder": encoder, "family": FAMILY_LABEL[encoder], "decoder": decoder,
            "support_size": support,
            "num_seeds": len(values), "mean_high_snr_pupe": float(np.mean(pupe)), "seed_standard_error": standard_error(pupe),
        })
    return sorted(aggregate, key=lambda row: (FAMILY_ORDER.index(row["encoder"]), row["support_size"], row["decoder"]))


def unique_paired(rows: list[dict], value_name: str) -> list[dict]:
    unique = {}
    for row in rows:
        key = (row["encoder"], row["support_size"], row["seed"])
        reduced = {name: value for name, value in row.items() if name not in {"run", "decoder", "num_high_snr_cells", "family"}}
        if key in unique and reduced != unique[key]:
            raise ValueError(f"paired D0/D1 {value_name} differ for {key}")
        unique[key] = reduced
    return [{"family": FAMILY_LABEL[key[0]], **unique[key]} for key in sorted(unique)]


def plot_performance(rows: list[dict], out_dir: Path, high_snr: list[float]) -> None:
    geometries = {(row["B"], row["n"]) for row in rows}
    if len(geometries) != 1:
        raise ValueError(f"one merge must contain one (B,n) geometry, got {sorted(geometries)}")
    B, n = next(iter(geometries))
    supports = sorted({row["support_size"] for row in rows if row["encoder"] != "dense_fixed"})
    if not supports:
        raise ValueError("performance plot needs at least one sparse support")
    fig, axes = plt.subplots(1, len(supports), figsize=(7 * len(supports), 5.4), sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    dense = [row for row in rows if row["encoder"] == "dense_fixed"]
    for axis, support in zip(axes, supports):
        selected = dense + [row for row in rows if row["support_size"] == support and row["encoder"] != "dense_fixed"]
        x = np.arange(len(FAMILY_ORDER))
        for decoder_index, (decoder, (label, color)) in enumerate(DECODER_STYLE.items()):
            values = {row["encoder"]: row for row in selected if row["decoder"] == decoder}
            available = [(index, values[family]) for index, family in enumerate(FAMILY_ORDER) if family in values]
            offset = -0.08 if decoder_index == 0 else 0.08
            if available:
                axis.errorbar([index + offset for index, _ in available], [row["mean_high_snr_pupe"] for _, row in available],
                              yerr=[row["seed_standard_error"] for _, row in available], marker="o", color=color,
                              linewidth=1.4, capsize=3, label=label)
        axis.set_xticks(x, [FAMILY_LABEL[family] for family in FAMILY_ORDER], rotation=22, ha="right")
        axis.set(title=f"B={B}, n={n}, support T={support}", ylabel="mean PUPE", ylim=(0, 1))
        axis.grid(axis="y", alpha=0.22)
    for axis in axes[1:]:
        axis.set_ylabel("")
    axes[0].legend()
    snr = " and ".join(f"{value:g}" for value in high_snr)
    fig.suptitle(f"Hash-skeleton certification: mean PUPE over {snr} dB and all evaluated loads")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out_dir / "hash_skeleton_pupe.png", dpi=200); plt.close(fig)


def plot_geometry(rows: list[dict], out_dir: Path) -> None:
    metrics = [
        ("sampled_support_overlap_mean", "mean pair support overlap"),
        ("row_nonzero_load_cv", "row-load CV"),
        ("sampled_abs_correlation_q999", "sampled correlation q99.9"),
        ("active_expansion_ratio_mean", "active occupied rows / K"),
        ("active_gram_min_eigenvalue_q05", "active Gram min eigenvalue q05"),
        ("normalised_disjoint_K_sum_distance_q05", "normalised K-sum distance q05"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2)); axes = axes.reshape(-1)
    sparse = [row for row in rows if row["encoder"] != "dense_fixed"]
    supports = sorted({row["support_size"] for row in sparse})
    markers = ("o", "s", "^", "D")
    for axis, (metric, ylabel) in zip(axes, metrics):
        for support, marker in zip(supports, markers):
            points = []
            for encoder in FAMILY_ORDER[1:]:
                values = [float(row[metric]) for row in sparse if row["encoder"] == encoder and row["support_size"] == support]
                if values:
                    points.append((FAMILY_ORDER[1:].index(encoder), float(np.mean(values))))
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], marker=marker, label=f"T={support}")
        axis.set_xticks(range(len(FAMILY_ORDER) - 1), [FAMILY_LABEL[family] for family in FAMILY_ORDER[1:]], rotation=22, ha="right")
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.22)
    axes[0].legend()
    fig.suptitle("Support, correlation, active-set, and K-sum diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_dir / "hash_skeleton_geometry.png", dpi=200); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs, diagnostics, constructions, notes = load_runs(args)
    aggregate = aggregate_runs(runs)
    unique_diagnostics = unique_paired(diagnostics, "diagnostics")
    unique_constructions = unique_paired(constructions, "constructions")
    payload = {"high_snr_db": args.high_snr, "num_runs": len(runs), "completeness_notes": notes,
               "aggregate": aggregate, "runs": runs, "codebook_diagnostics": unique_diagnostics,
               "codebook_constructions": unique_constructions}
    (out_dir / "hash_skeleton_summary.json").write_text(json.dumps(payload, indent=2))
    write_tsv(out_dir / "hash_skeleton_aggregate.tsv", aggregate)
    write_tsv(out_dir / "hash_skeleton_runs.tsv", runs)
    write_tsv(out_dir / "hash_skeleton_diagnostics.tsv", unique_diagnostics)
    write_tsv(out_dir / "hash_skeleton_constructions.tsv", unique_constructions)
    plot_performance(aggregate, out_dir, args.high_snr); plot_geometry(unique_diagnostics, out_dir)
    print(f"merged {len(runs)} runs into {out_dir}; completeness notes={len(notes)}")


if __name__ == "__main__":
    main()
