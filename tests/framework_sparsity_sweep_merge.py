"""Merge sparse-global density sweeps and plot PUPE against the nonzero fraction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from tests.framework_sparsity_diagnostics import log2_binomial


FAMILY_LABEL = {"sparse_global_fixed": "sparse global", "dense_fixed": "dense control", "odma_fixed": "ODMA control"}
DECODER_STYLE = {"d0": ("D0", "#76b7f2"), "d1": ("D1", "#f59d56")}


def parse_float_grid(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", action="append", required=True, help="repeat for multiple compatible result trees")
    p.add_argument("--manifest", action="append", default=[], help="optional manifest(s) used to enforce completeness")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--high-snr", type=parse_float_grid, default=parse_float_grid("8,12"))
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args(argv)


def read_expected_names(paths: list[str]) -> set[str]:
    names: set[str] = set()
    for manifest in paths:
        with Path(manifest).open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "name" not in reader.fieldnames:
                raise ValueError(f"manifest {manifest} must have a name column")
            for row in reader:
                if row["name"] in names:
                    raise ValueError(f"duplicate manifest run name {row['name']}")
                names.add(row["name"])
    return names


def standard_error(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def density_formatter(value: float, _position: int) -> str:
    if value <= 0:
        return ""
    inverse = round(1.0 / value)
    if inverse >= 1 and abs(value - 1.0 / inverse) < 1e-8:
        return "1" if inverse == 1 else f"1/{inverse}"
    return f"{value:.3g}"


def load_runs(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[str]]:
    paths = []
    for root in args.result_root:
        paths.extend(sorted(Path(root).glob("*/summary.json")))
    if not paths:
        raise SystemExit("no */summary.json files found under the supplied result roots")
    seen_names, run_rows, diagnostic_rows = set(), [], []
    for path in paths:
        name = path.parent.name
        if name in seen_names:
            raise ValueError(f"duplicate result run name {name}")
        seen_names.add(name)
        payload = json.loads(path.read_text()); metadata = payload["metadata"]; run_args = metadata["args"]
        diagnostics = metadata.get("codebook_sparsity")
        if diagnostics is None:
            raise ValueError(f"{path} has no codebook_sparsity diagnostics")
        high = [row for row in payload["learned"] if any(abs(float(row["ebn0_db"]) - target) < 1e-8 for target in args.high_snr)]
        high_mf = [row for row in payload["matched_filter"]
                   if any(abs(float(row["ebn0_db"]) - target) < 1e-8 for target in args.high_snr)]
        if not high or len(high) != len(high_mf):
            raise ValueError(f"{path} does not contain matching learned/MF high-SNR cells")
        common = {
            "run": name, "encoder": run_args["encoder"], "family": FAMILY_LABEL[run_args["encoder"]],
            "decoder": run_args["decoder"], "B": int(run_args["payload_bits"]), "n": int(run_args["n"]),
            "support_size": int(diagnostics["support_size"]), "nonzero_fraction": float(diagnostics["nonzero_fraction"]),
            "seed": int(run_args["seed"]), "num_high_snr_cells": len(high),
        }
        learned_pupe = float(np.mean([float(row["pupe"]) for row in high]))
        mf_pupe = float(np.mean([float(row["pupe"]) for row in high_mf]))
        run_rows.append({**common, "mean_high_snr_pupe": learned_pupe, "mean_high_snr_mf_pupe": mf_pupe,
                         "learned_minus_mf": learned_pupe - mf_pupe})
        diagnostic_rows.append({**common, **diagnostics})
    expected = read_expected_names(args.manifest)
    missing = sorted(expected - seen_names); unexpected = sorted(seen_names - expected) if expected else []
    if (missing or unexpected) and not args.allow_incomplete:
        details = []
        if missing: details.append(f"missing {len(missing)}: {', '.join(missing[:8])}")
        if unexpected: details.append(f"unexpected {len(unexpected)}: {', '.join(unexpected[:8])}")
        raise SystemExit("manifest/result mismatch; " + "; ".join(details))
    return run_rows, diagnostic_rows, missing + [f"unexpected:{name}" for name in unexpected]


def aggregate_runs(run_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in run_rows:
        key = (row["B"], row["n"], row["encoder"], row["decoder"], row["support_size"])
        grouped[key].append(row)
    out = []
    for (B, n, encoder, decoder, support), rows in sorted(grouped.items()):
        learned = [row["mean_high_snr_pupe"] for row in rows]
        matched = [row["mean_high_snr_mf_pupe"] for row in rows]
        delta = [row["learned_minus_mf"] for row in rows]
        out.append({"B": B, "n": n, "encoder": encoder, "family": FAMILY_LABEL[encoder], "decoder": decoder,
                    "support_size": support, "nonzero_fraction": support / n, "num_seeds": len(rows),
                    "mean_high_snr_pupe": float(np.mean(learned)), "seed_standard_error": standard_error(learned),
                    "mean_high_snr_mf_pupe": float(np.mean(matched)), "mean_learned_minus_mf": float(np.mean(delta))})
    return out


def unique_codebook_diagnostics(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        key = (row["B"], row["n"], row["encoder"], row["support_size"], row["seed"])
        reduced = {key_: value for key_, value in row.items() if key_ not in {"run", "decoder", "num_high_snr_cells"}}
        if key in unique:
            comparable = {key_: value for key_, value in reduced.items() if key_ != "family"}
            previous = {key_: value for key_, value in unique[key].items() if key_ != "family"}
            if comparable != previous:
                raise ValueError(f"paired D0/D1 codebook diagnostics differ for {key}")
        unique[key] = reduced
    return [unique[key] for key in sorted(unique)]


def plot_performance(rows: list[dict], out_dir: Path, high_snr: list[float]) -> None:
    geometries = sorted({(row["B"], row["n"]) for row in rows})
    columns = min(2, len(geometries)); panel_rows = math.ceil(len(geometries) / columns)
    fig, axes = plt.subplots(panel_rows, columns, figsize=(7.2 * columns, 5.4 * panel_rows), squeeze=False)
    axes_flat = axes.reshape(-1)
    max_y = max(row["mean_high_snr_pupe"] + row["seed_standard_error"] for row in rows)
    for axis, (B, n) in zip(axes_flat, geometries):
        selected = [row for row in rows if row["B"] == B and row["n"] == n]
        sparse = [row for row in selected if row["encoder"] == "sparse_global_fixed"]
        densities = sorted({row["nonzero_fraction"] for row in sparse}, reverse=True)
        for decoder, (label, color) in DECODER_STYLE.items():
            curve = sorted((row for row in sparse if row["decoder"] == decoder),
                           key=lambda row: row["nonzero_fraction"], reverse=True)
            if curve:
                axis.errorbar([row["nonzero_fraction"] for row in curve], [row["mean_high_snr_pupe"] for row in curve],
                              yerr=[row["seed_standard_error"] for row in curve], color=color, marker="o", markersize=4,
                              linewidth=1.5, capsize=2, label=f"sparse global {label}")
            for encoder, marker, control_label in (("dense_fixed", "D", "dense"), ("odma_fixed", "X", "ODMA")):
                control = [row for row in selected if row["decoder"] == decoder and row["encoder"] == encoder]
                if control:
                    row = control[0]
                    axis.errorbar(row["nonzero_fraction"], row["mean_high_snr_pupe"], yerr=row["seed_standard_error"],
                                  color=color, marker=marker, markersize=8, linestyle="none", capsize=3,
                                  markeredgecolor="black", markeredgewidth=0.5, label=f"{control_label} {label}")
        shortfall = [row["nonzero_fraction"] for row in sparse if row["support_size"] <= n // 2 and
                     log2_binomial(n, row["support_size"]) + 1e-10 < B]
        if shortfall:
            axis.axvspan(min(shortfall) / 1.35, max(shortfall) * 1.35, color="crimson", alpha=0.08,
                         label="support masks alone < $2^B$")
        axis.axvline(0.25, color="0.45", linestyle=":", linewidth=1)
        power_ticks = []
        for density in densities:
            support = round(n * density)
            if support > 0 and support & (support - 1) == 0:
                power_ticks.append(density)
        axis.set_xscale("log", base=2); axis.invert_xaxis(); axis.set_xticks(power_ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(density_formatter))
        axis.set(title=f"B={B}, n={n}", xlabel="nonzero fraction $p=s/n$ (integer support $s$)", ylabel="mean PUPE",
                 ylim=(0.0, min(1.0, max_y + 0.08)))
        axis.grid(alpha=0.22, which="both")
    for axis in axes_flat[len(geometries):]: axis.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(4, len(unique)), fontsize=9)
    snr_text = " and ".join(f"{value:g}" for value in high_snr)
    fig.suptitle(f"Sparse-global density frontier: mean PUPE over {snr_text} dB and all evaluated loads")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95)); fig.savefig(out_dir / "sparse_density_pupe.png", dpi=200); plt.close(fig)


def plot_diagnostics(rows: list[dict], out_dir: Path) -> None:
    sparse = [row for row in rows if row["encoder"] == "sparse_global_fixed"]
    if not sparse:
        return
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in sparse:
        grouped[(row["B"], row["n"], row["support_size"])].append(row)
    metrics = [("signed_support_duplicate_fraction", "repeated signed-support fraction\n(amplitudes ignored)"),
               ("row_nonzero_load_cv", "row nonzero-load CV"), ("row_energy_cv", "row-energy CV"),
               ("sampled_abs_correlation_q999", "sampled $|\\phi_i^H\\phi_j|$ q99.9")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.reshape(-1)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for B, n in sorted({(key[0], key[1]) for key in grouped}):
            points = []
            for (key_B, key_n, support), values in grouped.items():
                if (key_B, key_n) == (B, n):
                    points.append((support / n, float(np.mean([row[metric] for row in values]))))
            points.sort(reverse=True)
            axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=f"B={B}, n={n}")
        axis.set_xscale("log", base=2); axis.invert_xaxis(); axis.xaxis.set_major_formatter(FuncFormatter(density_formatter))
        axis.set(xlabel="nonzero fraction $p=s/n$", ylabel=ylabel); axis.grid(alpha=0.22, which="both")
    axes[0].legend()
    fig.suptitle("Sparse-global discrete and geometric diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_dir / "sparse_density_diagnostics.png", dpi=200); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_rows, raw_diagnostics, completeness_notes = load_runs(args)
    aggregate = aggregate_runs(run_rows); diagnostics = unique_codebook_diagnostics(raw_diagnostics)
    payload = {"result_roots": args.result_root, "manifests": args.manifest, "high_snr_db": args.high_snr,
               "num_runs": len(run_rows), "completeness_notes": completeness_notes,
               "aggregate": aggregate, "runs": run_rows, "codebook_diagnostics": diagnostics}
    (out_dir / "sparse_density_summary.json").write_text(json.dumps(payload, indent=2))
    write_tsv(out_dir / "sparse_density_aggregate.tsv", aggregate)
    write_tsv(out_dir / "sparse_density_runs.tsv", run_rows)
    write_tsv(out_dir / "sparse_density_diagnostics.tsv", diagnostics)
    plot_performance(aggregate, out_dir, args.high_snr); plot_diagnostics(diagnostics, out_dir)
    num_missing = sum(not note.startswith("unexpected:") for note in completeness_notes)
    print(f"merged {len(run_rows)} runs into {out_dir}; missing={num_missing}")


if __name__ == "__main__":
    main()
