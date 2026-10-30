"""Validate complete paired 2x2 cohorts and summarize signature/energy interactions."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment import JOB, common, source_hashes
import matplotlib.pyplot as plt
import numpy as np


def key(record):
    return (record["decoder"], record["K"], record["ebn0_db"], record["frame"])


def read_run(path, row):
    result = json.loads((path / "summary.json").read_text())
    if result["status"] != "complete" or result["row"] != row or result["source_sha256"] != source_hashes():
        raise ValueError(f"Run status/config/source mismatch: {path}")
    records = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
    c = row["config"]
    expected = {(name, k, snr, frame) for name in c["decoders"] for k in c["loads"]
                for snr in c["eval_ebn0"] for frame in range(c["eval_frames"])}
    indexed = {key(r): r for r in records}
    if len(indexed) != len(records) or indexed.keys() != expected:
        raise ValueError(f"Duplicate/missing/unexpected frames: {path}")
    if any(r["sampling"] != "iid" or r["seed"] != c["seed"] for r in records):
        raise ValueError(f"Wrong source/seed: {path}")
    pairs = ([json.loads(line) for line in (path / "pairs.jsonl").read_text().splitlines()] if row.get("pair_frames", 0) else [])
    expected_pairs = {(k, snr, f) for k in c["loads"] for snr in c["eval_ebn0"] for f in range(row.get("pair_frames", 0))}
    if len(pairs) != len(expected_pairs) or {(p["K"], p["ebn0_db"], p["frame"]) for p in pairs} != expected_pairs:
        raise ValueError(f"Incomplete oracle pair diagnostics: {path}")
    return result, indexed, pairs


def historical_check(root, result, records, name):
    c = result["config"]
    path = root / f"B{c['B']}_n256_{c['mapping']}_iid_{name}_s{c['seed']}"
    if not (path / "summary.json").is_file():
        return {"path": str(path), "status": "not_available", "decoder": name}
    old = json.loads((path / "summary.json").read_text())
    contract = json.loads((ROOT / "jobs/037_polar_tractable/reference_contract.json").read_text())
    sources = [contract["current_source_sha256"], *[v["source_sha256"] for v in contract["historical_snapshots"]]]
    if old["status"] != "complete" or old["source_sha256"] not in sources:
        raise ValueError(f"Historical run is outside the audited source set: {path}")
    if old["matrix_sha256"] != result["matrix_sha256"] or old["decoder_params"][name] != result["decoder_params"][name]:
        raise ValueError(f"Historical matrix/decoder parameters differ: {path}")
    old_records = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
    indexed = {key(r): r for r in old_records if r["sampling"] == "iid" and r["decoder"] == name}
    selected = [r for r in records.values() if r["decoder"] == name]
    changes = 0
    for record in selected:
        previous = indexed.get(key(record))
        if previous is None or previous["seed"] != c["seed"] or previous["input_sha256"] != record["input_sha256"]:
            raise ValueError(f"Historical frame is missing/unpaired: {path}")
        changes += previous["counts"] != record["counts"]
    return {"path": str(path), "status": "audited", "decoder": name, "paired_frames": len(selected),
            "changed_count_decisions": changes, "source_sha256": old["source_sha256"],
            "scope": "replayed physical frames and decisions; historical metric labels are not pooled"}


def compare(cohort):
    """Only common truth/noise are paired across encoders; observations must change with the code."""
    cells, contrasts = [], []
    for b in sorted({x[0] for x in cohort}):
        seeds = sorted({x[1] for x in cohort if x[0] == b})
        first = cohort[(b, seeds[0], "reference", "uniform")][0]["config"]
        for seed in seeds:
            reference = cohort[(b, seed, "reference", "uniform")][1]
            for signature in ("reference", "repair3"):
                for mapping in ("uniform", "balanced"):
                    result, records, _ = cohort[(b, seed, signature, mapping)]
                    if records.keys() != reference.keys():
                        raise ValueError("Unequal frame coverage across arms")
                    for k, r in records.items():
                        if any(r[field] != reference[k][field] for field in ("truth_sha256", "standard_noise_sha256")):
                            raise ValueError("Encoder arms do not use identical truth and standard noise")
                    ref_result = cohort[(b, seed, "reference", mapping)][0]
                    if any(result["encoder"][field] != ref_result["encoder"][field]
                           for field in ("positions_sha256", "amplitudes_sha256")):
                        raise ValueError("The signature contrast also changed placement or energy")
        for name in first["decoders"]:
            for k in first["loads"]:
                for snr in first["eval_ebn0"]:
                    seed_means = {}
                    for signature in ("reference", "repair3"):
                        for mapping in ("uniform", "balanced"):
                            records = [r for seed in seeds for r in cohort[(b, seed, signature, mapping)][1].values()
                                       if (r["decoder"], r["K"], r["ebn0_db"]) == (name, k, snr)]
                            values = [np.mean([r["metrics"]["pupe"] for r in records if r["seed"] == seed]) for seed in seeds]
                            seed_means[(signature, mapping)] = np.array(values)
                            recall = [r["meta"]["candidate_recall"] for r in records if "candidate_recall" in r["meta"]]
                            cells.append({"B": b, "decoder": name, "K": k, "ebn0_db": snr, "signature": signature,
                                          "mapping": mapping, "frames": len(records), "seeds": seeds,
                                          "pupe": common.interval(values), "confidence_unit": "seed",
                                          "seconds_per_frame": float(np.mean([r["seconds"] for r in records])),
                                          "candidate_recall": float(np.mean(recall)) if recall else None})
                    ru, rb = seed_means[("reference", "uniform")], seed_means[("reference", "balanced")]
                    su, sb = seed_means[("repair3", "uniform")], seed_means[("repair3", "balanced")]
                    contrasts.append({"B": b, "decoder": name, "K": k, "ebn0_db": snr,
                                      "repair_minus_reference_uniform": common.interval(su - ru, bounded=False),
                                      "repair_minus_reference_balanced": common.interval(sb - rb, bounded=False),
                                      "balanced_minus_uniform_reference": common.interval(rb - ru, bounded=False),
                                      "balanced_minus_uniform_repair": common.interval(sb - su, bounded=False),
                                      "interaction": common.interval((sb - su) - (rb - ru), bounded=False)})
    return cells, contrasts


def plot(cells, out):
    for b in sorted({c["B"] for c in cells}):
        names = sorted({c["decoder"] for c in cells})
        loads = sorted({c["K"] for c in cells})
        fig, axes = plt.subplots(len(names), len(loads), figsize=(10, 6), squeeze=False)
        for i, name in enumerate(names):
            for j, k in enumerate(loads):
                ax = axes[i, j]
                for signature in ("reference", "repair3"):
                    for mapping in ("uniform", "balanced"):
                        rows = sorted((c for c in cells if (c["B"], c["decoder"], c["K"], c["signature"], c["mapping"])
                                       == (b, name, k, signature, mapping)), key=lambda c: c["ebn0_db"])
                        ax.plot([c["ebn0_db"] for c in rows], [c["pupe"]["mean"] for c in rows], marker="o",
                                label=f"{signature}/{mapping}")
                ax.set(title=f"{name}, K={k}", xlabel="Eb/N0 (dB)", ylabel="Count PUPE", ylim=(0, 1))
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4)
        fig.suptitle(f"B={b}, fixed 32 symbols, unit energy; exploratory seeds")
        fig.tight_layout(rect=(0, .07, 1, .96))
        fig.savefig(out / f"B{b}_pupe.png", dpi=160)
        plt.close(fig)


def summarize_pairs(records):
    fields = ("B", "signature", "mapping", "K", "ebn0_db", "same_pattern")
    cells = []
    for key in sorted({tuple(r[f] for f in fields) for r in records}):
        rows = [r for r in records if tuple(r[f] for f in fields) == key]
        cells.append({**dict(zip(fields, key)), "pairs": len(rows),
                      "mean": {field: float(np.mean([r[field] for r in rows])) for field in
                               ("minimum_competing_distance2", "competitors_within_4sigma2", "true_pair_posterior",
                                "pair_count_pupe", "conditional_bayes_pair_risk", "overlap", "seconds")},
                      "distance2_quantiles_0_10_50_90_100": np.quantile(
                          [r["minimum_competing_distance2"] for r in rows], [0, .1, .5, .9, 1]).tolist()})
    return cells


def compare_all_d2(entries):
    cells, contrasts = [], []
    for b in sorted({key[0] for key in entries}):
        seeds = sorted({key[1] for key in entries if key[0] == b})
        labels = sorted({key[2] for key in entries if key[0] == b})
        first = entries[(b, seeds[0], "polar_reference_balanced")][0]["config"]
        for seed in seeds:
            reference = entries[(b, seed, "polar_reference_balanced")][1]
            for label in labels:
                current = entries[(b, seed, label)][1]
                if reference.keys() != current.keys() or any(
                        any(current[key][field] != reference[key][field] for field in
                            ("truth_sha256", "standard_noise_sha256")) for key in reference):
                    raise ValueError(f"Unpaired encoder truth/noise or coverage: {b}/{seed}/{label}")
        for k in first["loads"]:
            for snr in first["eval_ebn0"]:
                means = {}
                for label in labels:
                    rows = [r for seed in seeds for r in entries[(b, seed, label)][1].values()
                            if (r["decoder"], r["K"], r["ebn0_db"]) == ("D2", k, snr)]
                    means[label] = np.array([np.mean([r["metrics"]["pupe"] for r in rows if r["seed"] == seed]) for seed in seeds])
                    cells.append(dict(B=b, encoder_id=label, decoder="D2", K=k, ebn0_db=snr, seeds=seeds,
                                      frames=len(rows), pupe=common.interval(means[label]),
                                      seconds_per_frame=float(np.mean([r["seconds"] for r in rows]))))
                for label in ("polar_repair3_uniform", "polar_repair3_balanced"):
                    for reference in labels:
                        if label != reference:
                            contrasts.append(dict(B=b, encoder_id=label, reference=reference, K=k, ebn0_db=snr,
                                                  seeds=seeds, delta=common.interval(means[label] - means[reference], bounded=False)))
    return cells, contrasts


def plot_all_d2(cells, out):
    for b in sorted({r["B"] for r in cells}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        for ax, k in zip(axes[0], sorted({r["K"] for r in cells})):
            for label in sorted({r["encoder_id"] for r in cells}):
                rows = sorted((r for r in cells if (r["B"], r["K"], r["encoder_id"]) == (b, k, label)),
                              key=lambda r: r["ebn0_db"])
                ax.plot([r["ebn0_db"] for r in rows], [r["pupe"]["mean"] for r in rows], "o-", label=label)
            ax.set(title=f"B{b}, K{k}: common D2", xlabel="Eb/N0 (dB)", ylabel="Count PUPE", ylim=(0, 1))
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, .2, 1, 1))
        fig.savefig(out / f"B{b}_all_encoders_D2.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=JOB / "results_signatures")
    parser.add_argument("--B", type=int, nargs="+", choices=[12, 14], default=[12, 14])
    parser.add_argument("--seeds", type=int, nargs="+", choices=[3201, 3202, 3203], default=[3201, 3202, 3203])
    parser.add_argument("--baseline-results", type=Path, default=ROOT / "jobs/035_polar_decoders/results_all_decoders")
    parser.add_argument("--tractable-results", type=Path, default=ROOT / "jobs/037_polar_tractable/results_tractable")
    parser.add_argument("--fresh-reference", action="store_true", help="Expect deliberately regenerated reference arms; omit historical replay")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty")
    cohort, history, pair_rows, all_d2 = {}, [], [], {}
    for row in [json.loads(line) for line in (JOB / "manifest.jsonl").read_text().splitlines()]:
        c = row["config"]
        if c["B"] not in args.B or c["seed"] not in args.seeds:
            continue
        if args.fresh_reference and "signature" in row:
            row["reference_policy"] = "regenerated"
        entry = read_run(args.results / row["name"], row)
        all_d2[(c["B"], c["seed"], row["encoder_id"])] = entry
        if "signature" not in row:
            continue
        cohort[(c["B"], c["seed"], row["signature"], c["mapping"])] = entry
        pair_rows.extend({**p, "B": c["B"], "signature": row["signature"], "mapping": c["mapping"]} for p in entry[2])
        if row["signature"] == "reference" and not args.fresh_reference:
            history.append(historical_check(args.baseline_results, entry[0], entry[1], "D2"))
            if "DEC-26" in c["decoders"]:
                history.append(historical_check(args.tractable_results, entry[0], entry[1], "DEC-26"))
    cells, contrasts = compare(cohort)
    d2_cells, d2_contrasts = compare_all_d2(all_d2)
    args.out.mkdir(parents=True, exist_ok=True)
    common.write_json(args.out / "summary.json", {"cells": cells, "paired_seed_contrasts": contrasts,
                                                "historical_replay": history, "pair_diagnostics": pair_rows,
                                                "pair_cells": summarize_pairs(pair_rows),
                                                "all_encoder_D2_cells": d2_cells, "all_encoder_D2_contrasts": d2_contrasts,
                                                "scope": "exploratory seeds; no held-out validation or global optimality"})
    with (args.out / "cells.csv").open("w", newline="") as stream:
        rows = [{**{k: c[k] for k in ("B", "decoder", "K", "ebn0_db", "signature", "mapping", "frames",
                                    "seconds_per_frame", "candidate_recall")}, "pupe": c["pupe"]["mean"]} for c in cells]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(cells, args.out)
    plot_all_d2(d2_cells, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
