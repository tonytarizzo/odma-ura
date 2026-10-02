"""Check completeness, select pilot profiles, and plot matched held-out results."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from .ura_bounds import reference_curves
from .ura_comparison import DEFAULTS


def load_results(manifest, root, allow_incomplete=False):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)): raise ValueError("Duplicate names in manifest")
    results, missing = [], []
    for row in rows:
        path = Path(root) / row["name"] / "summary.json"
        if not path.is_file():
            missing.append(row["name"]); continue
        result = json.loads(path.read_text())
        if result.get("status") != "complete": raise ValueError(f"Incomplete status in {path}")
        for key, value in {**DEFAULTS, **row}.items():
            if result["config"].get(key) != value: raise ValueError(f"Mismatched {key} in {path}")
        config = result["config"]
        expected = {(s, k, x) for s in config["eval_sampling"] for k in config["loads"] for x in config["eval_ebn0"]}
        actual = [(cell["sampling"], cell["K"], cell["ebn0_db"]) for cell in result["evaluation"]]
        if len(actual) != len(set(actual)) or set(actual) != expected: raise ValueError(f"Missing/duplicate evaluation cells: {path}")
        if any(cell["frames"] != config["eval_frames"] for cell in result["evaluation"]):
            raise ValueError(f"Incomplete frame budget in {path}")
        for cell in result["evaluation"]:
            metrics = cell["means"]
            if not 0 <= metrics["pupe"] <= 1: raise ValueError(f"Invalid PUPE in {path}")
            if "candidate_recall" in metrics and metrics["pupe"] + metrics["candidate_recall"] < 1 - 1e-9:
                raise ValueError(f"Candidate misses have been dropped from PUPE in {path}")
        results.append(result)
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)}/{len(rows)} rows: {', '.join(missing[:12])}")
    if not results: raise ValueError("No completed results")
    # Different source code or sample budgets should not quietly become one comparison.
    fingerprints = {json.dumps(result["source_sha256"], sort_keys=True) for result in results}
    if len(fingerprints) > 1: raise ValueError("Results use different source versions; audit them before merging")
    matrices, profiles = defaultdict(set), defaultdict(set)
    for result in results:
        c = result["config"]
        if "selected_profile" in c:
            selection = c.get("profile_selection")
            if not selection or "baseline_params" not in selection:
                raise ValueError("Missing or inconsistent pilot selection provenance")
            expected_params = {**selection["baseline_params"], **c.get("baseline_overrides", {})}
            if expected_params != c.get("baseline_params"):
                raise ValueError("Missing or inconsistent pilot selection provenance")
            profiles[c["selected_profile"]].add(json.dumps([selection, expected_params], sort_keys=True))
        if result["initial_matrix_sha256"]:
            matrices[(c["B"], c["n"], c["seed"], c["family"])].add(result["initial_matrix_sha256"])
    if any(len(choices) != 1 for choices in profiles.values()):
        raise ValueError("Native/learned variants or seeds use different selected pilot profiles")
    if any(len(hashes) != 1 for hashes in matrices.values()):
        raise ValueError("Fixed/joint or decoder variants do not share the same initial encoder")
    return results, {"expected": len(rows), "completed": len(results), "missing": missing,
                     "status": "partial" if missing else "complete"}


def select_profiles(results, audit):
    if audit["missing"]: raise ValueError("Do not select profiles from an unfinished tuning pilot")
    grouped = defaultdict(list)
    for result in results:
        c = result["config"]
        if c["mode"] != "native" or c["decoder"] != "native": raise ValueError("Selection requires native receiver pilot rows")
        score = float(np.mean([cell["means"]["pupe"] for cell in result["evaluation"]]))
        grouped[f"B{c['B']}_n{c['n']}_{c['family']}"].append((score, c))
    choices = {}
    for key, values in grouped.items():
        values.sort(key=lambda row: (row[0], row[1]["name"]))
        score, config = values[0]
        choices[key] = {"baseline_params": config["baseline_params"], "pilot_name": config["name"],
                        "selection_mean_pupe": score, "selection_seed": config["seed"],
                        "runner_up_mean_pupe": values[1][0] if len(values)>1 else None,
                        "criterion": "mean duplicate-tolerant PUPE across predeclared pilot cells",
                        "all_profiles": [{"name": c["name"], "mean_pupe": s, "params": c["baseline_params"]} for s, c in values]}
    return {**audit, "profiles": choices, "warning": "Small-B tuning, not published-curve verification"}


def aggregate(results):
    grouped = defaultdict(list)
    for result in results:
        c = result["config"]
        for cell in result["evaluation"]:
            key = (c["B"], c["n"], cell["sampling"], c["family"], c["mode"], c["decoder"], cell["K"], cell["ebn0_db"])
            grouped[key].append((c["seed"], cell))
    points = []
    for key, values in sorted(grouped.items()):
        if len({seed for seed, _ in values}) != len(values): raise ValueError("Duplicate seed for an aggregate cell")
        point = dict(zip(("B", "n", "sampling", "family", "mode", "decoder", "K", "ebn0_db"), key))
        means = np.array([cell["means"]["pupe"] for _, cell in values])
        point.update(pupe=float(means.mean()), seeds=[seed for seed, _ in values],
                     seed_values=means.tolist(), frames=sum(cell["frames"] for _, cell in values),
                     mean_seconds=float(np.mean([cell["decoder_seconds_per_frame"] for _, cell in values])))
        if len(means) > 1:
            half = student_t.ppf(0.975, len(means)-1) * means.std(ddof=1) / np.sqrt(len(means))
            point["seed_t95_interval"] = [max(0., float(means.mean()-half)), min(1., float(means.mean()+half))]
        else:
            point["seed_t95_interval"] = None
        if "candidate_recall" in values[0][1]["means"]:
            point["candidate_recall"] = float(np.mean([cell["means"]["candidate_recall"] for _, cell in values]))
        points.append(point)
    return points


def plot_panel(points, curves, title, out_path, bounds, metric="pupe"):
    chosen = [p for p in points if (p["family"], p["mode"], p["decoder"]) in curves]
    loads = sorted({p["K"] for p in chosen})
    if not loads: return
    fig, axes = plt.subplots((len(loads)+1)//2, min(2, len(loads)), figsize=(11, 3.7*((len(loads)+1)//2)), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, k in zip(axes.flat, loads):
        for i, (curve, label) in enumerate(curves.items()):
            rows = sorted((p for p in chosen if p["K"] == k and (p["family"], p["mode"], p["decoder"]) == curve),
                          key=lambda p: p["ebn0_db"])
            if not rows: continue
            x, y = [r["ebn0_db"] for r in rows], [r[metric] for r in rows]
            ax.plot(x, y, "o-", color=colors(i), markersize=3, label=label)
            if metric == "pupe" and all(r["seed_t95_interval"] is not None for r in rows):
                ax.fill_between(x, [r["seed_t95_interval"][0] for r in rows],
                                [r["seed_t95_interval"][1] for r in rows], color=colors(i), alpha=.10)
        if bounds:
            xs = sorted({p["ebn0_db"] for p in chosen if p["K"] == k})
            b, n, sampling = chosen[0]["B"], chosen[0]["n"], chosen[0]["sampling"]
            ref = reference_curves(b, n, k, xs, distinct=sampling == "distinct")
            ax.plot(xs, ref["polyanskiy_achievability"], "k--", linewidth=1, label=ref["achievability_label"])
        ax.set(title=f"K={k}", xlabel="Physical Eb/N0 (dB)",
               ylabel="PUPE" if metric == "pupe" else "True-message candidate recall", ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(loads):]: ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.savefig(out_path, dpi=160); plt.close(fig)


def plots(points, out_dir):
    geometries = sorted({(p["B"], p["n"], p["sampling"]) for p in points})
    for b, n, sampling in geometries:
        subset = [p for p in points if (p["B"], p["n"], p["sampling"]) == (b, n, sampling)]
        prefix = f"B{b}_n{n}_{sampling}"
        for family in ("odma_polar", "dynamic_cs", "ccs_amp", "ccs_block"):
            if not any(p["family"] == family for p in subset): continue
            for decoder in ("d0", "d1"):
                if not any(p["family"] == family and p["decoder"] == decoder for p in subset): continue
                curves = {(family, "native", "native"): f"{family}: native receiver",
                          (family, "fixed", decoder): f"{family}: {decoder.upper()}"}
                for ref, mode in product_pairs(): curves[(ref, mode, decoder)] = f"{ref} {mode}: {decoder.upper()}"
                title = f"B={b}, n={n}, {sampling}: {family} vs explicit references"
                if family == "ccs_amp": title += "\nNominal CCS energy varies by message; no peak-power bound overlay"
                plot_panel(subset, curves, title, out_dir / f"{prefix}_{family}_{decoder}.png", family != "ccs_amp")
        for family in ("dense", "sparse"):
            for mode in ("fixed", "joint"):
                curves = {(family, mode, f"d{i}"): f"D{i}" for i in range(5)}
                plot_panel(subset, curves, f"B={b}, n={n}, {sampling}: {mode} {family} decoder ladder",
                           out_dir / f"{prefix}_{family}_{mode}_ladder.png", True)
            curves = {(family, mode, decoder): f"{mode} {decoder.upper()}" for mode in ("fixed", "joint")
                      for decoder in ("d3", "d4")}
            candidates = [p for p in subset if "candidate_recall" in p]
            plot_panel(candidates, curves, f"B={b}, n={n}, {sampling}: {family} proposal recall (higher is better)",
                       out_dir / f"{prefix}_{family}_candidate_recall.png", False, "candidate_recall")
        if b > 20:
            curves = {(p["family"], "native", "native"): p["family"] for p in subset}
            plot_panel(subset, curves, f"Native validation only: B={b}, n={n} (few frames; not a reproduced curve)",
                       out_dir / f"{prefix}_native.png", False)


def product_pairs(): return ((family, mode) for family in ("dense", "sparse") for mode in ("fixed", "joint"))


def plot_learning(results, out_dir):
    learned = [r for r in results if r["training"] is not None]
    if not learned: return
    groups = sorted({(r["config"]["B"], r["config"]["n"], r["config"]["seed"]) for r in learned})
    for b, n, seed in groups:
        rows = [r for r in learned if (r["config"]["B"], r["config"]["n"], r["config"]["seed"]) == (b, n, seed)]
        families = sorted({r["config"]["family"] for r in rows})
        fig, axes = plt.subplots((len(families)+1)//2, min(2, len(families)), figsize=(11, 3.3*((len(families)+1)//2)), squeeze=False)
        for ax, family in zip(axes.flat, families):
            for result in rows:
                c, training = result["config"], result["training"]
                if c["family"] != family: continue
                epochs = [0] + [v["epoch"] for v in training["history"]]
                values = [training["initial"]["loss"]] + [v["validation"]["loss"] for v in training["history"]]
                ax.plot(epochs, values, label=f"{c['mode']} {c['decoder'].upper()}")
            ax.set(title=family, xlabel="Epoch", ylabel="Validation loss", yscale="log")
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(families):]: ax.set_visible(False)
        fig.suptitle(f"B={b}, n={n}, seed={seed}: learning checks (D0/D1 and D2–D4 have different losses)")
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(out_dir / f"B{b}_n{n}_s{seed}_learning.png", dpi=160); plt.close(fig)


def paper_alignment(results, points, out_dir, make_plots=True):
    references = {}
    for result in results:
        c = result["config"]
        if "paper_reference" not in c: continue
        for k in c["loads"]:
            key = (c["B"], c["n"], c["family"], k)
            if key in references and references[key] != c["paper_reference"]:
                raise ValueError("Native rows disagree on their paper reference")
            references[key] = c["paper_reference"]
    checks = []
    for (b, n, family, k), reference in sorted(references.items()):
        rows = sorted((p for p in points if (p["B"], p["n"], p["family"], p["K"]) == (b, n, family, k)),
                      key=lambda p: p["ebn0_db"])
        target = reference["target_pupe"]
        first = next((i for i, row in enumerate(rows) if row["pupe"] <= target), None)
        bracket = ([rows[-1]["ebn0_db"], None] if first is None else
                   [None if first == 0 else rows[first-1]["ebn0_db"], rows[first]["ebn0_db"]])
        check = {"family": family, "B": b, "n": n, "K": k, "paper_reference": reference,
                 "mean_pupe_crossing_bracket_db": bracket,
                 "nonmonotone_mean_pupe": any(a["pupe"] < z["pupe"] for a, z in zip(rows, rows[1:])),
                 "interpretation": "Grid bracket of mean PUPE, not a confidence interval or automatic reproduction verdict",
                 "points": rows}
        checks.append(check)
        if not make_plots: continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([p["ebn0_db"] for p in rows], [p["pupe"] for p in rows], "o-", label="Independent native implementation")
        if all(p["seed_t95_interval"] is not None for p in rows):
            ax.fill_between([p["ebn0_db"] for p in rows], [p["seed_t95_interval"][0] for p in rows],
                            [p["seed_t95_interval"][1] for p in rows], alpha=.12, label="Indicative 95% interval across seeds")
        center, half = reference["approx_required_ebn0_db"], reference["reading_uncertainty_db"]
        ax.axvspan(center-half, center+half, color="tab:orange", alpha=.2, label="Paper threshold (approximate visual reading)")
        ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"Paper target PUPE={target:g}")
        ax.set(title=f"{family}: B={b}, n={n}, K={k}\nPaper alignment check; not yet a verified reproduction",
               xlabel="Physical Eb/N0 (dB)", ylabel="PUPE", ylim=(-.005, min(1.02, max(.15, max(p["pupe"] for p in rows)*1.15))))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / f"B{b}_n{n}_{family}_K{k}_paper_alignment.png", dpi=160); plt.close(fig)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--select-pilot", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    results, audit = load_results(args.manifest, args.results, args.allow_incomplete)
    if args.select_pilot:
        selected = select_profiles(results, audit)
        (args.results / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
        print(json.dumps(selected, indent=2)); return
    out_dir = args.results / "analysis"
    out_dir.mkdir(exist_ok=True)
    points = aggregate(results)
    alignment = paper_alignment(results, points, out_dir, not args.no_plots)
    (out_dir / "summary.json").write_text(json.dumps({"audit": audit, "points": points, "paper_alignment": alignment}, indent=2) + "\n")
    if not args.no_plots:
        plots(points, out_dir)
        plot_learning(results, out_dir)
    print(json.dumps(audit, indent=2))
    print(out_dir)


if __name__ == "__main__": main()
