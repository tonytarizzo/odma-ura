"""Audit 033 and reuse completed 032 controls after an exact, pinned source-change check."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.ura_comparison import DEFAULTS
from benchmarks.ura_merge import aggregate, load_results, plot_panel


# Audited extension only: add encoder factory routes/metadata, distinguish J in grouping, add plots.
# Training, evaluation, noise, decoders, objectives, native constructions and old factory branches are unchanged.
AUDITED_CHANGES = {
    "benchmarks/ura_comparison.py": (
        "42d029772bca38c8539387eddd2e10df7d404f7dd393be77ae6cd63e040e7540",
        "879caeed4c78bdd79f1b4c47cdc6e29353c286eb296e3f41c3eafa2f28a947b0"),
    "benchmarks/ura_merge.py": (
        "23facd3cae94483bb369ebaa16bc68eb8d74cfd7afc7a47ee98975e561080067",
        "bc2c0d6a9ebc87b8ca6c2347acb27213e81ee2b3e07a1c4a71d0bc9a83f4d7b2"),
}


def audit_source_change(old, new):
    if old.keys() != new.keys(): raise ValueError("Source inventory changed; audit before combining 032 and 033")
    changes = {p: (old[p], new[p]) for p in old if old[p] != new[p]}
    if any(pair != AUDITED_CHANGES.get(path) for path, pair in changes.items()):
        raise ValueError("Unreviewed source changes; refuse to combine 032 and 033")
    return changes


def audit_controls(old, new):
    lookup = {(r["config"]["B"], r["config"]["seed"], r["config"]["family"], r["config"]["decoder"]): r for r in old}
    supports = {}
    for result in new:
        c = result["config"]
        reference = lookup[(c["B"], c["seed"], "dense", "d2")]
        for key in DEFAULTS:
            if c[key] != reference["config"][key]: raise ValueError(f"032/033 mismatch: {key}")
        if "selected_profile" in c:
            reference = lookup[(c["B"], c["seed"], c["family"], "d1")]
            for key in ("profile_selection", "baseline_params", "baseline_overrides"):
                if c.get(key) != reference["config"].get(key): raise ValueError(f"Published profile changed: {key}")
            if result["initial_matrix_sha256"] != reference["initial_matrix_sha256"]:
                raise ValueError("Published D2 codebook differs from the original D0/D1 codebook")
        else:
            metadata = result["encoder_construction"]
            key, support = (c["B"], c["seed"]), (metadata["A"], metadata["b"])
            if supports.setdefault(key, support) != support: raise ValueError("Unpaired hash supports")
            if metadata["prototype_storage_values"] != c["support"] * (1 << c["amplitude_label_bits"]):
                raise ValueError("Unexpected amplitude bank size")
            for stage in ("initial", "final"):
                energy = result[f"energy_audit_{stage}"]
                if max(abs(energy[k] - 1) for k in ("min", "max")) > 1e-5:
                    raise ValueError("Hash codebook violates unit energy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--reference-results", type=Path, default=ROOT / "jobs/032_published_baselines/results/comparison")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    new, audit = load_results(Path(__file__).with_name("manifest.jsonl"), args.results, args.allow_incomplete)
    old, reference_audit = load_results(ROOT / "jobs/032_published_baselines/comparison.jsonl", args.reference_results, True)
    changes = audit_source_change(old[0]["source_sha256"], new[0]["source_sha256"])
    audit_controls(old, new)
    references = [r for r in old if r["config"]["decoder"] in {"native", "d2"}]
    if len(references) != 42: raise ValueError("Need all 18 native and 24 dense/sparse D2 reference rows from 032")
    points = aggregate(references + new)
    out = args.results / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"audit": audit, "reference_audit": reference_audit,
                                                 "audited_source_changes": changes, "points": points}, indent=2) + "\n")
    if not args.no_plots:
        for b, n, sampling in sorted({(p["B"], p["n"], p["sampling"]) for p in points}):
            subset = [p for p in points if (p["B"], p["n"], p["sampling"]) == (b, n, sampling)]
            prefix = f"B{b}_n{n}_{sampling}"
            families = ("dense", "sparse", "hash_full", "hash_coordinate_J4", "hash_coordinate_J8", "hash_coordinate_J10")
            for mode in ("fixed", "joint"):
                curves = {(f, mode, "d2"): f.replace("_", " ") for f in families}
                plot_panel(subset, curves, f"{prefix}: {mode} amplitudes; full-alphabet D2",
                           out / f"{prefix}_{mode}_amplitudes.png", True)
            for paper in ("odma_polar", "ccs_amp", "ccs_block"):
                curves = {(paper, "native", "native"): f"{paper}: native", (paper, "fixed", "d2"): f"{paper}: D2"}
                curves.update({(f, "fixed", "d2"): f.replace("_", " ") for f in families})
                title = f"{prefix}: fixed codebooks, trained D2 vs {paper} native"
                if paper == "ccs_amp": title += "\nBound is a unit-energy reference only; CCS energy varies by message"
                plot_panel(subset, curves, title, out / f"{prefix}_{paper}.png", True)
    print(json.dumps(audit, indent=2))
    print(f"Reused 42 completed 032 reference rows; outputs: {out}")


if __name__ == "__main__": main()
