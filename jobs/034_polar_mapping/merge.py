"""Audit learned mappings against unchanged 033 waveforms and reuse 032/033 controls."""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.ura_comparison import DEFAULTS
from benchmarks.ura_merge import aggregate, load_results, plot_learning, plot_panel


ADDED_SOURCES = {"framework/polar_mapping.py", "benchmarks/polar_mapping.py"}


def audit_source_extension(old, new):
    if set(new) - set(old) != ADDED_SOURCES or set(old) - set(new):
        raise ValueError("Unexpected source inventory change between 033 and 034")
    if any(old[path] != new[path] for path in old):
        raise ValueError("Historical source changed; re-audit before combining 033 and 034")


def audit_controls(old, new):
    references = {r["config"]["name"]: r for r in old}
    for result in new:
        c = result["config"]
        if c["reference_033"] not in references: raise ValueError("Missing matched 033 control")
        ref = references[c["reference_033"]]
        for key in (*DEFAULTS, "B", "n", "seed", "selected_profile", "profile_selection", "baseline_params", "baseline_overrides"):
            if c.get(key) != ref["config"].get(key): raise ValueError(f"034/033 mismatch: {key}")
        if result["initial_matrix_sha256"] != ref["initial_matrix_sha256"]:
            raise ValueError("034 initialization differs from the 033 fixed codebook")
        for stage in ("initial", "final"):
            energy = result[f"energy_audit_{stage}"]
            if not all(math.isfinite(energy[k]) and abs(energy[k] - 1) <= 1e-5 for k in ("min", "max")):
                raise ValueError("Codeword energy violation")
        b, nc = c["baseline_params"]["prefix_bits"], c["baseline_params"]["code_length"]
        expected = (1 << b) * nc if c["family"] == "polar_energy" else nc * (nc - 1) // 2
        if result["encoder_learned_parameters"] != expected: raise ValueError("Unexpected encoder parameter count")
        if result["decoder_parameters"] != ref["decoder_parameters"]: raise ValueError("D2 parameter count changed")
        if c["family"] == "polar_mixer":
            error = result["mapping_diagnostics_final"]["orthogonality_max_error"]
            if not math.isfinite(error) or error > 1e-5: raise ValueError("Mixer is not orthogonal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--reference-033", type=Path, default=ROOT / "jobs/033_matched_d2/results")
    parser.add_argument("--reference-032", type=Path, default=ROOT / "jobs/032_published_baselines/results/comparison")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    new, audit = load_results(Path(__file__).with_name("manifest.jsonl"), args.results, args.allow_incomplete)
    old, audit33 = load_results(ROOT / "jobs/033_matched_d2/manifest.jsonl", args.reference_033, True)
    earlier, audit32 = load_results(ROOT / "jobs/032_published_baselines/comparison.jsonl", args.reference_032, True)
    audit_source_extension(old[0]["source_sha256"], new[0]["source_sha256"])
    historical = runpy.run_path(str(ROOT / "jobs/033_matched_d2/merge.py"))
    historical["audit_source_change"](earlier[0]["source_sha256"], old[0]["source_sha256"])
    audit_controls(old, new)
    controls = [r for r in old if r["config"]["family"] == "odma_polar"]
    controls += [r for r in earlier if (r["config"]["family"] in {"dense", "sparse"} and r["config"]["decoder"] == "d2")
                 or (r["config"]["family"], r["config"]["decoder"]) == ("odma_polar", "native")]
    if len(controls) != 36: raise ValueError("Need six 033 ODMA-D2 and thirty 032 dense/sparse-D2/native-ODMA controls")
    points = aggregate(controls + new)
    out = args.results / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"audit": audit, "reference_033": audit33, "reference_032": audit32,
                                                 "points": points}, indent=2) + "\n")
    if not args.no_plots:
        plot_learning(new, out)
        for b, n, sampling in sorted({(p["B"], p["n"], p["sampling"]) for p in points}):
            subset = [p for p in points if (p["B"], p["n"], p["sampling"]) == (b, n, sampling)]
            curves = {("dense", "fixed", "d2"): "Dense / trained D2", ("dense", "joint", "d2"): "Joint dense + D2",
                      ("sparse", "fixed", "d2"): "Sparse s32 / trained D2",
                      ("odma_polar", "fixed", "d2"): "Fixed ODMA-polar / trained D2",
                      ("polar_energy", "joint", "d2"): "Learned symbol energy + D2",
                      ("polar_mixer", "joint", "d2"): "Learned shared mixer + D2",
                      ("odma_polar", "native", "native"): "ODMA-polar / native"}
            plot_panel(subset, curves, f"B={b}, n={n}, {sampling}: matched polar encoder extensions",
                       out / f"B{b}_n{n}_{sampling}_mapping.png", False)
    print(json.dumps(audit, indent=2))
    print(f"Reused 36 completed controls; outputs: {out}")


if __name__ == "__main__": main()
