"""Overlay the audited achievability reference on returned main-comparison results."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.ura_bounds import polyanskiy_achievability
from benchmarks.ura_merge import aggregate, load_results


CURVES = [
    ("odma_polar", "native", "native", "ODMA–polar native", "#8856a7", "-"),
    ("ccs_amp", "native", "native", "CCS native (nominal energy)*", "#977c63", ":"),
    ("ccs_block", "native", "native", "Block CCS native (adaptation)", "#b39bd1", "--"),
    ("dense", "fixed", "d2", "Fixed dense + D2", "#1678ad", "-"),
    ("sparse", "fixed", "d2", "Fixed sparse (s=32) + D2", "#d97a19", "-"),
    ("dense", "fixed", "d3", "Fixed dense + D3", "#2d9676", "--"),
]


def load_bound_profiles(path):
    data = json.loads(Path(path).read_text())
    hashes = {Path(p).name: sha for p, sha in data["source_sha256"].items()}
    for source in ("benchmarks/ura_bounds.py", "benchmarks/ura_bound_analysis.py", "src/ura_bound.py"):
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != hashes.get(Path(source).name):
            raise ValueError(f"Prepared bound uses different source: {source}; regenerate before overlaying")
    return {(r["B"], r["n"], r["K"], r["sampling"]): r for r in data["profiles"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results") / "comparison")
    parser.add_argument("--bounds", type=Path, default=ROOT / "results/032_bounds/summary.json")
    parser.add_argument("--sampling", choices=("distinct", "iid"), default="distinct")
    args = parser.parse_args()
    results, audit = load_results(Path(__file__).with_name("comparison.jsonl"), args.results, True)
    points, prepared = aggregate(results), load_bound_profiles(args.bounds)
    out = args.results / "analysis" / f"bound_overlay_{args.sampling}"
    out.mkdir(parents=True, exist_ok=True)
    curves = {}
    for b in (12, 14):
        for k in (7, 15, 22, 26):
            for family, mode, decoder, *_ in CURVES:
                group = [p for p in points if (p["B"], p["n"], p["K"], p["sampling"], p["family"], p["mode"], p["decoder"])
                         == (b, 256, k, args.sampling, family, mode, decoder)]
                if len(group) != 8 or any(len(p["seeds"]) != 3 for p in group):
                    raise ValueError(f"Need all matched seeds/SNRs for B={b}, K={k}, {family}/{decoder}")
            curve = prepared[b, 256, k, args.sampling]["curve"]
            values = dict(zip(curve["ebn0_db"], curve["polyanskiy_achievability"]))
            # Resolve the steep transition instead of drawing a straight line across a two-dB gap.
            for snr in np.arange(2, 6.1, .5):
                if snr not in values:
                    values[float(snr)] = polyanskiy_achievability(b, 256, k, snr, distinct=args.sampling == "distinct")
            curves[b, k] = sorted(values.items())
            print(f"Prepared B={b}, K={k}, {args.sampling}", flush=True)

    def panel(ax, b, k, logarithmic=False):
        for family, mode, decoder, label, color, style in CURVES:
            rows = sorted((p for p in points if (p["B"], p["n"], p["K"], p["sampling"], p["family"], p["mode"], p["decoder"])
                           == (b, 256, k, args.sampling, family, mode, decoder)), key=lambda p: p["ebn0_db"])
            ax.plot([p["ebn0_db"] for p in rows], [p["pupe"] for p in rows], color=color, ls=style,
                    marker="o", ms=3, lw=1.5, label=label)
        xs, ys = zip(*curves[b, k])
        ax.plot(xs, ys, color="#202020", ls="--", lw=2.1, label="Polyanskiy achievable-error upper bound (not a converse)")
        ax.axhline(.05, color="gray", ls=":", lw=.8)
        ax.set(title=f"B={b}, K={k}", xlabel="Physical Eb/N0 (dB)", ylabel="PUPE",
               ylim=(1e-4, 1.05) if logarithmic else (-.015, 1.015), yscale="log" if logarithmic else "linear")
        ax.grid(alpha=.18, which="both")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for zoom in (False, True):
        fig, axes = plt.subplots(1 if zoom else 2, 2 if zoom else 4, figsize=(12, 6.3) if zoom else (16, 8.8), squeeze=False)
        cases = [(b, 26) for b in (12, 14)] if zoom else [(b, k) for b in (12, 14) for k in (7, 15, 22, 26)]
        for ax, (b, k) in zip(axes.flat, cases): panel(ax, b, k, zoom)
        suffix = " — high-load log view" if zoom else ""
        fig.suptitle(f"Explicit codebooks vs native small-payload schemes, with achievability reference{suffix}\n"
                     f"n=256; {args.sampling} messages; three seeds × 256 frames per measured point", fontsize=13)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .035),
                   ncol=2 if zoom else 3, frameon=False, fontsize=9)
        note = "* Nominal-energy CCS is context only: the bound uses per-codeword energy ≤1. Dashed bound is not an error floor."
        if args.sampling == "iid": note += " Its collision-as-error correction is conservative for our PUPE."
        fig.text(.5, .012, note, ha="center", fontsize=8)
        fig.subplots_adjust(left=.065, right=.985, top=.82 if zoom else .85, bottom=.30 if zoom else .20,
                            wspace=.20 if zoom else .24, hspace=.48)
        fig.savefig(out / ("high_load_log.png" if zoom else "main_comparison_with_bound.png"), dpi=170)
        plt.close(fig)
    evidence = {"audit": audit, "sampling": args.sampling,
                "interpretation": "Achievable-error upper bound, not a converse; nominal CCS not peak-energy matched",
                "bound_source_sha256": hashlib.sha256((ROOT / "benchmarks/ura_bounds.py").read_bytes()).hexdigest(),
                "curves": [{"B": b, "n": 256, "K": k, "snr_and_bound": values} for (b, k), values in curves.items()]}
    (out / "bound_overlay.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(out)


if __name__ == "__main__": main()
