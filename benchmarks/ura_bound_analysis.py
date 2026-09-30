"""Prepare job-032 bound curves and audit the original B=100 published reference."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ura_bounds import polyanskiy_achievability, q1_bound, reference_curves, required_ebn0


PAPER_READINGS = {50: .30, 100: .43, 150: .50, 200: .78, 250: 1.30, 300: 1.82}


def profiles_from_manifests(paths):
    profiles = defaultdict(set)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            for sampling in row["eval_sampling"]:
                for k in row["loads"]:
                    profiles[row["B"], row["n"], k, sampling].update(row["eval_ebn0"])
    for k in (1, 3):
        for sampling in ("distinct", "iid"):
            profiles[6, 64, k, sampling].update((-4, 0, 4, 8))
    # Native simulations target practical decoders; their SNR grid can start
    # above the bound's transition. Add analysis-only points, never job rows.
    for (b, _, _, _), snrs in profiles.items():
        if b > 20: snrs.update((-.5, -.25, 0, .25, .5, .75))
    return profiles


def plot_results(results, paper, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["K"] for r in paper], [r["ebn0_db"] for r in paper], "o-", label="Our p_t + q_1 evaluation")
    ax.errorbar(list(PAPER_READINGS), list(PAPER_READINGS.values()), yerr=.15, fmt="s", capsize=3,
                label="Published Fig. 1, visual readings ±0.15 dB")
    ax.set(xlabel="Active users K", ylabel="Required physical Eb/N0 (dB)",
           title="Polyanskiy 2017: B=100, n=30000, PUPE=0.10")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "paper_alignment.png", dpi=160); plt.close(fig)
    for b, n in sorted({(r["B"], r["n"]) for r in results}):
        rows = [r for r in results if (r["B"], r["n"]) == (b, n)]
        samples = sorted({r["sampling"] for r in rows})
        fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 4.5), squeeze=False)
        for ax, sampling in zip(axes.flat, samples):
            for row in rows:
                if row["sampling"] != sampling: continue
                curve = row["curve"]
                line, = ax.plot(curve["ebn0_db"], curve["polyanskiy_achievability"], "o-", label=f"K={row['K']}")
                ax.plot(curve["ebn0_db"], curve["polyanskiy_gallager_achievability"], ":", color=line.get_color(), alpha=.6)
            ax.axhline(.05, color="grey", linestyle="--", linewidth=.8)
            ax.set(title=f"{sampling} messages", xlabel="Physical Eb/N0 (dB)", ylabel="Achievable-error upper bound",
                   yscale="log", ylim=(1e-5, 1.05))
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle(f"B={b}, n={n}: solid p_t + q_1; dotted Gallager-only\nNot a converse or a measured decoder curve")
        fig.tight_layout()
        fig.savefig(out / f"B{b}_n{n}_bounds.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/032_bounds"))
    parser.add_argument("--manifests", nargs="+", default=["jobs/032_published_baselines/comparison.jsonl",
                                                          "jobs/032_published_baselines/native.jsonl"])
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for (b, n, k, sampling), snrs in sorted(profiles_from_manifests(args.manifests).items()):
        distinct = sampling == "distinct"
        curve = reference_curves(b, n, k, sorted(snrs), distinct=distinct)
        thresholds = {str(target): required_ebn0(b, n, k, target, distinct=distinct) for target in (.05, .1)}
        row = {"B": b, "n": n, "K": k, "sampling": sampling, "curve": curve, "required_ebn0": thresholds}
        results.append(row)
        with (args.out / "profiles.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"B={b}, n={n}, K={k}, {sampling}: {thresholds}", flush=True)
    paper = []
    for k in (1, 10, 25, 50, 100, 150, 200, 250, 300):
        crossing = required_ebn0(100, 30000, k, .1)
        row = {"K": k, **crossing, "figure_reading_db": PAPER_READINGS.get(k)}
        if k in PAPER_READINGS:
            row["difference_from_visual_reading_db"] = crossing["ebn0_db"] - PAPER_READINGS[k]
        paper.append(row)
        print("Paper reference", row, flush=True)
    checks = []
    for b, n, k, distinct in [(6, 64, 1, True), (14, 256, 26, False), (100, 30000, 50, False),
                              (100, 30000, 300, False), (128, 38400, 100, False)]:
        coarse = required_ebn0(b, n, k, .1, distinct=distinct)
        fine = required_ebn0(b, n, k, .1, distinct=distinct, grid=61, power_grid=61, order=192)
        snr = coarse["ebn0_db"]
        detail = polyanskiy_achievability(b, n, k, snr, distinct=distinct, details=True)
        p = detail["auxiliary_power_fraction"] * 2 * b * 10 ** (snr / 10) / n
        q_coarse, q_fine = (q1_bound(b, n, k, p, order=order)["value"] for order in (96, 192))
        check = {"B": b, "n": n, "K": k, "distinct": distinct, "coarse_db": snr, "fine_db": fine["ebn0_db"],
                 "grid_refinement_db": snr - fine["ebn0_db"], "q1_quadrature_difference": abs(q_coarse - q_fine)}
        checks.append(check)
        print("Numerical refinement", check, flush=True)
    summary = {"profiles": results, "paper_reference": paper, "numerical_checks": checks,
               "source": "https://people.lids.mit.edu/yp/homepage/data/isit17_mac.pdf",
               "method": "Theorem 1 p_t for all t, q_1 only, as in Section III; exact conditional law, numerical quadrature",
               "paper_readings": "Manual Fig. 1 readings with ±0.15 dB visual uncertainty, not author-tabulated data",
               "metric": "Collision-as-error iid PUPE; conservative also for duplicate-tolerant PUPE. Distinct is separate.",
               "energy": "Peak norm<=1. Not a peak-power certification of nominal-energy CCS/dynamic-CS waveforms.",
               "scope": "Analysis only: no receiver training, HPC manifest changes or optimality claim",
               "seconds": time.perf_counter() - started,
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path("benchmarks/ura_bounds.py"), Path("src/ura_bound.py")]}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_results(results, paper, args.out)
    print(f"Saved {len(results)} profiles and paper/numerical checks to {args.out}", flush=True)


if __name__ == "__main__": main()
