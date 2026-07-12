"""Run the pinned authors' CCS-AMP implementation and compare with paper points.

``paper_b128`` matches the published one-pass core dimensions.  The published
curve additionally uses a two-pass SIC extension whose empirical delta schedule
is not included in the public code; outputs record that limitation.  The
``adapted_b100`` preset is useful experimentally but is not paper-comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.plotting import _configure_matplotlib
from tests.ccs_amp_author import default_author_dir, load_author_modules, preset_parameters, run_author_trial
from tests.equivalence_outputs import write_polyanskiy_outputs


PAPER_ENHANCED = {10: 1.70, 25: 1.85, 50: 2.08, 75: 2.31, 100: 2.38, 125: 2.65, 150: 2.99, 175: 3.12}
PAPER_ORIGINAL = {10: 3.375, 25: 3.47, 50: 3.49, 75: 3.50, 100: 3.50, 125: 3.51, 150: 3.69, 175: 3.83}
PAPER_URL = "https://arxiv.org/abs/2010.04364"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["paper_b128", "adapted_b100"], default="paper_b128")
    p.add_argument("--author-code-dir", type=Path, default=default_author_dir())
    p.add_argument("--allow-unpinned-author-code", action="store_true")
    p.add_argument("--K-values", nargs="+", type=int, default=[10, 25, 50, 75, 100, 125, 150, 175])
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--transform-seed", type=int, default=0)
    p.add_argument("--amp-iterations", type=int, default=10)
    p.add_argument("--bp-iterations", type=int, default=1)
    p.add_argument("--schemes", nargs="+", choices=["enhanced", "original"], default=["enhanced", "original"])
    p.add_argument("--target-pupe", type=float, default=0.05)
    p.add_argument("--out-dir", type=Path, default=Path("results/ccs_amp_author_curve"))
    return p.parse_args(argv)


def plot(points, required, bounds, args, params):
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for scheme, color in (("enhanced", "#15803D"), ("original", "#2563EB")):
        for K in args.K_values:
            rows = sorted([p for p in points if p["scheme"] == scheme and p["K"] == K], key=lambda x: x["ebn0_db"])
            if rows:
                axes[0].plot([r["ebn0_db"] for r in rows], [r["mean_pupe"] for r in rows], marker="o", color=color,
                             alpha=0.25 + 0.65 * (K == args.K_values[-1]), label=f"{scheme}, K={K}")
    axes[0].axhline(args.target_pupe, color="black", ls="--", lw=1)
    axes[0].set(xlabel="Eb/N0 (dB)", ylabel="Mean PUPE", ylim=(-0.02, 1.02))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    for scheme, color, marker in (("enhanced", "#15803D", "o"), ("original", "#2563EB", "s")):
        rows = [r for r in required if r["scheme"] == scheme and np.isfinite(r["required_ebn0_db"])]
        axes[1].plot([r["K"] for r in rows], [r["required_ebn0_db"] for r in rows], color=color, marker=marker,
                     lw=2, label=f"author code, {scheme}")
    if params["paper_comparable"]:
        axes[1].plot(PAPER_ENHANCED.keys(), PAPER_ENHANCED.values(), color="#15803D", ls="--", marker="x", label="paper enhanced")
        axes[1].plot(PAPER_ORIGINAL.keys(), PAPER_ORIGINAL.values(), color="#2563EB", ls="--", marker="x", label="paper original")
    canonical = sorted([r for r in bounds if r["variant"] == "canonical"], key=lambda r: r["K"])
    axes[1].plot([r["K"] for r in canonical], [r["ebn0_db_experiment"] for r in canonical], color="#111827", ls=":",
                 marker="*", label="Polyanskiy canonical")
    axes[1].set(xlabel="Active devices K", ylabel="Required Eb/N0 (dB)")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    fig.suptitle(f"CCS-AMP author-code validation: {args.preset}")
    fig.tight_layout(); fig.savefig(args.out_dir / "ccs_amp_validation.png", dpi=160); plt.close(fig)


def main(argv=None):
    args = parse_args(argv); params = preset_parameters(args.preset)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modules = load_author_modules(args.author_code_dir, transform_seed=args.transform_seed,
                                  allow_unpinned=args.allow_unpinned_author_code)
    rows = []
    for scheme in args.schemes:
        for K in args.K_values:
            for ebn0_db in args.ebn0_grid:
                for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                    t0 = time.time()
                    result = run_author_trial(modules, preset=args.preset, K=K, ebn0_db=ebn0_db, seed=seed,
                                              amp_iterations=args.amp_iterations, bp_iterations=args.bp_iterations,
                                              enhanced=scheme == "enhanced")
                    row = {"scheme": scheme, "K": K, "ebn0_db": ebn0_db, "seed": seed, **result,
                           "wall_s": time.time() - t0}
                    rows.append(row)
                    print(f"{scheme:<8s} K={K:<3d} Eb/N0={ebn0_db:>4.2f} seed={seed} PUPE={result['pupe']:.4f} "
                          f"({row['wall_s']:.1f}s)", flush=True)
                    (args.out_dir / "checkpoint.json").write_text(json.dumps(rows, indent=2))
    points, required = [], []
    for scheme in args.schemes:
        for K in args.K_values:
            curve = []
            for ebn0_db in args.ebn0_grid:
                vals = [r["pupe"] for r in rows if r["scheme"] == scheme and r["K"] == K and r["ebn0_db"] == ebn0_db]
                point = {"scheme": scheme, "K": K, "ebn0_db": ebn0_db, "mean_pupe": float(np.mean(vals)),
                         "seed_se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")}
                points.append(point); curve.append(point)
            reached = [p for p in curve if p["mean_pupe"] <= args.target_pupe]
            required.append({"scheme": scheme, "K": K,
                             "required_ebn0_db": min([p["ebn0_db"] for p in reached], default=float("nan"))})
    bound_args = argparse.Namespace(n=params["n"], payload_bits=params["payload_bits"], num_antennas=1, K_values=args.K_values)
    bounds = write_polyanskiy_outputs(bound_args, args.out_dir, target_pupe=args.target_pupe, grid=25, num_pprime=25, axis="physical")
    payload = {"args": {**vars(args), "author_code_dir": str(args.author_code_dir)}, "preset": params,
               "author_commit": modules.commit, "paper_url": PAPER_URL, "paper_enhanced": PAPER_ENHANCED,
               "paper_original": PAPER_ORIGINAL, "paper_curve_has_unpublished_delta_schedule_two_pass_sic": True,
               "rows": rows, "points": points, "required": required, "polyanskiy": bounds}
    (args.out_dir / "ccs_amp_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot(points, required, bounds, args, params)
    print(f"Wrote {args.out_dir / 'ccs_amp_summary.json'} and {args.out_dir / 'ccs_amp_validation.png'}")


if __name__ == "__main__":
    main()

