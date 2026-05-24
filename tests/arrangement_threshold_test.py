"""Paired ODMA-vs-dense Eb/N0 threshold experiment.

This is a small wrapper around tests.threshold_test for paired decoder comparisons: same payload, same n,
same active-user loads, same seeds, but two resource arrangements:

  - ODMA:  d=<odma_d>, num_blocks=<odma_blocks>
  - Dense: d=n,        num_blocks=1

The output plot has one curve per (arrangement, target PUPE).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402
from src.decoders.registry import all_names  # noqa: E402
from src.plotting import _configure_matplotlib  # noqa: E402
from tests.threshold_test import run_bisect_search  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-B", "--payload-bits", type=int, default=10,
                   help="payload bits per active user; the explicit message alphabet is M=2^B")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--odma-d", type=int, default=128)
    p.add_argument("--odma-blocks", type=int, default=16)
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--K-values", nargs="+", type=int, default=[2, 5, 10, 15, 20, 30, 40, 50])
    p.add_argument("--targets", nargs="+", type=float, default=[0.05, 0.10])
    p.add_argument("--ebn0-min", type=float, default=-8.0)
    p.add_argument("--ebn0-max", type=float, default=8.0)
    p.add_argument("--ebn0-tol", type=float, default=0.25)
    p.add_argument("--max-search-steps", type=int, default=9)
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--decoder", default="NNOMP-OracleK", choices=all_names())
    p.add_argument("--out-name", default="arrangement_threshold",
                   help="subdirectory under results/threshold")
    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.payload_bits <= 0:
        raise SystemExit(f"--payload-bits must be positive, got {args.payload_bits}")
    if args.n <= 0 or args.odma_d <= 0 or args.odma_d > args.n:
        raise SystemExit(f"invalid ODMA geometry: n={args.n}, odma_d={args.odma_d}")
    if args.odma_blocks <= 0:
        raise SystemExit(f"--odma-blocks must be positive, got {args.odma_blocks}")
    if args.num_antennas < 2:
        raise SystemExit(f"--num-antennas={args.num_antennas} is not supported by the V2 common-signature model")
    if any(K <= 0 for K in args.K_values):
        raise SystemExit(f"--K-values must be positive, got {args.K_values}")
    if any(t <= 0.0 or t >= 1.0 for t in args.targets):
        raise SystemExit(f"--targets must lie strictly between 0 and 1, got {args.targets}")
    if not np.isfinite(args.ebn0_min) or not np.isfinite(args.ebn0_max) or args.ebn0_min >= args.ebn0_max:
        raise SystemExit(f"invalid bisection bracket: [{args.ebn0_min}, {args.ebn0_max}]")
    if args.ebn0_tol <= 0.0:
        raise SystemExit(f"--ebn0-tol must be positive, got {args.ebn0_tol}")
    if args.max_search_steps <= 0:
        raise SystemExit(f"--max-search-steps must be positive, got {args.max_search_steps}")
    if args.num_seeds <= 0:
        raise SystemExit(f"--num-seeds must be positive, got {args.num_seeds}")


def arrangement_bases(args: argparse.Namespace) -> list[tuple[str, dict]]:
    num_codewords = 1 << int(args.payload_bits)
    common = {"n": int(args.n), "num_codewords": num_codewords, "num_antennas": int(args.num_antennas)}
    return [
        ("ODMA", {**common, "d": int(args.odma_d), "num_blocks": int(args.odma_blocks)}),
        ("Dense", {**common, "d": int(args.n), "num_blocks": 1}),
    ]


def plot_arrangement_threshold(summary_rows: list[dict], out_path: Path, *, title: str) -> None:
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"ODMA": "#0E7490", "Dense": "#7C2D12"}
    markers = {"ODMA": "o", "Dense": "D"}
    line_styles = {0.05: "-", 0.10: "--"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in summary_rows:
        groups.setdefault((row["arrangement"], float(row["target_pupe"])), []).append(row)

    for (arrangement, target), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        rows = sorted(rows, key=lambda row: row["num_devices_active"])
        x = np.array([row["num_devices_active"] for row in rows], dtype=float)
        y = np.array([row["required_ebn0_db"] for row in rows], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        any_data = True
        ls = line_styles.get(round(target, 2), ":" if target > 0.05 else "-")
        ax.plot(x[finite], y[finite], marker=markers.get(arrangement, "o"), lw=2, ms=5,
                color=colors.get(arrangement, "#333333"), ls=ls,
                label=f"{arrangement}, PUPE<={target:g}")

    ax.set_xlabel("Active devices K")
    ax.set_ylabel("Required Eb/N0 (dB)")
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    out_dir = RESULTS_DIR / "threshold" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Decoder      : {args.decoder}")
    print(f"Payload bits : {args.payload_bits}  (M={1 << int(args.payload_bits)})")
    print(f"K values     : {args.K_values}")
    print(f"Targets      : {args.targets}")
    print(f"Eb/N0 bracket: [{args.ebn0_min}, {args.ebn0_max}]  tol={args.ebn0_tol}")
    print(f"Seeds        : {seeds}")
    print(f"Output       : {out_dir}\n")

    t0 = time.time()
    all_trials = []
    all_summary = []
    for arrangement, base in arrangement_bases(args):
        print(f"=== {arrangement}: n={base['n']}, d={base['d']}, blocks={base['num_blocks']}, "
              f"antennas={base['num_antennas']} ===")
        trials, summary = run_bisect_search(
            base, [args.decoder], args.K_values, args.targets, seeds,
            float(args.ebn0_min), float(args.ebn0_max), float(args.ebn0_tol), int(args.max_search_steps))
        for row in trials:
            row["arrangement"] = arrangement
        for row in summary:
            row["arrangement"] = arrangement
            row["base"] = base
        all_trials.extend(trials)
        all_summary.extend(summary)
        print()

    payload = {"payload_bits": int(args.payload_bits), "K_values": args.K_values, "targets": args.targets,
               "decoder": args.decoder, "ebn0_min": args.ebn0_min, "ebn0_max": args.ebn0_max,
               "ebn0_tol": args.ebn0_tol, "seeds": seeds, "arrangements": arrangement_bases(args),
               "trials": all_trials, "summary": all_summary}
    (out_dir / "arrangement_threshold_summary.json").write_text(json.dumps(payload, indent=2, default=str))

    title = (f"{args.decoder}: ODMA vs dense required Eb/N0 "
             f"(n={args.n}, bits={args.payload_bits}, M_ant={args.num_antennas})")
    plot_arrangement_threshold(all_summary, out_dir / "arrangement_required_ebn0.png", title=title)

    print("Required Eb/N0 summary:")
    for row in all_summary:
        req = row["required_ebn0_db"]
        req_str = f"{req:.2f} dB" if np.isfinite(req) else "not reached"
        print(f"  {row['arrangement']:<6s} {row['decoder']:<14s} K={row['num_devices_active']:<4d} "
              f"PUPE<={row['target_pupe']:<5g} {req_str}  [{row['search_status']}]")
    print(f"\nWrote {out_dir / 'arrangement_threshold_summary.json'}")
    print(f"Wrote {out_dir / 'arrangement_required_ebn0.png'}")
    print(f"Total wall: {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
