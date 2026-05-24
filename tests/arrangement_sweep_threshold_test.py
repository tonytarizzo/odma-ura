"""Multi-arrangement Eb/N0 threshold experiment for HPC-style runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402
from src.decoders.registry import all_names  # noqa: E402
from tests.plot_arrangement_sweep import plot_series  # noqa: E402
from tests.threshold_test import run_bisect_search  # noqa: E402


def parse_arrangement(spec: str) -> tuple[str, int, int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("arrangements must be d:blocks or label:d:blocks")
    if len(parts) == 2:
        d, blocks = int(parts[0]), int(parts[1])
        return f"d={d}, blocks={blocks}", d, blocks
    label, d, blocks = parts[0], int(parts[1]), int(parts[2])
    return label, d, blocks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-B", "--payload-bits", type=int, required=True,
                   help="payload bits per active user; the explicit message alphabet is M=2^B")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--arrangements", nargs="+", type=parse_arrangement, required=True,
                   help="list of d:blocks or label:d:blocks entries, e.g. 128:16 dense:1024:1")
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--decoder", default="NNOMP-OracleK", choices=all_names())
    p.add_argument("--K-values", nargs="+", type=int, required=True)
    p.add_argument("--target", type=float, default=0.05)
    p.add_argument("--ebn0-min", type=float, default=-4.0)
    p.add_argument("--ebn0-max", type=float, default=4.0)
    p.add_argument("--ebn0-tol", type=float, default=0.1)
    p.add_argument("--max-search-steps", type=int, default=16)
    p.add_argument("--num-seeds", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--out-name", required=True, help="subdirectory under results/threshold")
    p.add_argument("--out-dir", default=None, help="explicit output directory; overrides --out-name when provided")
    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.payload_bits <= 0:
        raise SystemExit(f"--payload-bits must be positive, got {args.payload_bits}")
    if args.n <= 0:
        raise SystemExit(f"--n must be positive, got {args.n}")
    if args.num_antennas < 2:
        raise SystemExit(f"--num-antennas={args.num_antennas} is not supported by the V2 common-signature model")
    if any(K <= 0 for K in args.K_values):
        raise SystemExit(f"--K-values must be positive, got {args.K_values}")
    if args.target <= 0.0 or args.target >= 1.0:
        raise SystemExit(f"--target must lie strictly between 0 and 1, got {args.target}")
    if not np.isfinite(args.ebn0_min) or not np.isfinite(args.ebn0_max) or args.ebn0_min >= args.ebn0_max:
        raise SystemExit(f"invalid bisection bracket: [{args.ebn0_min}, {args.ebn0_max}]")
    if args.ebn0_tol <= 0.0:
        raise SystemExit(f"--ebn0-tol must be positive, got {args.ebn0_tol}")
    if args.max_search_steps <= 0:
        raise SystemExit(f"--max-search-steps must be positive, got {args.max_search_steps}")
    if args.num_seeds <= 0:
        raise SystemExit(f"--num-seeds must be positive, got {args.num_seeds}")
    for label, d, blocks in args.arrangements:
        if d <= 0 or d > args.n:
            raise SystemExit(f"invalid arrangement {label}: d={d}, n={args.n}")
        if blocks <= 0:
            raise SystemExit(f"invalid arrangement {label}: blocks={blocks}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    num_codewords = 1 << int(args.payload_bits)
    out_dir = Path(args.out_dir) if args.out_dir is not None else RESULTS_DIR / "threshold" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Decoder      : {args.decoder}")
    print(f"Payload bits : {args.payload_bits}  (M={num_codewords})")
    print(f"n            : {args.n}")
    print(f"Arrangements : {args.arrangements}")
    print(f"K values     : {args.K_values}")
    print(f"Target       : {args.target}")
    print(f"Eb/N0 bracket: [{args.ebn0_min}, {args.ebn0_max}]  tol={args.ebn0_tol}")
    print(f"Seeds        : {seeds}")
    print(f"Output       : {out_dir}\n")

    t0 = time.time()
    all_trials = []
    all_summary = []
    plot_rows: list[tuple[str, list[dict]]] = []

    for label, d, blocks in args.arrangements:
        base = {"n": int(args.n), "d": int(d), "num_blocks": int(blocks),
                "num_codewords": num_codewords, "num_antennas": int(args.num_antennas)}
        print(f"=== {label}: n={base['n']}, d={base['d']}, blocks={base['num_blocks']}, "
              f"antennas={base['num_antennas']} ===")
        trials, summary = run_bisect_search(
            base, [args.decoder], args.K_values, [args.target], seeds,
            float(args.ebn0_min), float(args.ebn0_max), float(args.ebn0_tol), int(args.max_search_steps))
        for row in trials:
            row["arrangement"] = label
            row["base"] = base
        for row in summary:
            row["arrangement"] = label
            row["base"] = base
        all_trials.extend(trials)
        all_summary.extend(summary)
        plot_rows.append((label, summary))
        print()

    payload = {"payload_bits": int(args.payload_bits), "K_values": args.K_values, "target": float(args.target),
               "decoder": args.decoder, "ebn0_min": args.ebn0_min, "ebn0_max": args.ebn0_max,
               "ebn0_tol": args.ebn0_tol, "seeds": seeds, "arrangements": args.arrangements,
               "trials": all_trials, "summary": all_summary}
    (out_dir / "arrangement_sweep_threshold_summary.json").write_text(json.dumps(payload, indent=2, default=str))

    title = f"{args.decoder}: arrangement sweep required Eb/N0 (n={args.n}, bits={args.payload_bits}, PUPE<={args.target:g})"
    plot_series(plot_rows, out_dir / "arrangement_sweep_required_ebn0.png", title=title)

    print("Required Eb/N0 summary:")
    for row in all_summary:
        req = row["required_ebn0_db"]
        req_str = f"{req:.2f} dB" if np.isfinite(req) else "not reached"
        print(f"  {row['arrangement']:<18s} K={row['num_devices_active']:<4d} PUPE<={row['target_pupe']:<5g} "
              f"{req_str}  [{row['search_status']}]")
    print(f"\nWrote {out_dir / 'arrangement_sweep_threshold_summary.json'}")
    print(f"Wrote {out_dir / 'arrangement_sweep_required_ebn0.png'}")
    print(f"Total wall: {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
