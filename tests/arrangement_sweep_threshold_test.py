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
from src.ura_bound import required_ebn0_curve  # noqa: E402
from tests.plot_arrangement_sweep import bootstrap_required_ci, plot_series  # noqa: E402
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
    p.add_argument("--no-resume", action="store_true", help="ignore any existing summary and recompute every (arrangement, K) from scratch")
    p.add_argument("--ci-bootstrap", type=int, default=500,
                   help="bootstrap resamples for required-Eb/N0 CIs (0 disables the CI plot)")
    p.add_argument("--overlay-ura-bound", action="store_true", help="overlay the Polyanskiy canonical/count/strict achievability variants (src/ura_bound.py)")
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


def load_checkpoint(summary_path: Path) -> tuple[list[dict], list[dict], set[tuple[str, int]]]:
    """Return (trials, summary, done) from an existing summary file, if any.

    ``done`` is the set of (arrangement_label, K) pairs already computed, used to
    skip finished work on resume. A corrupt/partial file is treated as empty so
    a restart never crashes on a half-written checkpoint.
    """
    if not summary_path.exists():
        return [], [], set()
    try:
        data = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, ValueError):
        print(f"[resume] could not parse {summary_path}; starting fresh")
        return [], [], set()
    trials = list(data.get("trials", []))
    summary = list(data.get("summary", []))
    done = {(row["arrangement"], int(row["num_devices_active"])) for row in summary}
    if done:
        print(f"[resume] loaded {len(summary)} completed (arrangement, K) points from {summary_path}")
    return trials, summary, done


def write_summary(summary_path: Path, args: argparse.Namespace, seeds: list[int],
                  all_trials: list[dict], all_summary: list[dict]) -> None:
    payload = {"payload_bits": int(args.payload_bits), "K_values": args.K_values, "target": float(args.target),
               "decoder": args.decoder, "ebn0_min": args.ebn0_min, "ebn0_max": args.ebn0_max,
               "ebn0_tol": args.ebn0_tol, "seeds": seeds, "arrangements": args.arrangements,
               "trials": all_trials, "summary": all_summary}
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(summary_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    num_codewords = 1 << int(args.payload_bits)
    out_dir = Path(args.out_dir) if args.out_dir is not None else RESULTS_DIR / "threshold" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "arrangement_sweep_threshold_summary.json"
    plot_path = out_dir / "arrangement_sweep_required_ebn0.png"

    print(f"Decoder      : {args.decoder}")
    print(f"Payload bits : {args.payload_bits}  (M={num_codewords})")
    print(f"n            : {args.n}")
    print(f"Arrangements : {args.arrangements}")
    print(f"K values     : {args.K_values}")
    print(f"Target       : {args.target}")
    print(f"Eb/N0 bracket: [{args.ebn0_min}, {args.ebn0_max}]  tol={args.ebn0_tol}")
    print(f"Seeds        : {seeds}")
    print(f"Output       : {out_dir}\n")

    if args.no_resume:
        all_trials, all_summary, done = [], [], set()
    else:
        all_trials, all_summary, done = load_checkpoint(summary_path)

    title = f"{args.decoder}: arrangement sweep required Eb/N0 (n={args.n}, bits={args.payload_bits}, PUPE<={args.target:g})"

    def plot_rows_from(summary: list[dict]) -> list[tuple[str, list[dict]]]:
        rows: dict[str, list[dict]] = {}
        for row in summary:
            rows.setdefault(row["arrangement"], []).append(row)
        return [(label, rows[label]) for label, _, _ in args.arrangements if label in rows]

    t0 = time.time()
    for label, d, blocks in args.arrangements:
        base = {"n": int(args.n), "d": int(d), "num_blocks": int(blocks),
                "num_codewords": num_codewords, "num_antennas": int(args.num_antennas)}
        print(f"=== {label}: n={base['n']}, d={base['d']}, blocks={base['num_blocks']}, "
              f"antennas={base['num_antennas']} ===")
        for K in args.K_values:
            if (label, int(K)) in done:
                print(f"  [resume] skipping completed K={K}")
                continue
            # One (arrangement, K) point at a time so each result is checkpointed.
            trials, summary = run_bisect_search(
                base, [args.decoder], [int(K)], [args.target], seeds,
                float(args.ebn0_min), float(args.ebn0_max), float(args.ebn0_tol), int(args.max_search_steps))
            for row in trials:
                row["arrangement"] = label
                row["base"] = base
            for row in summary:
                row["arrangement"] = label
                row["base"] = base
            all_trials.extend(trials)
            all_summary.extend(summary)
            done.add((label, int(K)))
            write_summary(summary_path, args, seeds, all_trials, all_summary)
            for row in summary:
                req = row["required_ebn0_db"]
                req_str = f"{req:.2f} dB" if np.isfinite(req) else "not reached"
                print(f"  -> K={K:<4d} {req_str}  [{row['search_status']}]  (checkpointed)")
        print()

    series = plot_rows_from(all_summary)

    bounds = None
    if args.overlay_ura_bound:
        curve = required_ebn0_curve(int(args.n), int(args.payload_bits), list(args.K_values),
                                    float(args.target), num_antennas=int(args.num_antennas))
        bounds = {v: [(K, entry[v]["ebn0_db_experiment"]) for K, entry in curve.items()]
                  for v in ("canonical", "count", "strict")}
        (out_dir / "ura_bound_diagnostics.json").write_text(
            json.dumps({"target": float(args.target), "num_antennas": int(args.num_antennas),
                        "curve": curve}, indent=2, default=str))
        print("[ura-bound] overlaying canonical / count / strict variants "
              "(strict truncates where its collision floor exceeds the target).")

    # Plain plot (no error bars) plus a CI variant, so the presentation choice is open.
    plot_series(series, plot_path, title=title, bounds=bounds)
    if args.ci_bootstrap > 0:
        yerr = {label: {} for label, _ in series}
        for label, rows in series:
            for r in rows:
                K = int(r["num_devices_active"])
                ci = bootstrap_required_ci(all_trials, label, K, float(args.target),
                                           num_boot=int(args.ci_bootstrap))
                if ci is not None:
                    yerr[label][K] = ci
        ci_path = out_dir / "arrangement_sweep_required_ebn0_ci.png"
        plot_series(series, ci_path, title=title + " (95% seed CI)", yerr=yerr, bounds=bounds)
        print(f"Wrote {ci_path}")

    print("Required Eb/N0 summary:")
    for row in all_summary:
        req = row["required_ebn0_db"]
        req_str = f"{req:.2f} dB" if np.isfinite(req) else "not reached"
        print(f"  {row['arrangement']:<18s} K={row['num_devices_active']:<4d} PUPE<={row['target_pupe']:<5g} "
              f"{req_str}  [{row['search_status']}]")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {plot_path}")
    print(f"Total wall: {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
