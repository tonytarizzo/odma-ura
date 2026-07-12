"""Direct URA-style Eb/N0 threshold experiment.

For each active-user load K, this script estimates mean PUPE as a function of Eb/N0. In grid mode it reports the
smallest supplied grid point whose mean PUPE is at or below each requested target. In bisect mode it searches an
Eb/N0 interval adaptively. It runs decoders directly and does not use the append-only cache; decoder exceptions
are experiment failures, not zero-score data points.

Example:
  uv run python -m tests.threshold_test --payload-bits 9 --n 128 --K-values 5 10 20 \
    --ebn0-grid -2 0 2 4 6 --num-seeds 5
  uv run python -m tests.threshold_test --search bisect --ebn0-min -8 --ebn0-max 8 --ebn0-tol 0.1
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

from src.config import BASE_SCENARIO, RESULTS_DIR  # noqa: E402
from src.decoders.registry import all_names, get  # noqa: E402
from src.metrics import evaluate_counts  # noqa: E402
from src.plotting import plot_required_ebn0  # noqa: E402
from src.scenario import build_scenario  # noqa: E402
from src.signal import ebn0_db_to_esn0_db  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoders", nargs="+", default=["NNOMP-OracleK", "NNOMP", "VAMP-BG-EMRhoSigma"],
                   choices=all_names())
    p.add_argument("-B", "--payload-bits", type=int, default=int(math.log2(BASE_SCENARIO["num_codewords"])),
                   help="payload bits per active user; the explicit message alphabet is M=2^B")
    p.add_argument("--K-values", nargs="+", type=int, default=[25, 50, 100, 150, 200, 300])
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[-2, 0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--search", choices=("grid", "bisect"), default="grid")
    p.add_argument("--ebn0-min", type=float, default=None, help="lower Eb/N0 bracket for --search bisect")
    p.add_argument("--ebn0-max", type=float, default=None, help="upper Eb/N0 bracket for --search bisect")
    p.add_argument("--ebn0-tol", type=float, default=0.25, help="Eb/N0 bracket tolerance in dB for --search bisect")
    p.add_argument("--max-search-steps", type=int, default=12, help="maximum bisection steps per decoder/K/target")
    p.add_argument("--targets", nargs="+", type=float, default=[0.05, 0.10])
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=42)

    for key, default in BASE_SCENARIO.items():
        if key in ("num_devices_active", "num_codewords", "esn0_db"):
            continue
        flag = "--" + key.replace("_", "-")
        kind = float if isinstance(default, float) else int
        p.add_argument(flag, type=kind, default=default)

    p.add_argument("--out-name", default="ura_threshold", help="subdirectory under results/threshold")
    return p.parse_args(argv)


def base_cfg_from_args(args: argparse.Namespace) -> dict:
    base = {key: getattr(args, key) for key in BASE_SCENARIO
            if key not in ("num_devices_active", "num_codewords", "esn0_db")}
    base["num_codewords"] = 1 << int(args.payload_bits)
    return base


def validate_args(args: argparse.Namespace, base: dict) -> None:
    if int(args.payload_bits) <= 0:
        raise SystemExit(f"--payload-bits must be positive, got {args.payload_bits}")
    if int(base["num_antennas"]) < 2:
        raise SystemExit(f"--num-antennas={base['num_antennas']} is not supported by the V2 common-signature model")
    if int(base["n"]) <= 0 or int(base["d"]) <= 0 or int(base["d"]) > int(base["n"]):
        raise SystemExit(f"invalid resource geometry: n={base['n']}, d={base['d']}")
    if any(K <= 0 for K in args.K_values):
        raise SystemExit(f"--K-values must be positive, got {args.K_values}")
    if args.num_seeds <= 0:
        raise SystemExit(f"--num-seeds must be positive, got {args.num_seeds}")
    if args.search == "grid" and any((not np.isfinite(v)) for v in args.ebn0_grid):
        raise SystemExit(f"--ebn0-grid must be finite, got {args.ebn0_grid}")
    if args.search == "bisect":
        if args.ebn0_min is None or args.ebn0_max is None:
            raise SystemExit("--search bisect requires --ebn0-min and --ebn0-max")
        if not np.isfinite(args.ebn0_min) or not np.isfinite(args.ebn0_max) or args.ebn0_min >= args.ebn0_max:
            raise SystemExit(f"invalid bisection bracket: [{args.ebn0_min}, {args.ebn0_max}]")
        if args.ebn0_tol <= 0.0:
            raise SystemExit(f"--ebn0-tol must be positive, got {args.ebn0_tol}")
        if args.max_search_steps <= 0:
            raise SystemExit(f"--max-search-steps must be positive, got {args.max_search_steps}")
    if any((t <= 0.0 or t >= 1.0) for t in args.targets):
        raise SystemExit(f"--targets must lie strictly between 0 and 1, got {args.targets}")


def run_trial_point(base: dict, decoder: str, K: int, ebn0_db: float, seeds: list[int],
                    *, progress_prefix: str = "") -> tuple[list[dict], dict]:
    cfg = {**base, "num_devices_active": int(K),
           "esn0_db": float(ebn0_db_to_esn0_db(ebn0_db, base["d"], base["num_codewords"]))}
    trials = []
    for seed in seeds:
        scenario = build_scenario(seed=seed, **cfg)
        spec = get(decoder)
        t0 = time.time()
        counts, _ = spec["fn"](scenario, **spec.get("params", {}))
        metrics = evaluate_counts(scenario.message_counts, np.asarray(counts), max_list_size=int(K))
        wall = time.time() - t0
        metrics["wall_s"] = wall
        trials.append({
            "scenario": cfg,
            "decoder": decoder,
            "seed": int(seed),
            "ebn0_db": float(ebn0_db),
            "metrics": metrics,
        })
        print(f"{progress_prefix}{decoder:<18s} K={K:<4d} Eb/N0={ebn0_db:>7.3f} "
              f"seed={seed} PUPE={metrics['pupe']:.4f} L1err={metrics['l1_err']:.4f} "
              f"rawL={metrics['raw_list_size']} ({wall:.2f}s)", flush=True)
    point = summarize_point(trials, decoder, K, ebn0_db)
    return trials, point


def summarize_point(rows: list[dict], decoder: str, K: int, ebn0_db: float) -> dict:
    pupe = [r["metrics"]["pupe"] for r in rows]
    l1_err = [r["metrics"]["l1_err"] for r in rows]
    raw_list_size = [r["metrics"]["raw_list_size"] for r in rows]
    list_overflow = [r["metrics"]["list_overflow"] for r in rows]
    mean_pupe = float(np.mean(pupe))
    pupe_se = float(np.std(pupe, ddof=1) / math.sqrt(len(pupe))) if len(pupe) > 1 else float("nan")
    return {
        "ebn0_db": float(ebn0_db),
        "mean_pupe": mean_pupe,
        "pupe_seed_se": pupe_se,
        "pupe_seed_ci95": [max(0.0, mean_pupe - 1.96 * pupe_se), min(1.0, mean_pupe + 1.96 * pupe_se)]
        if np.isfinite(pupe_se) else [float("nan"), float("nan")],
        "mean_l1_err": float(np.mean(l1_err)),
        "mean_raw_list_size": float(np.mean(raw_list_size)),
        "mean_list_overflow": float(np.mean(list_overflow)),
        "num_trials": len(rows),
        "decoder": decoder,
        "num_devices_active": int(K),
    }


def run_grid_trials(base: dict, decoders: list[str], K_values: list[int],
                    ebn0_grid: list[float], seeds: list[int]) -> list[dict]:
    trials = []
    total = len(K_values) * len(ebn0_grid) * len(seeds) * len(decoders)
    done = 0
    for K in K_values:
        for ebn0_db in ebn0_grid:
            for decoder in decoders:
                point_trials, _ = run_trial_point(
                    base, decoder, K, ebn0_db, seeds,
                    progress_prefix=f"  [{done + 1:4d}/{total}] ")
                done += len(seeds)
                trials.extend(point_trials)
    return trials


def summarize_thresholds(trials: list[dict], decoders: list[str], K_values: list[int],
                         ebn0_grid: list[float], targets: list[float]) -> list[dict]:
    summary = []
    for decoder in decoders:
        for K in K_values:
            curve = []
            for ebn0_db in ebn0_grid:
                rows = [r for r in trials if r["decoder"] == decoder
                        and r["scenario"]["num_devices_active"] == K and r["ebn0_db"] == ebn0_db]
                curve.append(summarize_point(rows, decoder, K, ebn0_db))
            for target in targets:
                feasible = [pt for pt in curve if pt["mean_pupe"] <= target]
                summary.append({
                    "decoder": decoder,
                    "num_devices_active": int(K),
                    "target_pupe": float(target),
                    "required_ebn0_db": min((pt["ebn0_db"] for pt in feasible), default=float("nan")),
                    "curve": curve,
                })
    return summary


def run_bisect_search(base: dict, decoders: list[str], K_values: list[int], targets: list[float],
                      seeds: list[int], ebn0_min: float, ebn0_max: float,
                      ebn0_tol: float, max_steps: int) -> tuple[list[dict], list[dict]]:
    trials = []
    summary = []
    point_cache: dict[tuple[str, int, float], dict] = {}

    def eval_point(decoder: str, K: int, ebn0_db: float) -> dict:
        key = (decoder, int(K), round(float(ebn0_db), 10))
        if key in point_cache:
            return point_cache[key]
        point_trials, point = run_trial_point(base, decoder, K, ebn0_db, seeds, progress_prefix="  ")
        trials.extend(point_trials)
        point_cache[key] = point
        return point

    for decoder in decoders:
        for K in K_values:
            low_pt = eval_point(decoder, K, ebn0_min)
            high_pt = eval_point(decoder, K, ebn0_max)
            for target in targets:
                low = float(ebn0_min)
                high = float(ebn0_max)
                status = "bracketed"
                if low_pt["mean_pupe"] <= target:
                    status = "below_range"
                    required = low
                    curve = sorted([low_pt, high_pt], key=lambda p: p["ebn0_db"])
                elif high_pt["mean_pupe"] > target:
                    status = "not_reached"
                    required = float("nan")
                    curve = sorted([low_pt, high_pt], key=lambda p: p["ebn0_db"])
                else:
                    curve = [low_pt, high_pt]
                    for _ in range(max_steps):
                        if high - low <= ebn0_tol:
                            break
                        mid = 0.5 * (low + high)
                        mid_pt = eval_point(decoder, K, mid)
                        curve.append(mid_pt)
                        if mid_pt["mean_pupe"] <= target:
                            high = mid
                        else:
                            low = mid
                    required = high
                    curve = sorted(curve, key=lambda p: p["ebn0_db"])
                ordered = sorted({float(p["ebn0_db"]): p for p in curve}.values(), key=lambda p: p["ebn0_db"])
                increases = [ordered[i + 1]["mean_pupe"] - ordered[i]["mean_pupe"] for i in range(len(ordered) - 1)]
                summary.append({
                    "decoder": decoder,
                    "num_devices_active": int(K),
                    "target_pupe": float(target),
                    "required_ebn0_db": float(required),
                    "search_status": status,
                    "ebn0_min": float(ebn0_min),
                    "ebn0_max": float(ebn0_max),
                    "ebn0_tol": float(ebn0_tol),
                    "curve": ordered,
                    "monotonicity_violations": int(sum(v > 1e-12 for v in increases)),
                    "max_pupe_increase": float(max([0.0, *increases])),
                })
    return trials, summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base = base_cfg_from_args(args)
    validate_args(args, base)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    out_dir = RESULTS_DIR / "threshold" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base scenario: {base}")
    print(f"Payload bits : {args.payload_bits}  (M={base['num_codewords']})")
    print(f"Decoders     : {args.decoders}")
    print(f"K values     : {args.K_values}")
    print(f"Search       : {args.search}")
    if args.search == "grid":
        print(f"Eb/N0 grid   : {args.ebn0_grid}")
    else:
        print(f"Eb/N0 bracket: [{args.ebn0_min}, {args.ebn0_max}]  tol={args.ebn0_tol}")
    print(f"Targets      : {args.targets}")
    print(f"Seeds        : {seeds}")
    print(f"Output       : {out_dir}\n")

    t0 = time.time()
    if args.search == "grid":
        trials = run_grid_trials(base, args.decoders, args.K_values, args.ebn0_grid, seeds)
        summary = summarize_thresholds(trials, args.decoders, args.K_values, args.ebn0_grid, args.targets)
    else:
        trials, summary = run_bisect_search(
            base, args.decoders, args.K_values, args.targets, seeds,
            float(args.ebn0_min), float(args.ebn0_max), float(args.ebn0_tol), int(args.max_search_steps))
    payload = {"base": base, "payload_bits": int(args.payload_bits), "K_values": args.K_values,
               "search": args.search, "ebn0_grid": args.ebn0_grid,
               "ebn0_min": args.ebn0_min, "ebn0_max": args.ebn0_max, "ebn0_tol": args.ebn0_tol,
               "targets": args.targets, "seeds": seeds, "trials": trials, "summary": summary}
    (out_dir / "threshold_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_required_ebn0(summary, out_dir / "required_ebn0.png",
                       title=f"Required Eb/N0 vs K (n={base['n']}, bits={math.log2(base['num_codewords']):.0f}, target PUPE)")

    print("\nRequired Eb/N0 summary:")
    for row in summary:
        req = row["required_ebn0_db"]
        req_str = f"{req:.2f} dB" if np.isfinite(req) else "not reached"
        status = row.get("search_status")
        suffix = f"  [{status}]" if status else ""
        print(f"  {row['decoder']:<18s} K={row['num_devices_active']:<4d} PUPE<={row['target_pupe']:<5g} {req_str}{suffix}")
    print(f"\nWrote {out_dir / 'threshold_summary.json'}")
    print(f"Wrote {out_dir / 'required_ebn0.png'}")
    print(f"Total wall: {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
