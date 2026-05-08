"""Run all (or selected) decoders on a single scenario.

Always runs decoders fresh so convergence histories can be plotted —
the cache is still updated with the resulting metrics.

Example:
  python -m tests.single_test --decoders Graph-BP NNOMP --K 20 --esn0-db 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Allow running as a script from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import append_result, make_row  # noqa: E402
from src.config import BASE_SCENARIO, CACHE_PATH, SINGLE_DIR  # noqa: E402
from src.decoders.registry import all_names, get  # noqa: E402
from src.metrics import evaluate_counts  # noqa: E402
from src.objectives import objective_diagnostics  # noqa: E402
from src.plotting import (  # noqa: E402
    plot_convergence,
    plot_count_estimates,
    plot_decoder_bars,
)
from src.scenario import build_scenario  # noqa: E402
from src.signal import esn0_db_to_ebn0_db  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoders", nargs="+", default=all_names(),
                   choices=all_names(), help="decoders to run")
    p.add_argument("--n", type=int, default=BASE_SCENARIO["n"])
    p.add_argument("--d", type=int, default=BASE_SCENARIO["d"])
    p.add_argument("--num-blocks", type=int, default=BASE_SCENARIO["num_blocks"])
    p.add_argument("--num-codewords", type=int, default=BASE_SCENARIO["num_codewords"])
    p.add_argument("-K", "--num-devices-active", dest="num_devices_active",
                   type=int, default=BASE_SCENARIO["num_devices_active"])
    p.add_argument("--num-antennas", type=int, default=BASE_SCENARIO["num_antennas"])
    p.add_argument("--esn0-db", type=float, default=BASE_SCENARIO["esn0_db"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bp-timeout", type=float, default=180.0,
                   help="wall-clock timeout for iterative decoders (s)")
    p.add_argument("--out-name", type=str, default=None,
                   help="subdir name under results/single/ (default = auto slug)")
    return p.parse_args(argv)


def make_slug(args) -> str:
    return (
        f"n{args.n}_d{args.d}_B{args.num_blocks}_M{args.num_codewords}"
        f"_K{args.num_devices_active}_ant{args.num_antennas}"
        f"_snr{args.esn0_db:+.0f}dB_s{args.seed}"
    )


def print_header(args, ebn0_db: float, noise_var: float) -> None:
    print("=" * 64)
    print("ODMA + URA single-scenario test")
    print("=" * 64)
    print(f"  n={args.n}  d={args.d}  B={args.num_blocks}  M={args.num_codewords}")
    print(f"  K={args.num_devices_active}  M_ant={args.num_antennas}")
    print(f"  Es/N0={args.esn0_db:.2f} dB   Eb/N0={ebn0_db:.2f} dB")
    print(f"  sigma^2 = {noise_var:.5f}    seed={args.seed}")
    print(f"  Decoders: {', '.join(args.decoders)}")
    print()


def print_table(rows: list[tuple[str, dict, float, dict]]) -> None:
    print()
    print("=" * 78)
    hdr = "{:<10s} {:>10s} {:>5s} {:>5s} {:>5s} {:>7s} {:>7s} {:>7s} {:>8s}"
    print(hdr.format("Decoder", "Oracle", "TP", "FP", "FN",
                     "F1", "L1acc", "L1err", "wall_s"))
    print("-" * 78)
    for name, m, wall, _ in rows:
        spec = get(name)
        print("{:<10s} {:>10s} {:>5d} {:>5d} {:>5d} {:>7.3f} {:>7.3f} {:>7.3f} {:>8.2f}".format(
            name, spec.get("oracle", "?"),
            int(m.get("tp", 0)), int(m.get("fp", 0)), int(m.get("fn", 0)),
            float(m.get("f1", 0.0)), float(m.get("l1_acc", 0.0)),
            float(m.get("l1_err", float("nan"))),
            float(wall),
        ))
    print("=" * 78)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.num_antennas) < 2:
        raise SystemExit(
            f"--num-antennas={args.num_antennas} is not supported. "
            f"The V2 common-signature model assumes M_ant >= 2.")
    cfg = {
        "n": args.n, "d": args.d, "num_blocks": args.num_blocks,
        "num_codewords": args.num_codewords,
        "num_devices_active": args.num_devices_active,
        "num_antennas": args.num_antennas, "esn0_db": float(args.esn0_db),
    }
    ebn0_db = esn0_db_to_ebn0_db(args.esn0_db, args.d, args.num_codewords)
    scenario = build_scenario(seed=args.seed, **cfg)
    print_header(args, ebn0_db, scenario.noise_var)

    out_dir = SINGLE_DIR / (args.out_name or make_slug(args))
    out_dir.mkdir(parents=True, exist_ok=True)

    table: list[tuple[str, dict, float, dict]] = []
    counts_per_decoder: dict[str, np.ndarray] = {}

    import inspect
    for name in args.decoders:
        spec = get(name)
        params = dict(spec.get("params", {}))
        sig = inspect.signature(spec["fn"])
        if "max_wall_seconds" in sig.parameters:
            params["max_wall_seconds"] = args.bp_timeout
        print(f"-- Running {name}  oracle={spec.get('oracle', '?')}  kind={spec['kind']}")
        t0 = time.time()
        try:
            counts, meta = spec["fn"](scenario, **params)
            counts = np.asarray(counts)
            wall = time.time() - t0
            metrics = evaluate_counts(scenario.message_counts, counts)
            metrics.update(objective_diagnostics(scenario, counts))
            metrics["wall_s"] = wall
            if isinstance(meta, dict):
                for k in ("converged", "timed_out", "decoder_failure", "iterations"):
                    if k in meta:
                        metrics[k] = meta[k]
            table.append((name, metrics, wall, meta if isinstance(meta, dict) else {}))
            counts_per_decoder[name] = counts

            slim = {k: meta[k] for k in ("converged", "timed_out",
                                         "decoder_failure", "failure_reason",
                                         "iterations",
                                         "wall_s", "lam", "noise_var_est",
                                         "K_hat", "K_prior", "K_target", "K_star",
                                         "selected_k", "rho",
                                         "rho_init", "rho_policy", "rho_updates",
                                         "r_pri", "r_dual", "anderson_steps",
                                         "cache_size", "cache_caps",
                                         "max_feasible_K", "rho_activity",
                                         "support_hat", "sigma_x_sq", "sigma_eff_sq",
                                         "sigma_K", "objective")
                    if isinstance(meta, dict) and k in meta}
            row = make_row(cfg, name, params, args.seed, metrics, decoder_meta=slim)
            append_result(CACHE_PATH, row)

            if isinstance(meta, dict) and meta.get("history"):
                true_vals = {
                    "lambda": args.num_devices_active / args.num_codewords,
                    "lam":    args.num_devices_active / args.num_codewords,
                    "noise_var": scenario.noise_var,
                    "K_hat":  float(args.num_devices_active),
                }
                plot_convergence(
                    meta["history"], out_dir / f"convergence_{name}.png",
                    name, true_values=true_vals)
        except Exception as exc:
            print(f"   {name} failed: {exc}")
            traceback.print_exc()

    print_table(table)

    rows_for_bars = [{"decoder": name, "metrics": m} for name, m, _, _ in table]
    plot_decoder_bars(rows_for_bars, out_dir / "comparison_metrics.png",
                      metrics=["f1", "l1_acc"])
    if counts_per_decoder:
        plot_count_estimates(scenario.message_counts, counts_per_decoder,
                             out_dir / "comparison_counts.png")

    summary = {
        "scenario": cfg, "seed": args.seed, "ebn0_db": ebn0_db,
        "noise_var": scenario.noise_var,
        "results": {name: m for name, m, _, _ in table},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
