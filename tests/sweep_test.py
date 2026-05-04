"""Run parameter sweeps and (re)generate plots from the cache.

The cache is the source of truth for plots: at the end of every run we
re-read results/cache.jsonl and plot every requested sweep using
whatever rows are present. This way the plots always reflect the full
history, not just this invocation.

Examples:
  python -m tests.sweep_test --sweeps K SNR --decoders Graph-BP NNOMP
  python -m tests.sweep_test --sweeps K --num-seeds 3 --force
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

# Allow running as a script from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import load_cache  # noqa: E402
from src.config import (  # noqa: E402
    BASE_SCENARIO,
    CACHE_PATH,
    PLOTS_DIR,
    SWEEP_CONFIGS,
)
from src.decoders.registry import all_names, get  # noqa: E402
from src.plotting import plot_sweep_lines  # noqa: E402
from src.sweep import run_grid  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps", nargs="+", default=list(SWEEP_CONFIGS.keys()),
                   choices=list(SWEEP_CONFIGS.keys()),
                   help="which sweeps to run / plot")
    p.add_argument("--decoders", nargs="+", default=all_names(),
                   choices=all_names(), help="decoders to evaluate")
    p.add_argument("--num-seeds", type=int, default=1,
                   help="seeds per sweep point: 42, 43, ... 42+N-1")
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--bp-timeout", type=float, default=180.0,
                   help="wall-clock timeout for iterative decoders (s)")

    # Base scenario overrides (applied to every sweep point not currently swept)
    for key, default in BASE_SCENARIO.items():
        flag = "--" + key.replace("_", "-")
        kind = float if isinstance(default, float) else int
        p.add_argument(flag, type=kind, default=default)

    p.add_argument("--force", action="store_true",
                   help="re-run even if cache hit exists")
    p.add_argument("--plot-only", action="store_true",
                   help="skip running, just regenerate plots from the cache")
    return p.parse_args(argv)


def base_cfg_from_args(args) -> dict:
    return {key: getattr(args, key) for key in BASE_SCENARIO}


def decoder_overrides(args) -> dict[str, dict]:
    """Inject the wall-clock timeout into iterative decoders that accept it."""
    out: dict[str, dict] = {}
    for name in args.decoders:
        spec = get(name)
        sig = inspect.signature(spec["fn"])
        if "max_wall_seconds" in sig.parameters:
            out[name] = {"max_wall_seconds": args.bp_timeout}
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base = base_cfg_from_args(args)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    print(f"Base scenario: {base}")
    print(f"Decoders     : {args.decoders}")
    print(f"Sweeps       : {args.sweeps}")
    print(f"Seeds        : {seeds}")
    print(f"Cache        : {CACHE_PATH}")
    print(f"Plots        : {PLOTS_DIR}")
    print(f"Force rerun  : {args.force}\n")

    overrides = decoder_overrides(args)
    t_total = time.time()

    if not args.plot_only:
        for sweep_name in args.sweeps:
            sc = SWEEP_CONFIGS[sweep_name]
            param, values = sc["param"], sc["values"]
            print(f"=== Sweep: {sweep_name}  ({param} in {values}) ===")
            scenario_cfgs = [{**base, param: v} for v in values]
            run_grid(scenario_cfgs, args.decoders, seeds,
                     decoder_overrides=overrides,
                     cache_path=CACHE_PATH, force=args.force, verbose=True)
            print()

    cache = load_cache(CACHE_PATH)
    rows = list(cache.values())
    print(f"Cache rows: {len(rows)}  ->  generating plots in {PLOTS_DIR}")
    for sweep_name in args.sweeps:
        sc = SWEEP_CONFIGS[sweep_name]
        sweep_dir = PLOTS_DIR / sweep_name
        plot_sweep_lines(
            rows,
            swept_param=sc["param"], values=sc["values"],
            decoders=args.decoders, scenario_filter=base,
            out_dir=sweep_dir, sweep_label=sc["label"],
            metrics=("f1", "l1_acc"))
        print(f"  -> {sweep_dir}/")

    print(f"\nTotal wall: {(time.time() - t_total) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
