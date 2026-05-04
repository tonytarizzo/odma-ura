"""Sweep / scenario runner that uses the cache + decoder registry.

Two entry points:
  run_one(scenario_cfg, decoder, seed, *, decoder_overrides=None,
          cache_path=None, force=False, verbose=False)
      -> (row, status)   status in {"hit", "ran", "failed"}

  run_grid(scenario_cfgs, decoders, seeds, **kwargs)
      -> list[row]   (one per (cfg, decoder, seed))

Failures are caught: a failed run returns NaN metrics with an "error" string
in `decoder_meta` and is still appended to the cache so we don't retry it
forever (use `--force` to retry).
"""

from __future__ import annotations

import gc
import time
import traceback
from typing import Iterable

import numpy as np

from .cache import append_result, load_cache, lookup, make_row
from .config import CACHE_PATH
from .decoders.registry import get
from .metrics import evaluate_counts
from .scenario import SCENARIO_KEYS, build_scenario


def _fail_metrics() -> dict:
    return dict(tp=0, fp=0, fn=0, f1=0.0, l1_err=float("nan"),
                l1_acc=0.0, nmse=float("nan"),
                total_count_err=float("nan"), exact_count=0.0,
                support_true=0)


def _slim_meta(meta: dict) -> dict:
    """Keep only small scalar fields from decoder meta for cache rows."""
    if not meta:
        return {}
    keep = {"converged", "timed_out", "iterations", "wall_s",
            "lam", "noise_var_est", "K_hat", "selected_k", "rho"}
    return {k: meta[k] for k in keep if k in meta}


def run_one(scenario_cfg: dict, decoder: str, seed: int, *,
            decoder_overrides: dict | None = None,
            cache: dict[str, dict] | None = None,
            cache_path=CACHE_PATH,
            force: bool = False,
            verbose: bool = False) -> tuple[dict, str]:
    """Run one (scenario, decoder, seed). Returns (row, status)."""
    spec = get(decoder)
    decoder_params = {**spec.get("params", {}), **(decoder_overrides or {})}

    if not force:
        if cache is None:
            cache = load_cache(cache_path)
        hit = lookup(cache, scenario_cfg, decoder, decoder_params, seed)
        if hit is not None:
            return hit, "hit"

    scenario = build_scenario(seed=seed, **scenario_cfg)

    t0 = time.time()
    try:
        counts, meta = spec["fn"](scenario, **decoder_params)
        metrics = evaluate_counts(scenario.message_counts, np.asarray(counts))
        metrics["wall_s"] = time.time() - t0
        if isinstance(meta, dict):
            for k in ("converged", "timed_out"):
                if k in meta:
                    metrics[k] = bool(meta[k])
            if "iterations" in meta:
                metrics["iterations"] = int(meta["iterations"])
            nv = meta.get("noise_var_est")
            if nv is not None and (not np.isfinite(nv) or nv > 100 * scenario.noise_var):
                metrics["diverged"] = True
        decoder_meta = _slim_meta(meta if isinstance(meta, dict) else {})
        status = "ran"
    except Exception as exc:
        traceback.print_exc()
        metrics = _fail_metrics()
        metrics["wall_s"] = time.time() - t0
        decoder_meta = {"error": str(exc)[:200]}
        status = "failed"

    row = make_row(scenario_cfg, decoder, decoder_params, seed, metrics,
                   decoder_meta=decoder_meta)
    append_result(cache_path, row)
    if cache is not None:
        cache[row["key"]] = row

    gc.collect()
    return row, status


def run_grid(scenario_cfgs: Iterable[dict], decoders: Iterable[str],
             seeds: Iterable[int], *,
             decoder_overrides: dict[str, dict] | None = None,
             cache_path=CACHE_PATH,
             force: bool = False,
             verbose: bool = True) -> list[dict]:
    """Iterate over the cartesian product (cfg, decoder, seed)."""
    cache = load_cache(cache_path)
    rows: list[dict] = []
    cfgs = list(scenario_cfgs)
    decoders = list(decoders)
    seeds = list(seeds)
    total = len(cfgs) * len(decoders) * len(seeds)
    done = 0
    n_hit = n_ran = n_failed = 0
    decoder_overrides = decoder_overrides or {}
    for cfg in cfgs:
        cfg_norm = {k: cfg[k] for k in SCENARIO_KEYS}
        for dec in decoders:
            ov = decoder_overrides.get(dec, {})
            for seed in seeds:
                done += 1
                row, status = run_one(cfg_norm, dec, seed,
                                      decoder_overrides=ov,
                                      cache=cache, cache_path=cache_path,
                                      force=force, verbose=verbose)
                rows.append(row)
                if status == "hit":
                    n_hit += 1
                elif status == "ran":
                    n_ran += 1
                else:
                    n_failed += 1
                if verbose:
                    f1 = row["metrics"].get("f1", float("nan"))
                    l1a = row["metrics"].get("l1_acc", float("nan"))
                    wall = row["metrics"].get("wall_s", 0.0)
                    short = ", ".join(f"{k}={v}" for k, v in cfg_norm.items()
                                       if k in ("num_devices_active", "esn0_db",
                                                "num_antennas", "num_blocks",
                                                "num_codewords"))
                    print(f"  [{done:4d}/{total}] {dec:<9s} seed={seed} "
                          f"{short}  F1={f1:.3f} L1acc={l1a:.3f} "
                          f"({wall:5.1f}s) [{status}]", flush=True)
    if verbose:
        print(f"  -- done: {n_ran} ran, {n_hit} cache hits, {n_failed} failed")
    return rows
