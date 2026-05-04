"""Append-only JSONL result cache.

Layout:
  results/cache.jsonl  -- one line = one (scenario, decoder, decoder_params, seed)
                          result.

Key = sha1(canonical_json({**scenario_cfg, "decoder": d, "decoder_params": p,
                            "seed": s})).

API:
  load_cache(path)               -> dict[key, row]   (last row wins on collision)
  lookup(cache, scenario_cfg, decoder, decoder_params, seed) -> row | None
  append_result(path, row)       -> appends one JSON line, fsync()'d
  make_row(scenario_cfg, decoder, decoder_params, seed, metrics, decoder_meta=None)
                                 -> row dict ready for append
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(obj):
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Cannot serialise {type(obj).__name__} for cache key")


def hash_key(scenario_cfg: dict, decoder: str, decoder_params: dict,
             seed: int) -> str:
    payload = {
        "scenario": {k: scenario_cfg[k] for k in sorted(scenario_cfg)},
        "decoder": decoder,
        "decoder_params": {k: decoder_params[k] for k in sorted(decoder_params)},
        "seed": int(seed),
    }
    return hashlib.sha1(_canonical(payload).encode("utf-8")).hexdigest()


def load_cache(path: str | Path) -> dict[str, dict]:
    """Load JSONL cache. Later rows overwrite earlier ones with the same key."""
    p = Path(path)
    cache: dict[str, dict] = {}
    if not p.exists():
        return cache
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key:
                cache[key] = row
    return cache


def lookup(cache: dict[str, dict], scenario_cfg: dict, decoder: str,
           decoder_params: dict, seed: int) -> dict | None:
    return cache.get(hash_key(scenario_cfg, decoder, decoder_params, seed))


def make_row(scenario_cfg: dict, decoder: str, decoder_params: dict,
             seed: int, metrics: dict,
             decoder_meta: dict | None = None) -> dict:
    return {
        "key": hash_key(scenario_cfg, decoder, decoder_params, seed),
        "scenario": dict(scenario_cfg),
        "decoder": decoder,
        "decoder_params": dict(decoder_params),
        "seed": int(seed),
        "metrics": metrics,
        "decoder_meta": decoder_meta or {},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def append_result(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=_jsonable) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _jsonable(obj):
    """Default JSON serialiser: numpy scalars/arrays -> native python."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Cannot serialise {type(obj).__name__}")
