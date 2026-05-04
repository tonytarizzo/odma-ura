"""Central decoder registry.

To add a new decoder:
  1. Implement `run(scenario, **decoder_params) -> (counts, meta)` in a module under
     `src/decoders/`.
  2. Append an entry below with a unique `name`, `oracle` tag, `kind`, and any
     default `params`.

Conventions:
  - `oracle = "none"` decoders use only Y + the codebook/pattern structure.
  - `oracle = "K,assign"` decoders need K and the per-active-user block assignment.
  - `oracle = "sigma,K"` decoders need true sigma^2 + K (for the activity rate / lambda).
  - `kind in {"iterative", "linear", "greedy", "enum"}` is informational and used
     by plotting (only iterative decoders get convergence plots).
"""

from __future__ import annotations

from typing import Callable

from . import admm, amp, graph_bp, lmmse, omp, sic


DecoderFn = Callable[..., tuple]


DECODER_REGISTRY: dict[str, dict] = {
    "Graph-BP": {
        "fn": graph_bp.run,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 20, "damping": 0.3},
    },
    "ADMM": {
        "fn": admm.run,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4, "alpha_lam": 0.1},
    },
    "LMMSE-2": {
        "fn": lmmse.run_2,
        "oracle": "K,assign",
        "kind": "linear",
        "params": {},
    },
    "LMMSE-3": {
        "fn": lmmse.run_3,
        "oracle": "K,assign",
        "kind": "linear",
        "params": {},
    },
    "LMMSE-4": {
        "fn": lmmse.run_4,
        "oracle": "K,assign",
        "kind": "linear",
        "params": {"max_kd": 800},
    },
    "SIC": {
        "fn": sic.run,
        "oracle": "none",
        "kind": "greedy",
        "params": {},
    },
    "AMP-BG": {
        "fn": amp.run_bg,
        "oracle": "sigma,K",
        "kind": "iterative",
        "params": {"max_iter": 30},
    },
    "BlockMAP": {
        "fn": amp.run_block_map,
        "oracle": "none",
        "kind": "enum",
        "params": {},
    },
    "NNOMP": {
        "fn": omp.run,
        "oracle": "none",
        "kind": "greedy",
        "params": {},
    },
}


PALETTE = {
    "Graph-BP": "#4C78C8",
    "ADMM":     "#3BAA5C",
    "LMMSE-2":  "#E07B2A",
    "LMMSE-3":  "#A6761D",
    "LMMSE-4":  "#C84C4C",
    "SIC":      "#8B5CF6",
    "AMP-BG":   "#F5A623",
    "BlockMAP": "#D0021B",
    "NNOMP":    "#417505",
}


LINESTYLE = {
    "Graph-BP": "-",
    "ADMM":     "-",
    "LMMSE-2":  "--",
    "LMMSE-3":  "--",
    "LMMSE-4":  "--",
    "SIC":      "-.",
    "AMP-BG":   ":",
    "BlockMAP": ":",
    "NNOMP":    "-",
}


def all_names() -> list[str]:
    return list(DECODER_REGISTRY.keys())


def get(name: str) -> dict:
    if name not in DECODER_REGISTRY:
        raise KeyError(f"Unknown decoder '{name}'. Known: {sorted(DECODER_REGISTRY)}")
    return DECODER_REGISTRY[name]
