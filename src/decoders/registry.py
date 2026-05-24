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

from . import admm, amp, block_cd, block_map_decoder, graph_bp, lmmse, omp, sic, vamp


DecoderFn = Callable[..., tuple]


DECODER_REGISTRY: dict[str, dict] = {
    "Graph-BP": {
        "fn": graph_bp.run,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 20, "damping": 0.3},
    },
    "ADMM-Poisson": {
        "fn": admm.run_poisson,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4, "alpha_lam": 0.1},
    },
    "ADMM-Multinom": {
        "fn": admm.run_multinomial,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4},
    },
    "Residual-MAP": {
        "fn": admm.run_residual_map,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 30, "tol": 1e-6},
    },
    "ADMM-KDP-OracleK": {
        "fn": admm.run_kdp_oracle,
        "oracle": "K",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4},
    },
    "ADMM-KDP-SoftK": {
        "fn": admm.run_kdp_soft,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4},
    },
    "ADMM-KDP-SpectralRho": {
        "fn": admm.run_kdp_spectral_rho,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4},
    },
    "ADMM-KDP-Anderson": {
        "fn": admm.run_kdp_anderson,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4},
    },
    "ADMM-KDP-SRA": {
        "fn": admm.run_kdp_sra,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50, "tol": 1e-4},
    },
    "BlockCD-OracleK": {
        "fn": block_cd.run_oracle_k,
        "oracle": "K",
        "kind": "iterative",
        "params": {"max_iter": 30, "tol": 1e-6},
    },
    "BlockCD-SoftK": {
        "fn": block_cd.run_soft_k,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 30, "tol": 1e-6},
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
    "VAMP-BG-OracleK": {
        "fn": vamp.run_bg_oracle_k,
        "oracle": "K",
        "kind": "iterative",
        "params": {"max_iter": 50},
    },
    "VAMP-BG-EMRho": {
        "fn": vamp.run_bg_em_rho,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50},
    },
    "VAMP-BG-EMRhoSigma": {
        "fn": vamp.run_bg_em_rho_sigma,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50},
    },
    "VAMP-BG-EMAll": {
        "fn": vamp.run_bg_em_all,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50},
    },
    "VAMP-Poisson-EM": {
        "fn": vamp.run_poisson_em,
        "oracle": "none",
        "kind": "iterative",
        "params": {"max_iter": 50},
    },
    "BlockMAP": {
        "fn": block_map_decoder.run_poisson,
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
    "NNOMP-OracleK": {
        "fn": omp.run_oracle_k,
        "oracle": "K",
        "kind": "greedy",
        "params": {},
    },
}


PALETTE = {
    "Graph-BP": "#4C78C8",
    "ADMM-Poisson": "#3BAA5C",
    "ADMM-Multinom": "#00A6D6",
    "Residual-MAP": "#4B7F00",
    "ADMM-KDP-OracleK": "#1D4ED8",
    "ADMM-KDP-SoftK": "#0D9488",
    "ADMM-KDP-SpectralRho": "#7C3AED",
    "ADMM-KDP-Anderson": "#C026D3",
    "ADMM-KDP-SRA": "#BE123C",
    "BlockCD-OracleK": "#A855F7",
    "BlockCD-SoftK": "#8C564B",
    "LMMSE-2":  "#E07B2A",
    "LMMSE-3":  "#A6761D",
    "LMMSE-4":  "#C84C4C",
    "SIC":      "#FF7F0E",
    "AMP-BG":   "#F5A623",
    "VAMP-BG-OracleK": "#14B8A6",
    "VAMP-BG-EMRho": "#0891B2",
    "VAMP-BG-EMRhoSigma": "#0E7490",
    "VAMP-BG-EMAll": "#155E75",
    "VAMP-Poisson-EM": "#0F766E",
    "BlockMAP": "#D0021B",
    "NNOMP":    "#417505",
    "NNOMP-OracleK": "#2F6B00",
}


LINESTYLE = {
    "Graph-BP": "-",
    "ADMM-Poisson": "-",
    "ADMM-Multinom": "--",
    "Residual-MAP": "-.",
    "ADMM-KDP-OracleK": "--",
    "ADMM-KDP-SoftK": "-",
    "ADMM-KDP-SpectralRho": ":",
    "ADMM-KDP-Anderson": "-.",
    "ADMM-KDP-SRA": ":",
    "BlockCD-OracleK": "--",
    "BlockCD-SoftK": "-",
    "LMMSE-2":  "--",
    "LMMSE-3":  "--",
    "LMMSE-4":  "--",
    "SIC":      "-.",
    "AMP-BG":   ":",
    "VAMP-BG-OracleK": "--",
    "VAMP-BG-EMRho": "-",
    "VAMP-BG-EMRhoSigma": "-.",
    "VAMP-BG-EMAll": ":",
    "VAMP-Poisson-EM": "--",
    "BlockMAP": ":",
    "NNOMP":    "-",
    "NNOMP-OracleK": "--",
}


MARKER = {
    "Graph-BP": "o",
    "ADMM-Poisson": "o",
    "ADMM-Multinom": "s",
    "Residual-MAP": "^",
    "ADMM-KDP-OracleK": "D",
    "ADMM-KDP-SoftK": "P",
    "ADMM-KDP-SpectralRho": "X",
    "ADMM-KDP-Anderson": "v",
    "ADMM-KDP-SRA": "<",
    "BlockCD-OracleK": ">",
    "BlockCD-SoftK": "h",
    "LMMSE-2": "1",
    "LMMSE-3": "2",
    "LMMSE-4": "3",
    "SIC": "x",
    "AMP-BG": "*",
    "VAMP-BG-OracleK": "d",
    "VAMP-BG-EMRho": "s",
    "VAMP-BG-EMRhoSigma": "^",
    "VAMP-BG-EMAll": "v",
    "VAMP-Poisson-EM": "P",
    "BlockMAP": "p",
    "NNOMP": "H",
    "NNOMP-OracleK": "D",
}


def all_names() -> list[str]:
    return list(DECODER_REGISTRY.keys())


def get(name: str) -> dict:
    if name not in DECODER_REGISTRY:
        raise KeyError(f"Unknown decoder '{name}'. Known: {sorted(DECODER_REGISTRY)}")
    return DECODER_REGISTRY[name]
