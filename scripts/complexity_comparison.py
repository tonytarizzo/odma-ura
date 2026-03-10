#!/usr/bin/env python3
"""
ODMA + URA Decoder Complexity Comparison
=========================================
Proxy operation-count models for all decoders in the V2 comparison suite.
Sweeps one parameter at a time and plots how each decoder scales.

These are NOT measured runtimes — they are analytic flop-count proxies
intended to compare asymptotic scaling behaviour across decoders.

Decoders modelled
-----------------
  Graph-BP    — iterative factor-graph BP with discrete Poisson posterior enumeration
  LMMSE-2     — ignores ODMA structure; joint M×M linear solve
  LMMSE-3     — TIN per user with overlap-aware Sherman-Morrison inversion
  LMMSE-4     — full joint vectorisation; (Kd)×(Kd) linear solve
  SIC-LS      — greedy MF-SIC with count estimation; R rounds over all (b,m) pairs
  AMP-BG      — per-block GAMP with Bernoulli-Gaussian prior; T_amp iterations
  AMP-disc    — per-block exact discrete Poisson posterior; same state enumeration as Graph-BP
  OMP-glob    — global OMP on full n×M dictionary; K OMP steps each with a growing least-squares solve

Usage
-----
  python complexity_comparison.py                          # sweep K (default)
  python complexity_comparison.py --sweep d
  python complexity_comparison.py --sweep num-codewords --save scaling_M.png
  python complexity_comparison.py --sweep num-antennas --no-normalize
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def _block_states(L_b: int, k_max: int, c_max: int) -> float:
    """Number of enumerated states: Σ_{k=0}^{k_max} C(L_b,k) * c_max^k."""
    return sum(_comb(L_b, k) * (c_max ** k) for k in range(k_max + 1))


def _kmax_proxy(K: int, B: int, L_b: int, mode: str) -> int:
    """Proxy support truncation per block based on expected active users."""
    mu = K / max(B, 1)
    if mode == "expected":      k = max(1, math.ceil(mu))
    elif mode == "sqrt":        k = max(1, math.ceil(mu + 2.0 * math.sqrt(max(mu, 1e-9))))
    elif mode == "conservative": k = max(1, math.ceil(mu + 3.0 * math.sqrt(max(mu, 1e-9))))
    else: raise ValueError(f"Unknown kmax mode: {mode}")
    return min(k, L_b)


# ─────────────────────────────────────────────────────────────────────────────
# Per-decoder complexity models
# ─────────────────────────────────────────────────────────────────────────────

def C_graph_bp(n: int, d: int, B: int, M: int, K: int, M_ant: int, T: int, c_max: int, kmax_mode: str) -> float:
    """
    Per iteration:
      Resource updates : n × avg_degree  (LMMSE at each resource node)
      Block posteriors : B × num_states × d  (state enumeration + likelihood)
    Total: T × (n × Bd/n + B × num_states × d)  =  T × (Bd + B × num_states × d)
    """
    L_b = max(1, M // B)
    k_max = _kmax_proxy(K, B, L_b, kmax_mode)
    n_states = _block_states(L_b, k_max, c_max)
    return T * (B * d + B * n_states * d)


def C_lmmse2(n: int, d: int, B: int, M: int, K: int, M_ant: int) -> float:
    """
    Build H^T H  : O(K × M_ant²)
    Invert M×M   : O(M_ant³)
    Apply to Y   : O(n × M_ant² + n × M_ant × K)
    Total: O(M_ant³ + n × M_ant² + n × M_ant × K)
    """
    return M_ant**3 + n * M_ant**2 + n * M_ant * K


def C_lmmse3(n: int, d: int, B: int, M: int, K: int, M_ant: int) -> float:
    """
    Per user: block projection O(n × d), overlap sum O(K × d), scalar solve O(1).
    Total: O(K × (n × d + K × d))  =  O(K × d × (n + K))
    """
    return K * d * (n + K)


def C_lmmse4(n: int, d: int, B: int, M: int, K: int, M_ant: int) -> float:
    """
    Build A ∈ R^{nM × Kd}  : O(n × M_ant × K × d)
    Compute A^H A (Kd×Kd)  : O(n × M_ant × (Kd)²)  — dominant
    Solve (Kd×Kd) system   : O((Kd)³)
    Total: O((Kd)³ + n × M_ant × (Kd)²)
    """
    kd = K * d
    return kd**3 + n * M_ant * kd**2


def C_sic_ls(n: int, d: int, B: int, M: int, K: int, M_ant: int, R: int | None) -> float:
    """
    Per round:
      MF combine          : O(n × M_ant)
      Scan all (b,m) pairs: O(M × d)           (each codeword inner product)
      Subtract + update   : O(n × M_ant)
    R rounds ≈ K_unique (sparse active set).
    Total: O(R × (n × M_ant + M × d))
    """
    R = K if R is None else R
    return R * (n * M_ant + M * d)


def C_amp_bg(n: int, d: int, B: int, M: int, K: int, M_ant: int, T: int) -> float:
    """
    Per block per iteration: two matrix-vector ops with A_b (d × L_b).
    MF collapse upfront: O(n × M_ant).
    Total: O(n × M_ant + T × M × d)
    """
    return n * M_ant + T * M * d


def C_amp_disc(n: int, d: int, B: int, M: int, K: int, M_ant: int, c_max: int, kmax_mode: str) -> float:
    """
    Same state enumeration per block as Graph-BP's decode_block, called once (no iteration).
    MF collapse upfront: O(n × M_ant).
    Total: O(n × M_ant + B × num_states × d)
    """
    L_b = max(1, M // B)
    k_max = _kmax_proxy(K, B, L_b, kmax_mode)
    n_states = _block_states(L_b, k_max, c_max)
    return n * M_ant + B * n_states * d


def C_omp_glob(n: int, d: int, B: int, M: int, K: int, M_ant: int) -> float:
    """
    OMP with BIC stopping; at most K steps.
    Per step k:
      Correlation scan   : O(n × M)
      Least-squares solve: O(n × k²)  via QR update ≈ O(n × k)
    Dominant: Σ_{k=1}^{K} (n × M + n × k)  =  O(K × n × M + n × K²)
    MF collapse upfront  : O(n × M_ant)
    Total: O(n × M_ant + K × n × M)  for large M
    """
    return n * M_ant + K * n * M


# ─────────────────────────────────────────────────────────────────────────────
# Sweep + curve computation
# ─────────────────────────────────────────────────────────────────────────────

DECODERS = ["Graph-BP", "LMMSE-2", "LMMSE-3", "LMMSE-4", "SIC-LS", "AMP-BG", "AMP-disc", "OMP-glob"]

PALETTE = {
    "Graph-BP":  "#4C78C8",
    "LMMSE-2":   "#E07B2A",
    "LMMSE-3":   "#3BAA5C",
    "LMMSE-4":   "#C84C4C",
    "SIC-LS":    "#8B5CF6",
    "AMP-BG":    "#F5A623",
    "AMP-disc":  "#D0021B",
    "OMP-glob":  "#417505",
}


def compute_curves(args, sweep_name: str, sweep_vals: np.ndarray) -> dict[str, np.ndarray]:
    curves: dict[str, list[float]] = {name: [] for name in DECODERS}

    for val in sweep_vals:
        p = dict(n=args.n, d=args.d, B=args.num_blocks, M=args.num_codewords,
                 K=args.num_devices_active, M_ant=args.num_antennas)
        key_map = {"n": "n", "d": "d", "num-blocks": "B", "num-codewords": "M",
                   "num-devices-active": "K", "num-antennas": "M_ant"}
        p[key_map[sweep_name]] = int(val)

        curves["Graph-BP"].append(C_graph_bp(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"], args.graph_iters, args.graph_cmax, args.graph_kmax_mode))
        curves["LMMSE-2"].append(C_lmmse2(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"]))
        curves["LMMSE-3"].append(C_lmmse3(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"]))
        curves["LMMSE-4"].append(C_lmmse4(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"]))
        curves["SIC-LS"].append(C_sic_ls(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"], args.sic_rounds))
        curves["AMP-BG"].append(C_amp_bg(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"], args.amp_iters))
        curves["AMP-disc"].append(C_amp_disc(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"], args.graph_cmax, args.graph_kmax_mode))
        curves["OMP-glob"].append(C_omp_glob(p["n"], p["d"], p["B"], p["M"], p["K"], p["M_ant"]))

    return {k: np.array(v, dtype=np.float64) for k, v in curves.items()}


def sweep_values(base: int | float, name: str) -> np.ndarray:
    """Sensible sweep range for each parameter."""
    base = max(1, int(base))
    if name == "d":
        vals = [4, 8, 12, 16, 24, 32, 48, 64]
    elif name == "num-antennas":
        vals = [1, 2, 4, 8, 16, 32]
    elif name == "num-devices-active":
        vals = np.linspace(1, max(4, int(2.5 * base)), 10).astype(int).tolist()
    elif name == "num-codewords":
        vals = np.geomspace(max(8, base // 4), max(16, 4 * base), 10).astype(int).tolist()
    elif name == "num-blocks":
        vals = np.geomspace(max(2, base // 4), max(4, 4 * base), 10).astype(int).tolist()
    elif name == "n":
        vals = np.geomspace(max(8, base // 4), max(16, 4 * base), 10).astype(int).tolist()
    else:
        raise ValueError(f"Unsupported sweep: {name}")
    return np.unique(vals).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_LABELS = {
    "n": "Resource grid size n",
    "d": "Codeword / block length d",
    "num-blocks": "Num blocks B",
    "num-codewords": "Num codewords M",
    "num-devices-active": "Active devices K",
    "num-antennas": "Num antennas M_ant",
}

ORACLE_NOTES = {
    "Graph-BP": "none",
    "LMMSE-2":  "K, assign",
    "LMMSE-3":  "K, assign",
    "LMMSE-4":  "K, assign",
    "SIC-LS":   "none",
    "AMP-BG":   "σ², K",
    "AMP-disc": "σ², K",
    "OMP-glob": "none",
}


def plot_curves(sweep_name: str, sweep_vals: np.ndarray, curves: dict[str, np.ndarray],
                normalize: bool, logy: bool, save: str | None, args) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, y in curves.items():
        y_plot = y / max(y[0], 1e-12) if normalize else y
        label = f"{name}  [{ORACLE_NOTES[name]}]"
        ax.plot(sweep_vals, y_plot, marker="o", lw=2, ms=5, label=label, color=PALETTE[name])

    ax.set_xlabel(SWEEP_LABELS.get(sweep_name, sweep_name), fontsize=11)
    ax.set_ylabel("Relative complexity (normalised to first point)" if normalize else "Proxy flop count", fontsize=10)
    ax.set_title(
        f"Decoder complexity vs {SWEEP_LABELS.get(sweep_name, sweep_name)}\n"
        f"n={args.n}, d={args.d}, B={args.num_blocks}, M={args.num_codewords}, "
        f"K={args.num_devices_active}, M_ant={args.num_antennas}  "
        f"(BP iters={args.graph_iters}, AMP iters={args.amp_iters}, c_max={args.graph_cmax})",
        fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, title="Decoder  [oracle]", title_fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Saved → {out}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ODMA+URA decoder proxy complexity comparison")
    # Simulation parameters (mirror graph_based_decoder_v2_full.py)
    parser.add_argument("--n", type=int, default=128, help="resource grid size")
    parser.add_argument("--d", type=int, default=16, help="codeword / block length")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-codewords", type=int, default=64)
    parser.add_argument("--num-devices-active", type=int, default=10)
    parser.add_argument("--num-antennas", type=int, default=4)
    # Complexity model parameters
    parser.add_argument("--graph-iters", type=int, default=15, help="proxy BP iteration count")
    parser.add_argument("--amp-iters", type=int, default=20, help="proxy AMP iteration count")
    parser.add_argument("--graph-cmax", type=int, default=2, help="max count per active message in state enumeration")
    parser.add_argument("--graph-kmax-mode", choices=["expected", "sqrt", "conservative"], default="sqrt", help="support truncation model")
    parser.add_argument("--sic-rounds", type=int, default=None, help="SIC rounds; defaults to K_active")
    # Plot controls
    parser.add_argument("--sweep", default="num-devices-active", choices=["n", "d", "num-blocks", "num-codewords", "num-devices-active", "num-antennas"], help="parameter to sweep on x-axis")
    parser.add_argument("--no-normalize", action="store_true", help="plot absolute proxy counts")
    parser.add_argument("--no-logy", action="store_true", help="linear y-axis")
    parser.add_argument("--save", type=str, default=None, help="path to save figure")
    args = parser.parse_args()

    base = getattr(args, args.sweep.replace("-", "_"))
    vals = sweep_values(base, args.sweep)
    curves = compute_curves(args, args.sweep, vals)

    print(f"\nSweep: {args.sweep}  →  {vals.astype(int).tolist()}")
    print(f"{'Decoder':<12s}  {'first-point proxy':>18s}  complexity formula (dominant term)")
    print("-" * 72)
    formulas = {
        "Graph-BP":  f"T × B × states × d  (T={args.graph_iters}, c_max={args.graph_cmax})",
        "LMMSE-2":   "M_ant³ + n·M_ant²",
        "LMMSE-3":   "K·d·(n + K)",
        "LMMSE-4":   "(Kd)³ + n·M·(Kd)²",
        "SIC-LS":    "K × (n·M_ant + M·d)",
        "AMP-BG":    f"n·M_ant + T·M·d  (T={args.amp_iters})",
        "AMP-disc":  "n·M_ant + B × states × d  (single-shot)",
        "OMP-glob":  "n·M_ant + K·n·M",
    }
    for name, y in curves.items():
        print(f"  {name:<12s}  {y[0]:>18.3e}  {formulas[name]}")

    plot_curves(args.sweep, vals, curves, not args.no_normalize, not args.no_logy, args.save, args)


if __name__ == "__main__":
    main()
