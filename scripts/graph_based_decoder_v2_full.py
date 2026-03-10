"""
ODMA + URA Decoder Testbed — V2 Comparison Suite
=================================================
Runs five decoders on the same synthesised data and produces a full comparison.

Signal model (V2: multi-antenna, no fading, h = 1_M):
    Y[r,:] = h * s_r + z_r    s_r = Σ_{(b,j): S_b[j]=r} x_{b,j}
    h = 1_M,  z_r ~ N(0, σ² I_M)

Decoders
--------
  [A] Graph-BP   : iterative message-passing with discrete Poisson posterior.
                   No oracle — estimates K, λ, σ² from data via EM.
  [B] LMMSE-2    : Approach 2 from supervisor notes.  Ignores ODMA pattern
                   structure when forming the covariance; estimates x'_k = P_k x_k
                   jointly from Y, then extracts x̂_k = P_k^T x̂'_k.
                   Oracle: K and user→block assignments.
  [C] LMMSE-3    : Approach 3 (TIN) from supervisor notes.  Per-user P_k^T Y,
                   single-shot LMMSE with exact interference covariance from
                   pattern overlaps.  Oracle: K and user→block assignments.
  [D] LMMSE-4    : Approach 4 (joint vectorisation) from supervisor notes.
                   Exact joint LMMSE via y = Ax + z vectorisation.
                   Oracle: K and user→block assignments.
                   Matrix sizes at default args: A ∈ R^{512×160} — tractable.
  [E] SIC        : Successive interference cancellation.  Each round finds the
                   block/codeword pair with the highest matched-filter energy,
                   declares it active, subtracts its contribution, and repeats
                   until residual energy drops below a threshold.  No oracle.

LMMSE decoders [B/C/D] estimate continuous x̂_k vectors; message decisions are
made by nearest-neighbour (NN) search in the codebook.  They require oracle
knowledge of K and which block each device uses — an advantage Graph-BP and SIC
do not have.

Run:
    python graph_based_decoder_v2_compare.py --seed 42 --n 128 --d 16 \
        --num-blocks 8 --num-codewords 64 --num-devices-active 10 \
        --num-antennas 4 --esn0-db 10
"""

from __future__ import annotations
import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path
from itertools import combinations, product as iproduct
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Data-generation helpers  (identical to V2 originals)
# ═══════════════════════════════════════════════════════════════════════════════

def make_codebook(num_codewords: int, d: int, rng: np.random.Generator,
                  complex_valued: bool = False) -> np.ndarray:
    """Random Gaussian codebook with unit-normalised rows.  Returns (num_codewords, d)."""
    if complex_valued:
        raw = (rng.standard_normal((num_codewords, d))
               + 1j * rng.standard_normal((num_codewords, d)))
    else:
        raw = rng.standard_normal((num_codewords, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_odma_blocks(num_blocks: int, n: int, d: int,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Random ODMA blocks: each is a sorted array of d resource indices from {0,...,n-1}."""
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(num_blocks)]


def build_pattern_matrices(blocks: list[np.ndarray], n: int) -> dict[int, np.ndarray]:
    """Build per-block embedding matrices P_b ∈ {0,1}^{n×d}.  P_b^T P_b = I_d."""
    P_mats: dict[int, np.ndarray] = {}
    for b, S_b in enumerate(blocks):
        d = len(S_b)
        P = np.zeros((n, d), dtype=np.float64)
        P[S_b, np.arange(d)] = 1.0
        P_mats[b] = P
    return P_mats


def make_message_block_mapping(num_codewords: int, num_blocks: int):
    """Deterministic mapping: message m → block m % num_blocks."""
    msg_to_block: dict[int, int] = {m: m % num_blocks for m in range(num_codewords)}
    block_to_msg_list: dict[int, list[int]] = {b: [] for b in range(num_blocks)}
    for m in range(num_codewords):
        block_to_msg_list[m % num_blocks].append(m)
    return msg_to_block, block_to_msg_list


def sample_active_messages(num_devices_active: int, num_codewords: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Each active device independently picks a message uniformly at random."""
    return rng.integers(0, num_codewords, size=num_devices_active)


def build_message_counts(active_msgs: np.ndarray, num_codewords: int) -> np.ndarray:
    """Global message count vector — the direct decoder target.  Returns (num_codewords,)."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for m in active_msgs:
        counts[int(m)] += 1.0
    return counts


def build_block_coefficients(active_msgs: np.ndarray,
                              msg_to_block: dict[int, int],
                              block_to_msg_list: dict[int, list[int]],
                              num_blocks: int) -> dict[int, np.ndarray]:
    """Blockwise view of the decoder target: sparse multiplicity vector per block."""
    coeffs: dict[int, np.ndarray] = {}
    for b in range(num_blocks):
        msg_list = block_to_msg_list[b]
        a_b = np.zeros(len(msg_list), dtype=np.float64)
        msg_to_local = {m: i for i, m in enumerate(msg_list)}
        for m in active_msgs:
            if msg_to_block[m] == b:
                a_b[msg_to_local[m]] += 1.0
        coeffs[b] = a_b
    return coeffs


def build_block_dictionaries(codebook: np.ndarray,
                              block_to_msg_list: dict[int, list[int]],
                              num_blocks: int) -> dict[int, np.ndarray]:
    """Gather codebook rows for each block's assigned messages.  block → (L_b, d)."""
    return {b: codebook[block_to_msg_list[b]] for b in range(num_blocks)}


def esn0_db_to_noise_var(esn0_db: float, d: int) -> float:
    """Es/N0 (dB) → per-entry noise variance.  Es = 1/d (unit-norm codewords)."""
    esn0_lin = 10.0 ** (esn0_db / 10.0)
    return 1.0 / (d * esn0_lin)


def synthesize_received_signal(P_mats: dict[int, np.ndarray],
                                block_dicts: dict[int, np.ndarray],
                                block_coeffs: dict[int, np.ndarray],
                                num_antennas: int,
                                noise_var: float,
                                rng: np.random.Generator,
                                complex_valued: bool = False,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise Y = outer(y_scalar, h) + Z with h = 1_M.

    Returns (Y_noisy, Y_clean) both (n, M).
    """
    n = next(iter(P_mats.values())).shape[0]
    dtype = np.complex128 if complex_valued else np.float64
    y_scalar = np.zeros(n, dtype=dtype)
    for b in P_mats:
        a_b = block_coeffs[b]
        if np.any(a_b):
            x_b = block_dicts[b].T @ a_b        # (d,)
            y_scalar += P_mats[b] @ x_b          # (n,)
    h = np.ones(num_antennas, dtype=dtype)
    Y_clean = np.outer(y_scalar, h)              # (n, M)
    if complex_valued:
        noise = np.sqrt(noise_var / 2) * (
            rng.standard_normal((n, num_antennas))
            + 1j * rng.standard_normal((n, num_antennas)))
    else:
        noise = np.sqrt(noise_var) * rng.standard_normal((n, num_antennas))
    return Y_clean + noise, Y_clean


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER A — Graph-BP  (your original V2 decoder, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def graph_based_decoder(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    max_iter: int = 50,
    damping: float = 0.3,
    tol: float = 1e-4,
    lambda_init: float | None = None,
    noise_var_init: float | None = None,
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Iterative Gaussian resource update (matrix LMMSE) + exact blockwise discrete
    Poisson posterior, with EM updates for noise variance σ² and Poisson rate λ.

    V2 multi-antenna LMMSE at resource r:
        γ         = ||h||² = M
        denom     = σ² + γ · v_sum
        innov_r   = h^T y_r − γ · μ_s
        μ_post[k] = μ_k + v_k · innov_r / denom
        v_post[k] = v_k − v_k² · γ / denom
        τ_ext[k]  = γ / (denom − v_k · γ)

    Setting M=1, h=1 recovers the V1 scalar formula exactly.
    """
    n, num_ant = Y.shape
    dtype = Y.dtype
    var_floor = 1e-10
    tau_floor = 1e-10
    h = np.ones(num_ant, dtype=dtype)
    gamma = float(np.real(np.dot(h.conj(), h)))   # = M

    # ------------------------------------------------------------------ helpers
    def logsumexp(v: np.ndarray) -> float:
        vmax = float(np.max(v))
        return vmax + float(np.log(np.sum(np.exp(v - vmax)))) if np.isfinite(vmax) else vmax

    def poisson_pmf_vec(lam: float) -> np.ndarray:
        """Truncated Poisson PMF p(0), p(1), ..., normalised to sum to 1."""
        lam = max(float(lam), 1e-12)
        probs = [np.exp(-lam)]
        total = probs[0]
        c = 0
        while 1.0 - total > poisson_tail_tol:
            c += 1
            probs.append(probs[-1] * lam / c)
            total += probs[-1]
            if probs[-1] == 0.0:
                break
        p = np.array(probs, dtype=np.float64)
        return p / np.sum(p)

    def max_active_per_block(L_b: int, p_nz: float) -> int:
        """Largest support size k with binomial tail > support_tail_tol."""
        p_nz = float(np.clip(p_nz, 1e-12, 1.0 - 1e-12))
        p0 = 1.0 - p_nz
        pk = p0 ** L_b
        cdf = pk
        for k in range(L_b):
            pk = pk * ((L_b - k) / (k + 1)) * (p_nz / p0)
            cdf += pk
            if 1.0 - cdf <= support_tail_tol:
                return k + 1
        return L_b

    def decode_block(C_b: np.ndarray, r_b: np.ndarray,
                     v_b: np.ndarray, lam: float):
        """Exact enumeration of discrete Poisson posterior over a_b."""
        pmf = poisson_pmf_vec(lam)
        c_max = len(pmf) - 1
        L_b = C_b.shape[0]
        p_nz = float(1.0 - pmf[0])
        k_max = max_active_per_block(L_b, p_nz)
        log_pmf = np.log(pmf + 1e-300)
        states: list[np.ndarray] = [np.zeros(L_b, dtype=np.float64)]
        log_prior: list[float] = [L_b * log_pmf[0]]
        for k in range(1, k_max + 1):
            lp_zeros = (L_b - k) * log_pmf[0]
            for idxs in combinations(range(L_b), k):
                for cnts in iproduct(range(1, c_max + 1), repeat=k):
                    a = np.zeros(L_b, dtype=np.float64)
                    a[list(idxs)] = np.array(cnts, dtype=np.float64)
                    states.append(a)
                    log_prior.append(lp_zeros + sum(log_pmf[c] for c in cnts))
        A = np.array(states, dtype=np.float64)
        X = A @ C_b
        err = X - r_b[None, :]
        ll = -np.real(np.sum((np.abs(err) ** 2) / v_b[None, :], axis=1))
        log_post = ll + np.array(log_prior, dtype=np.float64)
        log_post -= logsumexp(log_post)
        w = np.exp(log_post)
        a_mean = w @ A
        x_mean = w @ X
        x_var = np.maximum(
            np.real(w @ (np.abs(X) ** 2)) - np.abs(x_mean) ** 2, var_floor)
        a_map = A[int(np.argmax(log_post))]
        return a_mean, x_mean, x_var, a_map

    # ----------------------------------------- build resource → edge adjacency
    block_supports = {b: np.argmax(P_mats[b], axis=0).astype(int) for b in P_mats}
    resource_to_edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b, S_b in block_supports.items():
        for j, r in enumerate(S_b):
            resource_to_edges[r].append((b, j))

    # ----------------------------------------- initialise edge messages
    block_out_mu  = {b: np.zeros(C_b.shape[1], dtype=dtype) for b, C_b in block_dicts.items()}
    block_out_var = {b: np.ones(C_b.shape[1],  dtype=np.float64) for b, C_b in block_dicts.items()}
    block_in_mu   = {b: np.zeros(C_b.shape[1], dtype=dtype) for b, C_b in block_dicts.items()}
    block_in_var  = {b: np.ones(C_b.shape[1],  dtype=np.float64) for b, C_b in block_dicts.items()}
    coeffs_hat    = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}
    coeffs_map    = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}

    M_total = float(sum(C_b.shape[0] for C_b in block_dicts.values()))
    lambda_est = float(lambda_init) if lambda_init is not None else 1.0 / M_total
    noise_var = (float(noise_var_init) if noise_var_init is not None
                 else float(np.real(np.vdot(Y.ravel(), Y.ravel()))) / (n * num_ant))

    converged = False
    it_used = 0
    history: list[dict] = []

    for it in range(1, max_iter + 1):
        it_used = it

        # --- Step A+B: Resource LMMSE + extrinsic messages --------------------
        for r in range(n):
            edges = resource_to_edges[r]
            if not edges:
                continue
            mu_in  = np.array([block_out_mu[b][j]  for b, j in edges], dtype=dtype)
            var_in = np.maximum([block_out_var[b][j] for b, j in edges], var_floor)
            v_sum  = float(np.sum(var_in))
            mu_s   = np.sum(mu_in)
            denom  = noise_var + gamma * v_sum
            innov  = np.dot(h.conj(), Y[r]) - gamma * mu_s
            hat_mu  = mu_in + (var_in * innov) / denom
            hat_var = var_in - (var_in ** 2) * gamma / denom
            denom_ext = denom - var_in * gamma
            tau_ext   = np.maximum(gamma / denom_ext, tau_floor)
            eta_ext   = hat_mu / hat_var - mu_in / var_in
            for idx, (b, j) in enumerate(edges):
                block_in_mu[b][j]  = eta_ext[idx] / tau_ext[idx]
                block_in_var[b][j] = 1.0 / tau_ext[idx]

        # --- Step C+D+E: Block discrete posterior + extrinsic messages --------
        delta = 0.0
        total_mean_count = 0.0
        total_x_var_post = 0.0
        for b, C_b in block_dicts.items():
            r_b = block_in_mu[b]
            v_b = np.maximum(block_in_var[b], var_floor)
            a_mean, x_mean, x_var, a_map = decode_block(C_b, r_b, v_b, lambda_est)
            coeffs_hat[b]     = a_mean
            coeffs_map[b]     = a_map
            total_mean_count += float(np.sum(a_mean))
            total_x_var_post += float(np.sum(x_var))

            tau_post = np.maximum(1.0 / x_var, tau_floor)
            tau_in   = np.maximum(1.0 / np.maximum(block_in_var[b], var_floor), tau_floor)
            tau_ext  = np.maximum(tau_post - tau_in, tau_floor)
            eta_ext  = x_mean * tau_post - r_b * tau_in
            tau_old  = np.maximum(1.0 / np.maximum(block_out_var[b], var_floor), tau_floor)
            eta_old  = block_out_mu[b] * tau_old
            tau_damp = (1.0 - damping) * tau_ext + damping * tau_old
            eta_damp = (1.0 - damping) * eta_ext + damping * eta_old
            mu_new   = eta_damp / tau_damp
            var_new  = np.maximum(1.0 / tau_damp, var_floor)
            delta = max(delta, float(np.max(np.abs(mu_new - block_out_mu[b]))))
            block_out_mu[b]  = mu_new
            block_out_var[b] = var_new

        # --- EM updates for λ and σ² ------------------------------------------
        lambda_est = max(total_mean_count / M_total, 1e-12)
        y_hat = np.zeros(n, dtype=dtype)
        for b, C_b in block_dicts.items():
            y_hat[block_supports[b]] += C_b.T @ coeffs_hat[b]
        Y_hat        = np.outer(y_hat, h)
        resid_mat    = Y - Y_hat
        resid_energy = float(np.real(np.vdot(resid_mat.ravel(), resid_mat.ravel())))
        noise_var    = max((resid_energy + gamma * total_x_var_post) / (n * num_ant), var_floor)

        history.append({"delta": delta, "lambda": lambda_est,
                         "noise_var": noise_var, "k_est": total_mean_count})
        print(
            f"  [iter {it:03d}] delta={delta:.3e}  k_est={total_mean_count:.2f}"
            f"  lambda={lambda_est:.3e}  sigma2={noise_var:.3e}",
            flush=True,
        )
        if delta < tol:
            converged = True
            break

    return coeffs_hat, coeffs_map, {
        "converged": converged,
        "iterations": it_used,
        "history": history,
        "tol": tol,
        "damping": damping,
        "lambda_est": lambda_est,
        "noise_var_est": noise_var,
        "lambda_init": lambda_init,
        "noise_var_init": noise_var_init,
        "poisson_tail_tol": poisson_tail_tol,
        "support_tail_tol": support_tail_tol,
    }


def assemble_global_counts(block_coeffs: dict[int, np.ndarray],
                            block_to_msg_list: dict[int, list[int]],
                            num_codewords: int) -> np.ndarray:
    """Convert blockwise coefficient vectors to a global message count vector."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, a_b in block_coeffs.items():
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_b[local_idx]
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper: nearest-neighbour codeword → message decision
# ═══════════════════════════════════════════════════════════════════════════════

def nn_votes_to_counts(x_hat_list: list[np.ndarray],
                       codebook: np.ndarray,
                       num_codewords: int) -> np.ndarray:
    """NN-match each estimated codeword against the codebook and tally votes.

    Each x_hat_list[k] is the codeword-domain estimate for one device slot.
    Returns a global message count vector of shape (num_codewords,).
    """
    counts = np.zeros(num_codewords, dtype=np.float64)
    for x_hat in x_hat_list:
        dists = np.sum(np.abs(x_hat[None, :] - codebook) ** 2, axis=1)
        counts[int(np.argmin(dists))] += 1.0
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER B — Approach 2: ignore patterns, joint LMMSE on x'_k = P_k x_k
# ═══════════════════════════════════════════════════════════════════════════════

def lmmse_approach2(
    Y: np.ndarray,
    active_msgs: np.ndarray,
    codebook: np.ndarray,
    P_mats: dict[int, np.ndarray],
    msg_to_block: dict[int, int],
    block_to_msg_list: dict[int, list[int]],
    noise_var: float,
    num_codewords: int,
) -> np.ndarray:
    """Approach 2: ignore ODMA pattern structure, estimate jointly.

    Model:  Y = X' H + Z,  X'(n,K) where x'_k = P_k x_k,  H(K,M) rows = h^T.

    LMMSE for right-hand-multiply model (per row y_i = H^T x'_i + z_i):
        X̂'^T = p_d H (p_d H^T H + σ² I_M)^{-1} Y^T
     ⟹ X̂'  = Y (p_d H^T H + σ² I_M)^{-1} (p_d H^T)

    Matrix to invert is M×M — cheap regardless of K.
    Then x̂_k = P_k^T x̂'_k.

    Oracle advantage: K and which block each device uses are known.
    p_d = 1/d (unit-norm codewords → per-symbol energy = 1/d).

    V2 degeneracy note: with h = 1_M for all K users, H is a (K, M) matrix
    with all rows identical to h^T.  H^T H = K · h h^T is a rank-1 (M×M)
    all-ones matrix (NOT K·I_M).  Consequently X_prime_hat has identical
    columns — every user receives exactly the same estimate — so all K NN
    searches return the same codeword.  This makes LMMSE-2 effectively a
    single-user estimator in V2: a useful lower-bound ablation, but not a
    competitive multi-user decoder.
    """
    K = len(active_msgs)
    n, M_ant = Y.shape
    dtype = Y.dtype
    d = codebook.shape[1]
    p_d = 1.0 / d
    h = np.ones(M_ant, dtype=dtype)             # V2: h = 1_M for all users
    H = np.tile(h[None, :], (K, 1))             # (K, M)

    # (p_d H^T H + σ² I_M) ∈ R^{M×M}
    HtH_M = p_d * (H.conj().T @ H)             # (M, M)  = p_d * K * h h^T in V2 (rank-1)
    reg_M  = noise_var * np.eye(M_ant, dtype=dtype)
    inv_M  = np.linalg.inv(HtH_M + reg_M)      # (M, M)
    # X̂'(n,K) = Y(n,M) inv_M(M,M) (p_d H^T)(M,K)
    X_prime_hat = p_d * (Y @ inv_M @ H.conj().T)   # (n, K)

    # Extract x̂_k = P_k^T x̂'_k for each active device k
    x_hat_list = []
    for k, m in enumerate(active_msgs):
        b = msg_to_block[int(m)]
        P_b = P_mats[b]                         # (n, d)
        x_hat_k = P_b.T @ X_prime_hat[:, k]    # (d,)
        x_hat_list.append(x_hat_k)

    return nn_votes_to_counts(x_hat_list, codebook, num_codewords)


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER C — Approach 3: TIN per user (pattern-aware interference covariance)
# ═══════════════════════════════════════════════════════════════════════════════

def lmmse_approach3(
    Y: np.ndarray,
    active_msgs: np.ndarray,
    codebook: np.ndarray,
    P_mats: dict[int, np.ndarray],
    msg_to_block: dict[int, int],
    noise_var: float,
    num_codewords: int,
) -> np.ndarray:
    """Approach 3: TIN (treat interference as noise) with exact interference covariance.

    For device k using block b_k with resource set S_k:
        Y'_k = P_k^T Y  ∈ R^{d×M}
        LMMSE (eq. 15 in notes):
            x̂_k = Y'_k · C_mat^{-1} · h_k^H
        where
            C_mat = (1/d) Σ_j ov_{jk} · h_j h_j^H + (σ²/p_d) I_M
            ov_{jk} = tr{P_j^T P_k P_k^T P_j} = |S_k ∩ S_j|  (pattern overlap)

    In V2 all h_j = h = 1_M, so C_mat = α (h h^H) + β I_M with:
        α = (1/d) Σ_j ov_{jk},   β = σ² / p_d = σ² d
    Sherman-Morrison inversion:
        (α h h^H + β I)^{-1} h = 1/(β + α M) · h

    Oracle advantage: K and which block each device uses are known.
    """
    K = len(active_msgs)
    n, M_ant = Y.shape
    dtype = Y.dtype
    d = codebook.shape[1]
    p_d = 1.0 / d
    h = np.ones(M_ant, dtype=dtype)

    device_blocks = [msg_to_block[int(m)] for m in active_msgs]
    # Resource sets: argmax extracts the single 1 in each column of P_b
    device_S = [np.argmax(P_mats[b], axis=0).astype(int) for b in device_blocks]

    x_hat_list = []
    for k in range(K):
        S_k  = device_S[k]
        P_k  = P_mats[device_blocks[k]]         # (n, d)
        Y_prime_k = P_k.T @ Y                   # (d, M)

        # Total overlap sum Σ_j |S_k ∩ S_j|  (j=k contributes |S_k|=d, the self-term)
        total_overlap = sum(
            len(np.intersect1d(S_k, device_S[j], assume_unique=True))
            for j in range(K)
        )

        # C_mat = α (h h^H) + β I_M
        alpha = total_overlap / d               # coefficient on h h^H
        beta  = noise_var / p_d                 # = noise_var * d

        # Sherman-Morrison: (α h h^H + β I)^{-1} h = 1/(β + α M) * h
        inv_h_coeff = 1.0 / (beta + alpha * M_ant)
        v = inv_h_coeff * h                     # (M,)

        # x̂_k = Y'_k · v  ∈  R^d
        x_hat_k = Y_prime_k @ v.conj()         # (d,)
        x_hat_list.append(x_hat_k)

    return nn_votes_to_counts(x_hat_list, codebook, num_codewords)


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER D — Approach 4: full joint vectorisation LMMSE
# ═══════════════════════════════════════════════════════════════════════════════

def lmmse_approach4(
    Y: np.ndarray,
    active_msgs: np.ndarray,
    codebook: np.ndarray,
    P_mats: dict[int, np.ndarray],
    msg_to_block: dict[int, int],
    noise_var: float,
    num_codewords: int,
) -> np.ndarray:
    """Approach 4: full joint vectorisation LMMSE (exact best linear estimator).

    Model:  y = vec(Y) = A x + z,
        A = [h^T ⊗ P_1, ..., h^T ⊗ P_K] ∈ R^{nM × Kd}
        x = [x_1^T, ..., x_K^T]^T ∈ R^{Kd}

    With x_k ~ N(0, p_d I_d) and z ~ N(0, σ² I_{nM}), information-form LMMSE:
        x̂ = (A^H A + (σ²/p_d) I_{Kd})^{-1} A^H y

    Matrix to invert: Kd × Kd.  At default args (K=10, d=16): 160×160 — fast.
    For large K this becomes expensive: O((Kd)^3).

    Kronecker construction:  (h^T ⊗ P_k)[r*M+m, j] = h[m] * P_k[r, j]

    Oracle advantage: K and which block each device uses are known.
    """
    K = len(active_msgs)
    n, M_ant = Y.shape
    dtype = Y.dtype
    d = codebook.shape[1]
    p_d = 1.0 / d
    h = np.ones(M_ant, dtype=dtype)

    device_blocks = [msg_to_block[int(m)] for m in active_msgs]

    # Build A ∈ R^{nM × Kd}
    A = np.zeros((n * M_ant, K * d), dtype=dtype)
    for k in range(K):
        P_k = P_mats[device_blocks[k]]          # (n, d)
        # (h^T ⊗ P_k): for each resource r and symbol j:
        #   A[r*M:r*M+M, j] = h * P_k[r, j]
        # Use broadcasting: (n, d, M) → transpose → reshape
        block_k = (P_k[:, :, None] * h[None, None, :])   # (n, d, M)
        A_k = block_k.transpose(0, 2, 1).reshape(n * M_ant, d)  # (nM, d)
        A[:, k * d:(k + 1) * d] = A_k

    # y = vec(Y) using same (r*M+m) row ordering as A
    y = Y.reshape(-1)                           # (nM,)

    # Information-form LMMSE: x̂ = (A^H A + reg I)^{-1} A^H y
    reg = (noise_var / p_d) * np.eye(K * d, dtype=dtype)
    AhA = A.conj().T @ A                        # (Kd, Kd)
    Ahy = A.conj().T @ y                        # (Kd,)
    x_hat_vec = np.linalg.solve(AhA + reg, Ahy)  # (Kd,)

    x_hat_list = [x_hat_vec[k * d:(k + 1) * d] for k in range(K)]
    return nn_votes_to_counts(x_hat_list, codebook, num_codewords)


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER E — SIC: successive interference cancellation
# ═══════════════════════════════════════════════════════════════════════════════

def sic_decoder(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    block_to_msg_list: dict[int, list[int]],
    num_codewords: int,
    max_detections: int = 100,
    energy_threshold_factor: float = 2.0,
) -> np.ndarray:
    """Successive interference cancellation over the ODMA resource grid.

    Algorithm (no oracle):
      1. Estimate noise floor from initial MF scores across all (block, codeword) pairs.
      2. MF-combine across antennas: ŷ = Y_residual h / M  ∈ R^n
      3. For every (b, m): compute MF energy |c_m^T P_b^T ŷ|²
      4. Declare the highest-energy (b, m) active; estimate the integer count
         from the MF projection c_m^T P_b^T ŷ ≈ true_count (handles multiplicities
         > 1 in one step rather than one subtraction per detection round).
         Subtract count copies from Y_residual and repeat.
      5. Stop when max energy falls below the noise-floor threshold.

    Noise floor estimation (no oracle):
        Each noise-only (b, m) MF score ~ noise_var/M_ant × chi²(1) (real)
        or ~ noise_var/M_ant × Exp(1) (complex). With a sparse active set
        (K ≪ B·L), the 25th percentile of all MF scores is noise-dominated.
        Correction factors: F⁻¹_chi²(1)(0.25) ≈ 0.1015 (real),
        F⁻¹_Exp(1)(0.25) = -ln(0.75) ≈ 0.2877 (complex).

    No oracle — does not know K, λ, user assignments, or noise variance.
    """
    n, M_ant = Y.shape
    dtype = Y.dtype
    h = np.ones(M_ant, dtype=dtype)
    h_norm_sq = float(np.real(np.dot(h.conj(), h)))    # = M
    d = next(iter(block_dicts.values())).shape[1]

    # ---- Data-driven noise floor estimate -----------------------------------
    y_mf0 = Y @ h.conj() / h_norm_sq                   # (n,) initial MF
    all_mf0 = np.array([
        float(s)
        for b, C_b in block_dicts.items()
        for s in np.abs(C_b @ (P_mats[b].T @ y_mf0)) ** 2
    ])
    q25_factor = 0.2877 if np.iscomplexobj(Y) else 0.1015
    noise_var_est = float(np.percentile(all_mf0, 25)) * M_ant / q25_factor
    # Threshold is fixed from the initial residual; it can become stale at high K
    # (signal-dominated percentile) or low SNR. Tune energy_threshold_factor if needed.
    threshold = energy_threshold_factor * noise_var_est * d / M_ant

    counts = np.zeros(num_codewords, dtype=np.float64)
    Y_residual = Y.copy()

    for _ in range(max_detections):
        y_mf = Y_residual @ h.conj() / h_norm_sq       # (n,)

        best_energy = -1.0
        best_b = -1
        best_local_idx = -1
        best_y_local: np.ndarray | None = None

        for b, C_b in block_dicts.items():
            P_b = P_mats[b]                             # (n, d)
            y_local = P_b.T @ y_mf                      # (d,)
            mf_scores = np.abs(C_b @ y_local) ** 2     # (L_b,)
            best_local_b = int(np.argmax(mf_scores))
            if mf_scores[best_local_b] > best_energy:
                best_energy = float(mf_scores[best_local_b])
                best_b = b
                best_local_idx = best_local_b
                best_y_local = y_local

        if best_energy < threshold:
            break

        # Estimate integer count from MF projection before subtracting.
        # E[c_best · P_b^T y_mf] ≈ true_count when the best signal dominates.
        # This handles multiplicities > 1 in a single subtraction step.
        c_best = block_dicts[best_b][best_local_idx]    # (d,)
        mf_proj = float(np.real(np.vdot(c_best, best_y_local)))
        count_est = max(1, round(mf_proj))

        global_msg = block_to_msg_list[best_b][best_local_idx]
        counts[global_msg] += count_est
        x_contribution = P_mats[best_b] @ c_best        # (n,)
        Y_residual -= count_est * np.outer(x_contribution, h)  # (n, M)

    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER F — AMP-BG: per-block AMP with Bernoulli-Gaussian prior
# ═══════════════════════════════════════════════════════════════════════════════

def amp_bg_per_block(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    block_to_msg_list: dict[int, list[int]],
    num_codewords: int,
    noise_var: float,
    num_devices_active: int,
    sigma_x_sq: float = 1.0,
    max_iter: int = 30,
) -> np.ndarray:
    """Per-block GAMP with Bernoulli-Gaussian (spike-and-slab) prior.

    Each block b is decoded independently from y_b = P_b^T y_mf, using
    iterative AMP with Onsager correction and BG-MMSE denoiser.

    AMP iteration (A_b = C_b^T ∈ R^{d×L_b}, y_b ∈ R^d, x_b ∈ R^{L_b}):
        r^t     = A_b^T z^t + x_hat^t           (approximate free message)
        τ^t     = ||z^t||² / d                   (empirical effective noise)
        x_hat^{t+1} = η_BG(r^t; τ^t, ρ, σ_x²)  (BG-MMSE denoiser)
        ξ^t     = (1/L_b) Σ η'(r^t; τ^t)        (Onsager weight via Stein)
        z^{t+1} = y_b - A_b x_hat^{t+1} + (L_b/d) ξ^t z^t

    Oracle: noise variance σ² and K (for activity rate ρ = K / M_total).
    """
    if np.iscomplexobj(Y) or np.iscomplexobj(next(iter(block_dicts.values()))):
        raise NotImplementedError("AMP-BG baseline currently supports real-valued setup only.")
    n, M_ant = Y.shape
    dtype = Y.dtype
    h = np.ones(M_ant, dtype=dtype)
    h_norm_sq = float(np.real(np.dot(h.conj(), h)))
    y_mf = np.real(Y @ h.conj() / h_norm_sq)          # (n,) real MF

    M_total = num_codewords
    rho = num_devices_active / M_total                  # activity rate per codeword
    sigma_eff_sq = noise_var / M_ant                    # per-dim noise after MF

    counts = np.zeros(num_codewords, dtype=np.float64)

    for b, C_b in block_dicts.items():
        y_b = P_mats[b].T @ y_mf                       # (d,) block observation
        A_b = np.real(C_b).T                            # (d, L_b) sensing matrix
        d_b, L_b = A_b.shape

        x_hat = np.zeros(L_b)
        z = y_b.copy()

        for _ in range(max_iter):
            # Approximate free message (AMP step without explicit 1/d normalization)
            r = A_b.T @ z + x_hat                      # (L_b,)
            tau = max(float(np.sum(z ** 2)) / d_b, sigma_eff_sq)

            # BG-MMSE denoiser in log domain for numerical stability
            # log p(x≠0|r) vs log p(x=0|r)
            log_p1 = (np.log(rho + 1e-300)
                      - 0.5 * np.log1p(sigma_x_sq / tau)
                      - r ** 2 / (2.0 * (tau + sigma_x_sq)))
            log_p0 = (np.log(1.0 - rho + 1e-300) - r ** 2 / (2.0 * tau))
            p_act = 1.0 / (1.0 + np.exp(np.clip(log_p0 - log_p1, -50.0, 50.0)))
            coeff = sigma_x_sq / (sigma_x_sq + tau)
            x_hat_new = p_act * coeff * r

            # Onsager weight via Stein's lemma: ξ = mean Var(x|r) / τ
            var_x_r = p_act * coeff * tau + p_act * (1.0 - p_act) * (coeff * r) ** 2
            xi = float(np.mean(var_x_r)) / tau

            z_new = y_b - A_b @ x_hat_new + (L_b / d_b) * xi * z
            delta = float(np.max(np.abs(x_hat_new - x_hat)))
            x_hat = x_hat_new
            z = z_new
            if delta < 1e-6:
                break

        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = max(0.0, round(float(x_hat[local_idx])))

    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER G — AMP-disc: per-block exact discrete Poisson posterior (no graph)
# ═══════════════════════════════════════════════════════════════════════════════

def _discrete_block_map(
    C_b: np.ndarray, y_b: np.ndarray, noise_var_per_dim: float, lam: float,
    poisson_tail_tol: float = 1e-4, support_tail_tol: float = 1e-4,
) -> np.ndarray:
    """MAP estimate for a single block under the discrete Poisson count prior.

    Enumerates all support configurations up to the tail tolerance and returns
    the maximum-a-posteriori count vector.  Same model as Graph-BP's decode_block
    but called once on the raw per-block projection (no iterative message passing).
    """
    L_b = C_b.shape[0]
    var_floor = 1e-10
    lam = max(float(lam), 1e-12)

    # Truncated Poisson PMF
    probs = [np.exp(-lam)]; total = probs[0]; c = 0
    while 1.0 - total > poisson_tail_tol:
        c += 1; probs.append(probs[-1] * lam / c); total += probs[-1]
        if probs[-1] == 0.0: break
    pmf = np.array(probs, dtype=np.float64) / np.sum(probs)
    c_max = len(pmf) - 1
    log_pmf = np.log(pmf + 1e-300)

    # Max support size via binomial tail
    p_nz = float(np.clip(1.0 - pmf[0], 1e-12, 1.0 - 1e-12))
    p0_b = 1.0 - p_nz; pk = p0_b ** L_b; cdf_b = pk; k_max = 0
    for k in range(L_b):
        pk = pk * ((L_b - k) / (k + 1)) * (p_nz / p0_b); cdf_b += pk
        if 1.0 - cdf_b <= support_tail_tol: k_max = k + 1; break
    else:
        k_max = L_b

    v_b = np.maximum(noise_var_per_dim * np.ones(C_b.shape[1], dtype=np.float64), var_floor)
    states: list[np.ndarray] = [np.zeros(L_b, dtype=np.float64)]
    log_prior: list[float] = [L_b * log_pmf[0]]
    for k in range(1, k_max + 1):
        lp_zeros = (L_b - k) * log_pmf[0]
        for idxs in combinations(range(L_b), k):
            for cnts in iproduct(range(1, c_max + 1), repeat=k):
                a = np.zeros(L_b, dtype=np.float64)
                a[list(idxs)] = np.array(cnts, dtype=np.float64)
                states.append(a)
                log_prior.append(lp_zeros + sum(log_pmf[c_] for c_ in cnts))

    A = np.array(states, dtype=np.float64)             # (num_states, L_b)
    X = A @ C_b                                         # (num_states, d)
    err = X - y_b[None, :]                             # (num_states, d)
    ll = -np.real(np.sum((np.abs(err) ** 2) / v_b[None, :], axis=1))
    log_post = ll + np.array(log_prior, dtype=np.float64)
    log_post -= log_post.max()
    return A[int(np.argmax(log_post))]


def amp_discrete_per_block(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    block_to_msg_list: dict[int, list[int]],
    num_codewords: int,
    noise_var: float,
    num_devices_active: int,
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
) -> np.ndarray:
    """Per-block decoder with exact discrete Poisson posterior (no graph coupling).

    Uses the same prior model and exact posterior enumeration as Graph-BP, but
    each block is decoded independently from its projected observation
    y_b = P_b^T y_mf without any cross-block message passing.

    This directly isolates the contribution of the factor-graph coupling:
    if Graph-BP beats this, the cross-block information exchange is key.

    Oracle: noise variance σ² and K (for Poisson rate λ = K / M_total).
    """
    n, M_ant = Y.shape
    dtype = Y.dtype
    h = np.ones(M_ant, dtype=dtype)
    h_norm_sq = float(np.real(np.dot(h.conj(), h)))
    y_mf = Y @ h.conj() / h_norm_sq

    M_total = num_codewords
    lam = num_devices_active / M_total                  # Poisson rate per codeword
    noise_var_per_dim = noise_var / M_ant               # per-dim noise after MF

    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, C_b in block_dicts.items():
        y_b = P_mats[b].T @ y_mf                       # (d,) block observation
        a_map = _discrete_block_map(
            C_b, y_b, noise_var_per_dim, lam,
            poisson_tail_tol, support_tail_tol)
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_map[local_idx]

    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER H — OMP-glob: global OMP on the full n × M_total dictionary
# ═══════════════════════════════════════════════════════════════════════════════

def global_omp(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    codebook: np.ndarray,
    block_to_msg_list: dict[int, list[int]],
    num_codewords: int,
    max_steps: int | None = None,
    bic_patience: int = 3,
) -> np.ndarray:
    """Global OMP on the full n × M_total dictionary Φ with BIC stopping.

    Builds Φ where column m = P_{b(m)} @ c_m  (the n-dimensional contribution
    of message m to y_mf), then runs Orthogonal Matching Pursuit on:
        y_mf = Φ a + noise_mf,   a sparse

    No ODMA structure is exploited — treats this as a flat global sparse
    recovery problem.

    Stopping is oracle-free and model-order principled:
    select support size k that minimises BIC(k) = n log(RSS_k/n) + k log n
    along the OMP path, with a short patience window for efficiency.
    """
    n, M_ant = Y.shape
    dtype = Y.dtype
    h = np.ones(M_ant, dtype=dtype)
    h_norm_sq = float(np.real(np.dot(h.conj(), h)))
    y_mf = Y @ h.conj() / h_norm_sq

    # Build global dictionary: column m = P_{b(m)} @ c_m ∈ R^n
    phi_dtype = np.complex128 if np.iscomplexobj(codebook) or np.iscomplexobj(Y) else np.float64
    Phi = np.zeros((n, num_codewords), dtype=phi_dtype)
    for b, msg_list in block_to_msg_list.items():
        for m in msg_list:
            Phi[:, m] = P_mats[b] @ codebook[m]
    # Columns are already unit-norm (P_b orthonormal, unit-norm codewords)

    residual = y_mf.copy()
    support: list[int] = []
    x_ls = np.zeros(0, dtype=phi_dtype)
    best_support: list[int] = []
    best_x = np.zeros(0, dtype=phi_dtype)
    best_bic = float("inf")
    worse_count = 0
    max_k = min(n, num_codewords) if max_steps is None else min(max_steps, n, num_codewords)

    for _ in range(max_k):
        corrs = np.abs(Phi.conj().T @ residual)
        if support:
            corrs[np.array(support, dtype=int)] = -np.inf
        best_m = int(np.argmax(corrs))
        if not np.isfinite(corrs[best_m]):
            break
        support.append(best_m)
        Phi_s = Phi[:, support]
        x_ls, _, _, _ = np.linalg.lstsq(Phi_s, y_mf, rcond=None)
        residual = y_mf - Phi_s @ x_ls
        rss = float(np.real(np.vdot(residual, residual)))
        rss = max(rss, 1e-12)
        k = len(support)
        bic = n * np.log(rss / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_support = support.copy()
            best_x = x_ls.copy()
            worse_count = 0
        else:
            worse_count += 1
            if worse_count >= bic_patience:
                break

    counts = np.zeros(num_codewords, dtype=np.float64)
    for i, m in enumerate(best_support):
        counts[m] = max(0.0, round(float(np.real(best_x[i]))))

    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_counts(counts_true: np.ndarray, counts_hard: np.ndarray) -> dict:
    """Compare true vs estimated global message count vectors.

    Metrics:
      tp/fp/fn  — support detection (counts_hard > 0 as estimated active set).
      f1        — harmonic mean of precision and recall.
      l1_err    — ||a_hat - a_true||_1 / ||a_true||_1.  Covers both FP over-counts
                  and FN under-counts; 0 = perfect, values > 1 indicate heavy FP.
      nmse      — ||a_hat - a_true||_2² / ||a_true||_2².  Quadratic penalization
                  of count errors across the full sparse vector.
    """
    supp_true = counts_true > 0
    supp_hard = counts_hard > 0
    tp = int(np.sum(supp_true & supp_hard))
    fp = int(np.sum(~supp_true & supp_hard))
    fn = int(np.sum(supp_true & ~supp_hard))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    norm1 = float(np.sum(np.abs(counts_true)))
    norm2sq = float(np.sum(counts_true ** 2))
    diff = counts_hard - counts_true
    l1_err = float(np.sum(np.abs(diff))) / max(norm1, 1e-12)
    nmse   = float(np.sum(diff ** 2)) / max(norm2sq, 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, f1=f1, l1_err=l1_err, nmse=nmse,
                support_true=int(np.sum(supp_true)))


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics printout
# ═══════════════════════════════════════════════════════════════════════════════

def print_setup_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list,
                             active_msgs, message_counts, block_coeffs,
                             Y_clean, Y_noisy, noise_var, esn0_db):
    """Print a full setup summary matching the style of the original V2 script."""
    num_blocks = len(P_mats)
    n, d = next(iter(P_mats.values())).shape
    M_ant = Y_noisy.shape[1]
    K = len(active_msgs)
    num_codewords = len(message_counts)

    print("\n" + "=" * 60)
    print("ODMA + URA Simulation — V2 comparison (multi-antenna, h=1_M)")
    print("=" * 60)
    print(f"\nCodebook          : {codebook.shape}  (num_codewords × d)")
    print(f"Resource grid     : n = {n}")
    print(f"Block size        : d = {d}")
    print(f"Num blocks        : {num_blocks}")
    print(f"Pattern matrices  : {num_blocks} × ({n}, {d})")
    print(f"Num antennas      : M = {M_ant}")
    print(f"Spatial signature : h = 1_M,  ||h||² = γ = {M_ant}")
    print(f"Active devices    : K = {K}")
    print(f"Es/N0             : {esn0_db:.1f} dB")
    print(f"Noise variance    : σ² = {noise_var:.6f},  σ²/M = {noise_var/M_ant:.6f}")
    print(f"\nActive messages (per user) : {active_msgs.tolist()}")
    print(f"Active blocks   (per user) : {[msg_to_block[int(m)] for m in active_msgs]}")

    active_indices = np.nonzero(message_counts)[0]
    print(f"\nDecoder target (global message counts):")
    print(f"  Unique messages    : {int(np.count_nonzero(message_counts))} / {num_codewords}")
    print(f"  Max multiplicity   : {int(message_counts.max())}")
    print(f"  Active msg → count : "
          f"{dict(zip(active_indices.tolist(), message_counts[active_indices].astype(int).tolist()))}")

    print(f"\nPer-block coefficient summary:")
    for b in range(num_blocks):
        a_b = block_coeffs[b]
        nnz = int(np.count_nonzero(a_b))
        total = int(a_b.sum())
        max_mult_b = int(a_b.max()) if nnz > 0 else 0
        print(f"  block {b:3d}: {len(block_to_msg_list[b]):4d} msgs, "
              f"nnz={nnz:3d}, total_users={total:3d}, max_mult={max_mult_b}")

    supports = {b: np.any(P_mats[b], axis=1) for b in range(num_blocks)}
    resource_usage = np.zeros(n, dtype=int)
    for b in range(num_blocks):
        resource_usage += supports[b].astype(int)
    overlap_counts = []
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            ov = int(np.sum(supports[i] & supports[j]))
            if ov > 0:
                overlap_counts.append((i, j, ov))

    print(f"\nBlock overlap statistics:")
    print(f"  Pairs with overlap : {len(overlap_counts)} / {num_blocks*(num_blocks-1)//2}")
    if overlap_counts:
        ovs = [c for _, _, c in overlap_counts]
        print(f"  Overlap sizes      : min={min(ovs)}, max={max(ovs)}, mean={np.mean(ovs):.1f}")
    print(f"  Max blocks/resource: {resource_usage.max()}")
    print(f"  Resources unused   : {int((resource_usage == 0).sum())} / {n}")

    print(f"\nReceived signal:")
    print(f"  Y shape        : {Y_noisy.shape}  (n × M)")
    print(f"  ||Y_clean||_F  : {np.linalg.norm(Y_clean):.4f}")
    print(f"  ||Y_noisy||_F  : {np.linalg.norm(Y_noisy):.4f}")
    print(f"  ||Z||_F        : {np.linalg.norm(Y_noisy - Y_clean):.4f}")
    print("=" * 60 + "\n")


def print_comparison_table(results: dict[str, dict], noise_var_true: float,
                            num_devices: int, num_codewords: int) -> None:
    """Print the final comparison table across all decoders."""
    print("\n" + "=" * 79)
    print("  DECODER COMPARISON")
    print("=" * 79)
    hdr = "{:<14s} {:>10s} {:>5s} {:>5s} {:>5s} {:>7s} {:>8s} {:>8s}"
    print(hdr.format("Decoder", "Oracle?", "TP", "FP", "FN", "F1", "L1err", "NMSE"))
    print("-" * 79)
    row_fmt = "{:<14s} {:>10s} {:>5d} {:>5d} {:>5d} {:>7.4f} {:>8.4f} {:>8.4f}"
    oracle_flags = {
        "Graph-BP": "none", "LMMSE-2": "K,assign", "LMMSE-3": "K,assign",
        "LMMSE-4": "K,assign", "SIC-LS": "none",
        "AMP-BG": "σ²,K", "AMP-disc": "σ²,K", "OMP-glob": "none",
    }
    for name, r in results.items():
        m = r["metrics"]
        print(row_fmt.format(name, oracle_flags.get(name, "?"),
                             m["tp"], m["fp"], m["fn"],
                             m["f1"], m["l1_err"], m["nmse"]))
    print("=" * 79)

    gm = results.get("Graph-BP", {}).get("meta", {})
    if gm:
        print(f"\nGraph-BP details:  converged={gm.get('converged','?')}  "
              f"iters={gm.get('iterations','?')}  "
              f"λ_est={gm.get('lambda_est',0):.4f}  "
              f"σ²_est={gm.get('noise_var_est',0):.5f}  "
              f"(σ²_true={noise_var_true:.5f})")
    print("=" * 83)
    print(f"\nOracle key: K,assign = oracle K + user→block assignments; "
          f"σ²,K = oracle noise + K.\n"
          f"Graph-BP, SIC-LS, OMP-glob use no oracle.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting  (7 figures, matching the original V2 output style)
# ═══════════════════════════════════════════════════════════════════════════════

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


def save_all_plots(results: dict, message_counts: np.ndarray,
                   noise_var_true: float, args,
                   P_mats: dict, Y_clean: np.ndarray, Y_noisy: np.ndarray,
                   out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names  = list(results.keys())
    colors = [PALETTE.get(n, "#333333") for n in names]
    n_res  = Y_clean.shape[0]
    M_ant  = Y_noisy.shape[1]
    num_blocks = len(P_mats)

    # ── Fig 1: F1 / L1err / NMSE ──────────────────────────────────────────────
    f1s     = [results[n]["metrics"]["f1"]     for n in names]
    l1errs  = [results[n]["metrics"]["l1_err"] for n in names]
    nmses   = [results[n]["metrics"]["nmse"]   for n in names]
    x = np.arange(len(names)); w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2 + 2), 4.5))
    ax.bar(x - w, f1s,    w, label="F1",     color="#4C78C8", alpha=0.85)
    ax.bar(x,     l1errs, w, label="L1err",  color="#E07B2A", alpha=0.85)
    ax.bar(x + w, nmses,  w, label="NMSE",   color="#C84C4C", alpha=0.85)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score  (lower L1err/NMSE = better,  higher F1 = better)")
    ax.set_title(
        f"Detection metrics  —  Es/N0={args.esn0_db:.0f} dB, "
        f"K={args.num_devices_active}, n={args.n}, d={args.d}, M={args.num_antennas}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_metrics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: TP / FP / FN ───────────────────────────────────────────────────
    tps = [results[n]["metrics"]["tp"] for n in names]
    fps = [results[n]["metrics"]["fp"] for n in names]
    fns = [results[n]["metrics"]["fn"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, tps, w, label="TP", color="#3BAA5C", alpha=0.85)
    ax.bar(x,     fps, w, label="FP", color="#E07B2A", alpha=0.85)
    ax.bar(x + w, fns, w, label="FN", color="#C84C4C", alpha=0.85)
    ax.axhline(results[names[0]]["metrics"]["support_true"], color="black",
               lw=1.2, ls="--",
               label=f"K_unique={results[names[0]]['metrics']['support_true']}")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Count"); ax.set_title("TP / FP / FN per decoder")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_tpfpfn.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Graph-BP convergence ────────────────────────────────────────────
    if "Graph-BP" in results:
        history = results["Graph-BP"].get("meta", {}).get("history", [])
        if history:
            iters = list(range(1, len(history) + 1))
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
            fig.suptitle("Graph-BP convergence — per-iteration EM diagnostics",
                         fontsize=11, y=1.01)

            ax = axes[0]
            ax.semilogy(iters, [h["delta"] for h in history],
                        color=PALETTE["Graph-BP"], lw=2, marker="o", ms=4)
            ax.axhline(results["Graph-BP"]["meta"]["tol"], color="red",
                       lw=1, ls="--",
                       label=f"tol={results['Graph-BP']['meta']['tol']:.0e}")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Max message Δ (log)")
            ax.set_title("Message convergence")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.plot(iters, [h["lambda"] for h in history],
                    color=PALETTE["Graph-BP"], lw=2, marker="o", ms=4, label="λ_est")
            ax.axhline(args.num_devices_active / args.num_codewords,
                       color="grey", lw=1.5, ls="--", label="λ_true = K/M")
            ax.set_xlabel("Iteration"); ax.set_ylabel("λ")
            ax.set_title("Poisson rate λ")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax = axes[2]
            ax.plot(iters, [h["noise_var"] for h in history],
                    color="#8B5CF6", lw=2, marker="o", ms=4, label="σ²_est")
            ax.axhline(noise_var_true, color="grey", lw=1.5, ls="--",
                       label=f"σ²_true={noise_var_true:.4f}")
            ax.set_xlabel("Iteration"); ax.set_ylabel("σ²")
            ax.set_title("Noise variance σ²")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(out_dir / "graph_bp_convergence.png",
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

    # ── Fig 4: Count estimates per active message ──────────────────────────────
    active_idx = np.nonzero(message_counts)[0]
    n_active   = len(active_idx)
    n_decoders = len(names)
    w2 = 0.75 / (n_decoders + 1)
    offsets = np.linspace(-(n_decoders / 2.0) * w2, (n_decoders / 2.0) * w2,
                          n_decoders + 1)

    fig, ax = plt.subplots(figsize=(max(9, n_active * 1.2 + 2), 4.5))
    ax.bar(np.arange(n_active) + offsets[0], message_counts[active_idx],
           w2, label="True", color="#888888", alpha=0.75)
    for i, (name, col) in enumerate(zip(names, colors)):
        ax.bar(np.arange(n_active) + offsets[i + 1],
               results[name]["counts"][active_idx],
               w2, label=name, color=col, alpha=0.85)
    ax.set_xticks(np.arange(n_active))
    ax.set_xticklabels([str(i) for i in active_idx], fontsize=8)
    ax.set_xlabel("Message index"); ax.set_ylabel("Count")
    ax.set_title("Estimated counts — active messages only")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_counts.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: ODMA pattern heatmap + resource usage histogram ─────────────────
    resource_usage = np.zeros(n_res, dtype=int)
    pattern_mat = np.zeros((num_blocks, n_res), dtype=np.float32)
    for b, P in P_mats.items():
        mask = np.any(P, axis=1)
        pattern_mat[b] = mask.astype(float)
        resource_usage += mask.astype(int)

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 5), layout="constrained",
        gridspec_kw={"height_ratios": [num_blocks, 1], "hspace": 0.08})
    axes[0].imshow(pattern_mat, aspect="auto", interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=1)
    axes[0].set_ylabel("Block"); axes[0].set_xlabel("")
    axes[0].set_title("ODMA pattern matrix  (blue = resource used by block)")
    axes[0].set_yticks(range(num_blocks))
    axes[1].bar(range(n_res), resource_usage, color="#4C78C8", alpha=0.7, width=1.0)
    axes[1].set_xlim(-0.5, n_res - 0.5)
    axes[1].set_ylabel("# blocks"); axes[1].set_xlabel("Resource index")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "odma_patterns.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 6: Received signal — clean vs noisy, MF average, noise trace ───────
    y_sc  = np.real(Y_clean[:, 0])              # clean scalar signal
    Y_mf  = np.real(Y_noisy).mean(axis=1)       # MF average ỹ = Y·h/M
    r_idx = np.arange(n_res)
    n_show = min(M_ant, 6)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, layout="constrained",
                             gridspec_kw={"hspace": 0.1})
    for m in range(n_show):
        axes[0].plot(r_idx, np.real(Y_noisy[:, m]),
                     color=PALETTE["LMMSE-2"], lw=0.4, alpha=0.2)
    axes[0].plot(r_idx, y_sc, color=PALETTE["Graph-BP"], lw=1.3,
                 label="y_scalar / clean (Re)")
    axes[0].plot(r_idx, Y_mf, color=PALETTE["LMMSE-2"],  lw=1.0,
                 label=f"ỹ = MF avg over {M_ant} ant (Re)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Received signal  (faint = individual antennas, bold = MF average)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    noise_mf = Y_mf - y_sc
    axes[1].plot(r_idx, noise_mf, color="#888888", lw=0.8, alpha=0.9,
                 label="MF noise residual (Re)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Resource index"); axes[1].set_ylabel("MF noise")
    bp_meta = results.get("Graph-BP", {}).get("meta", {})
    axes[1].set_title(
        f"MF noise  σ²={noise_var_true:.4f},  σ²/M={noise_var_true/M_ant:.4f}  "
        f"(effective),  σ²_est={bp_meta.get('noise_var_est', float('nan')):.4f}")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.savefig(out_dir / "received_signal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 7: Per-block total active count (true vs all decoders) ─────────────
    block_ids = sorted(P_mats.keys())
    num_cw = len(message_counts)

    def per_block_totals(counts_vec):
        return [float(np.sum(counts_vec[
            [m for m in range(num_cw) if m % num_blocks == b]
        ])) for b in block_ids]

    true_pb = per_block_totals(message_counts)
    w3 = 0.75 / (n_decoders + 1)
    offsets3 = np.linspace(-(n_decoders / 2.0) * w3, (n_decoders / 2.0) * w3,
                            n_decoders + 1)

    fig, ax = plt.subplots(figsize=(max(7, num_blocks * 0.9 + 2), 4))
    ax.bar(np.arange(num_blocks) + offsets3[0], true_pb, w3,
           label="True", color="#888888", alpha=0.75)
    for i, (name, col) in enumerate(zip(names, colors)):
        ax.bar(np.arange(num_blocks) + offsets3[i + 1],
               per_block_totals(results[name]["counts"]),
               w3, label=name, color=col, alpha=0.85)
    ax.set_xticks(np.arange(num_blocks))
    ax.set_xticklabels([f"B{b}" for b in block_ids], fontsize=9)
    ax.set_xlabel("Block"); ax.set_ylabel("Total count")
    ax.set_title("Per-block total active count (true vs estimated)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "per_block_counts.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved → {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown + JSON results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results_markdown(results: dict, args, noise_var_true: float,
                          message_counts: np.ndarray, out_dir: Path) -> None:
    names = list(results.keys())
    oracle_flags = {
        "Graph-BP": "none", "LMMSE-2": "K,assign", "LMMSE-3": "K,assign",
        "LMMSE-4": "K,assign", "SIC-LS": "none",
        "AMP-BG": "σ²,K", "AMP-disc": "σ²,K", "OMP-glob": "none",
    }
    rows = "\n    ".join(
        f"| {n} | {oracle_flags.get(n, '?')} | "
        f"{results[n]['metrics']['tp']} | {results[n]['metrics']['fp']} | "
        f"{results[n]['metrics']['fn']} | "
        f"{results[n]['metrics']['f1']:.4f} | "
        f"{results[n]['metrics']['l1_err']:.4f} | "
        f"{results[n]['metrics']['nmse']:.4f} |"
        for n in names
    )

    gm = results.get("Graph-BP", {}).get("meta", {})
    active_true = {int(m): int(message_counts[m]) for m in np.nonzero(message_counts)[0]}

    md = textwrap.dedent(f"""
    # ODMA + URA Decoder Comparison
    **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## Setup
    | Parameter | Value |
    |-----------|-------|
    | Resource grid n | {args.n} |
    | Codeword length d | {args.d} |
    | Num blocks B | {args.num_blocks} |
    | Num codewords M | {args.num_codewords} |
    | Active devices K | {args.num_devices_active} |
    | Antennas | {args.num_antennas} |
    | Es/N0 | {args.esn0_db:.1f} dB |
    | σ²_true | {noise_var_true:.6f} |
    | Complex-valued | {args.complex_valued} |
    | Seed | {args.seed} |
    | Graph-BP max iters | {args.max_iter} |
    | Graph-BP damping | {args.damping} |

    ## Detection Results

    | Decoder | Oracle | TP | FP | FN | F1 | L1err | NMSE |
    |---------|--------|----|----|----|----|-------|------|
    {rows}

    **Decoder notes:**
    - **Graph-BP**: message passing + discrete Poisson posterior + EM for λ and σ². No oracle.
      Converged={gm.get('converged', '?')}, iters={gm.get('iterations', '?')},
      λ_est={gm.get('lambda_est', 0):.4f}, σ²_est={gm.get('noise_var_est', 0):.5f}
    - **LMMSE-2**: ignores ODMA pattern structure. Oracle: K, user→block. Degenerate with h=1_M.
    - **LMMSE-3**: TIN per user with overlap-aware covariance. Oracle: K, user→block.
    - **LMMSE-4**: full joint vectorisation LMMSE (exact best linear estimator). Oracle: K, user→block.
    - **SIC-LS**: greedy SIC with MF-projected count estimation. No oracle.
    - **AMP-BG**: per-block GAMP with Bernoulli-Gaussian prior. Oracle: σ², K.
    - **AMP-disc**: per-block exact discrete Poisson posterior (no graph coupling). Oracle: σ², K.
    - **OMP-glob**: global OMP on full n×M dictionary with BIC model-order selection. No oracle.

    ## Ground Truth Counts
    ```
    {active_true}
    ```

    ## Plots
    - `comparison_metrics.png`    — F1 / L1err / NMSE per decoder
    - `comparison_tpfpfn.png`     — TP / FP / FN per decoder
    - `comparison_counts.png`     — estimated counts per active message
    - `graph_bp_convergence.png`  — Graph-BP per-iteration delta, λ, σ²
    - `odma_patterns.png`         — block×resource heatmap + usage histogram
    - `received_signal.png`       — clean vs noisy signal and MF noise trace
    - `per_block_counts.png`      — per-block total count (true vs all decoders)
    """).strip()

    (out_dir / "results.md").write_text(md)
    (out_dir / "raw.json").write_text(json.dumps(
        {n: {"metrics": results[n]["metrics"]} for n in names}, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def make_slug(args) -> str:
    cx = "cx" if args.complex_valued else "re"
    return (
        f"compare_{cx}_n{args.n}_d{args.d}_B{args.num_blocks}"
        f"_M{args.num_codewords}_K{args.num_devices_active}"
        f"_ant{args.num_antennas}_snr{args.esn0_db:.0f}dB_s{args.seed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="ODMA+URA decoder comparison: Graph-BP, LMMSE, SIC-LS, AMP baselines, OMP-glob")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=128, help="total resource grid size")
    parser.add_argument("--d", type=int, default=16, help="codeword length / block size")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-codewords", type=int, default=64)
    parser.add_argument("--num-devices-active", type=int, default=30)
    parser.add_argument("--esn0-db", type=float, default=0.0, help="Es/N0 in dB")
    parser.add_argument("--num-antennas", type=int, default=4, help="number of receive antennas M")
    parser.add_argument("--complex-valued", action="store_true")
    parser.add_argument("--max-iter", type=int, default=20, help="max iterations for Graph-BP")
    parser.add_argument("--damping", type=float, default=0.3, help="message damping for Graph-BP")
    parser.add_argument("--lambda-init", type=float, default=None, help="initial Poisson mean per message (Graph-BP)")
    parser.add_argument("--noise-var-init", type=float, default=None, help="initial noise variance (Graph-BP)")
    parser.add_argument("--poisson-tail-tol", type=float, default=1e-4)
    parser.add_argument("--support-tail-tol", type=float, default=1e-4)
    parser.add_argument("--sic-energy-factor", type=float, default=2.0, help="SIC stopping threshold multiplier")
    parser.add_argument("--sic-max-detections", type=int, default=100, help="SIC max detections per run")
    parser.add_argument("--results-dir", type=str, default="results_compare")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Data generation (shared across ALL decoders) ──────────────────────────
    print("Generating data...")
    codebook = make_codebook(args.num_codewords, args.d, rng, args.complex_valued)
    blocks   = make_odma_blocks(args.num_blocks, args.n, args.d, rng)
    P_mats   = build_pattern_matrices(blocks, args.n)
    msg_to_block, block_to_msg_list = make_message_block_mapping(
        args.num_codewords, args.num_blocks)
    block_dicts = build_block_dictionaries(codebook, block_to_msg_list, args.num_blocks)

    active_msgs    = sample_active_messages(args.num_devices_active, args.num_codewords, rng)
    message_counts = build_message_counts(active_msgs, args.num_codewords)
    block_coeffs   = build_block_coefficients(
        active_msgs, msg_to_block, block_to_msg_list, args.num_blocks)
    noise_var = esn0_db_to_noise_var(args.esn0_db, args.d)

    Y_noisy, Y_clean = synthesize_received_signal(
        P_mats, block_dicts, block_coeffs,
        args.num_antennas, noise_var, rng, args.complex_valued)

    print_setup_diagnostics(
        codebook, P_mats, msg_to_block, block_to_msg_list,
        active_msgs, message_counts, block_coeffs,
        Y_clean, Y_noisy, noise_var, args.esn0_db)

    results: dict[str, dict] = {}

    # ── A: Graph-BP ────────────────────────────────────────────────────────────
    print("=" * 56)
    print("Decoder A — Graph-BP  (no oracle)")
    print("=" * 56)
    coeffs_hat, coeffs_map, meta_bp = graph_based_decoder(
        Y_noisy, P_mats, block_dicts,
        max_iter=args.max_iter,
        damping=args.damping,
        lambda_init=args.lambda_init,
        noise_var_init=args.noise_var_init,
        poisson_tail_tol=args.poisson_tail_tol,
        support_tail_tol=args.support_tail_tol,
    )
    counts_bp = assemble_global_counts(coeffs_map, block_to_msg_list, args.num_codewords)
    results["Graph-BP"] = {
        "counts":  counts_bp,
        "metrics": evaluate_counts(message_counts, counts_bp),
        "meta":    meta_bp,
    }
    print(f"\nGraph-BP done — converged={meta_bp['converged']}, "
          f"iters={meta_bp['iterations']}\n")

    # ── B: LMMSE-2 ─────────────────────────────────────────────────────────────
    print("Decoder B — LMMSE-2  (oracle: K + user→block assignments)")
    counts_l2 = lmmse_approach2(
        Y_noisy, active_msgs, codebook, P_mats, msg_to_block,
        block_to_msg_list, noise_var, args.num_codewords)
    results["LMMSE-2"] = {
        "counts":  counts_l2,
        "metrics": evaluate_counts(message_counts, counts_l2),
    }
    print("  done\n")

    # ── C: LMMSE-3 ─────────────────────────────────────────────────────────────
    print("Decoder C — LMMSE-3  (TIN, oracle: K + user→block assignments)")
    counts_l3 = lmmse_approach3(
        Y_noisy, active_msgs, codebook, P_mats, msg_to_block,
        noise_var, args.num_codewords)
    results["LMMSE-3"] = {
        "counts":  counts_l3,
        "metrics": evaluate_counts(message_counts, counts_l3),
    }
    print("  done\n")

    # ── D: LMMSE-4 ─────────────────────────────────────────────────────────────
    print("Decoder D — LMMSE-4  (joint vec, oracle: K + user→block assignments)")
    counts_l4 = lmmse_approach4(
        Y_noisy, active_msgs, codebook, P_mats, msg_to_block,
        noise_var, args.num_codewords)
    results["LMMSE-4"] = {
        "counts":  counts_l4,
        "metrics": evaluate_counts(message_counts, counts_l4),
    }
    print("  done\n")

    # ── E: SIC-LS ──────────────────────────────────────────────────────────────
    print(f"Decoder E — SIC-LS  (no oracle, energy_factor={args.sic_energy_factor})")
    counts_sic = sic_decoder(
        Y_noisy, P_mats, block_dicts, block_to_msg_list, args.num_codewords,
        max_detections=args.sic_max_detections,
        energy_threshold_factor=args.sic_energy_factor,
    )
    results["SIC-LS"] = {
        "counts":  counts_sic,
        "metrics": evaluate_counts(message_counts, counts_sic),
    }
    print(f"  done  ({int(np.sum(counts_sic > 0))} messages detected)\n")

    # ── F: AMP-BG ──────────────────────────────────────────────────────────────
    print("Decoder F — AMP-BG  (oracle: σ², K for ρ)")
    counts_amp_bg = amp_bg_per_block(
        Y_noisy, P_mats, block_dicts, block_to_msg_list,
        args.num_codewords, noise_var, args.num_devices_active)
    results["AMP-BG"] = {
        "counts":  counts_amp_bg,
        "metrics": evaluate_counts(message_counts, counts_amp_bg),
    }
    print("  done\n")

    # ── G: AMP-disc ────────────────────────────────────────────────────────────
    print("Decoder G — AMP-disc  (oracle: σ², K for λ)")
    counts_amp_disc = amp_discrete_per_block(
        Y_noisy, P_mats, block_dicts, block_to_msg_list,
        args.num_codewords, noise_var, args.num_devices_active,
        poisson_tail_tol=args.poisson_tail_tol,
        support_tail_tol=args.support_tail_tol)
    results["AMP-disc"] = {
        "counts":  counts_amp_disc,
        "metrics": evaluate_counts(message_counts, counts_amp_disc),
    }
    print("  done\n")

    # ── H: OMP-glob ────────────────────────────────────────────────────────────
    print("Decoder H — OMP-glob  (no oracle, BIC stopping)")
    counts_omp = global_omp(
        Y_noisy, P_mats, codebook, block_to_msg_list,
        args.num_codewords)
    results["OMP-glob"] = {
        "counts":  counts_omp,
        "metrics": evaluate_counts(message_counts, counts_omp),
    }
    print("  done\n")

    # ── Print comparison table ─────────────────────────────────────────────────
    print_comparison_table(results, noise_var, args.num_devices_active, args.num_codewords)

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir) / make_slug(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_all_plots(results, message_counts, noise_var, args,
                   P_mats, Y_clean, Y_noisy, out_dir)
    save_results_markdown(results, args, noise_var, message_counts, out_dir)
    print(f"\nAll results saved → {out_dir}/")

    return results, message_counts


if __name__ == "__main__":
    main()