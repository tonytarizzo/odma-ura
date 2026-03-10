"""
ODMA + URA Decoder Testbed — V3: Multi-antenna with unknown Rayleigh fading
============================================================================
Full Rayleigh fading: each active device has an unknown channel vector
h_u ~ CN(0, I_M) which is latent at the decoder.

Signal model:
    Y = sum_u (P_{b_u} c_{m_u}) h_u^T+ Z       ∈ C^{n × M}
      = sum_b P_b C_b^T U_b a_b + Z

  where U_b ∈ C^{L_b × M}:
      U_b[i, :] = sum of h_u over all users u that chose local message i in block b.

  Since h_u ~ CN(0, I_M):
      U_b[i, :] | a_{b,i} = c  ~  CN(0, c · I_M)

  The decoder treats h_u as fully latent with Gaussian prior h_u ~ CN(0, gamma_h I_M).
  gamma_h is estimated by EM.

Decoder algorithm:
  For each active user/symbol node k, maintain channel belief q(h_k) = CN(m_k, R_k).

  Iterate:
    A.  Resource node x-update (mean-field Gaussian, using expected channel Gram G_t):
          G_t[i,i]  = tr(R_i) + |m_i|^2
          G_t[i,j]  = m_i^H m_j              (i ≠ j)
          Σ̂_t       = ((V_t^in)^{-1} + σ^{-2} G_t)^{-1}
          μ̂_t       = Σ̂_t ((V_t^in)^{-1} μ_t^in + σ^{-2} H̄_t^H y_t)

    B.  Extrinsic x messages (Gaussian division, same as V2):
          τ_{t→k} = τ̂_{k,t} - τ_{k→t},  η_{t→k} = η̂_{k,t} - η_{k→t}

    C.  Channel node update (pools over all resources S(k)):
          For each active symbol-node k:
            ν_{k,t}     = |μ_{k,t}|² + [Σ̂_t]_{kk}   (E[|x_{k,t}|²])
            r_{k,t}     = y_t - Σ_{j≠k} m_j μ_{j,t}  (residual excl. k)
            Λ_k^post    = (γ_h^{-1} I + σ^{-2} Σ_{t∈S(k)} ν_{k,t} I)^{-1}
            m_k^post    = Λ_k^post (γ_h^{-1}·0 + σ^{-2} Σ_{t∈S(k)} μ_{k,t}^* r_{k,t})

    D.  Block node: augmented (a_b, U_b) posterior.
          For a candidate count state a_b with support S:
            Conditioned on a_b, integrate out U_b:
              p(Y_b | a_b) = ∏_{t∈S_b} CN(y_t^T C_{b,S}; 0, [a_b_S C_{b,S} C_{b,S}^H + σ² I_d] ⊗ I_M)
            Approximation: collapse per-resource evidence using the block pseudo-observation
            r_b (from extrinsic x-messages) plus the full-Y block contribution.

            Block likelihood (integrated over U_b):
              For state a_b with support S:
                ℓ(a_b) = -Σ_{m=0}^{M-1} [r_b^{(m)} - C_b^T a_b·δ_{m,0}]^H Ψ_b^{-1} [...]
              where Ψ_b encodes channel uncertainty.

            Practically, we use a two-term block log-likelihood:
              log p(R_b | a_b) ≈
                -(r_b[:,0] - C_b^T a_b)^H diag(v_b)^{-1} (r_b[:,0] - C_b^T a_b)  [count term]
                - Σ_{m=1}^{M-1} r_b[:,m]^H (C_{b,S} Λ_S C_{b,S}^H + σ² I)^{-1} r_b[:,m]  [channel term]

            Here the channel term marginalizes U_b[:,m] ~ CN(0, Λ_S) with
            Λ_S = diag(a_{b,S} * gamma_h_est).

    E.  EM updates: λ, σ², γ_h (channel power prior).

Decoder target:
    message_counts ∈ Z_+^{num_codewords}

Run:
    python graph_based_decoder_v3.py --seed 42 --n 128 --d 16 --num-blocks 8 \
        --num-codewords 64 --num-devices-active 10 --num-antennas 4 --esn0-db 10
"""

from __future__ import annotations
import argparse
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from itertools import combinations, product as iproduct
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utility functions (identical to V2 where applicable)
# ═══════════════════════════════════════════════════════════════════════════════

def make_codebook(num_codewords: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian codebook with unit-norm rows. Returns (num_codewords, d)."""
    raw = rng.standard_normal((num_codewords, d)) + 1j * rng.standard_normal((num_codewords, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_odma_blocks(num_blocks: int, n: int, d: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random ODMA blocks. Each block: d resource indices drawn without replacement from {0,...,n-1}."""
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(num_blocks)]


def build_pattern_matrices(blocks: list[np.ndarray], n: int) -> dict[int, np.ndarray]:
    """Build per-block embedding matrices P_b ∈ {0,1}^{n×d}. P_b^T P_b = I_d."""
    P_mats: dict[int, np.ndarray] = {}
    for b, S_b in enumerate(blocks):
        d = len(S_b)
        P = np.zeros((n, d), dtype=np.float64)
        P[S_b, np.arange(d)] = 1.0
        P_mats[b] = P
    return P_mats


def make_message_block_mapping(num_codewords: int, num_blocks: int):
    """Deterministic mapping: message m → block m % num_blocks."""
    msg_to_block: dict[int, int] = {}
    block_to_msg_list: dict[int, list[int]] = {b: [] for b in range(num_blocks)}
    for m in range(num_codewords):
        b = m % num_blocks
        msg_to_block[m] = b
        block_to_msg_list[b].append(m)
    return msg_to_block, block_to_msg_list


def sample_active_messages(num_devices_active: int, num_codewords: int, rng: np.random.Generator) -> np.ndarray:
    """Each active device picks a message index uniformly at random. Duplicates allowed."""
    return rng.integers(0, num_codewords, size=num_devices_active)


def build_message_counts(active_msgs: np.ndarray, num_codewords: int) -> np.ndarray:
    """Global message count vector — the decoder target. Returns (num_codewords,)."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for m in active_msgs:
        counts[int(m)] += 1.0
    return counts


def build_block_coefficients(
    active_msgs: np.ndarray,
    msg_to_block: dict[int, int],
    block_to_msg_list: dict[int, list[int]],
    num_blocks: int,
) -> dict[int, np.ndarray]:
    """Blockwise decoder target: sparse count vectors a_b."""
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


def build_block_dictionaries(
    codebook: np.ndarray,
    block_to_msg_list: dict[int, list[int]],
    num_blocks: int,
) -> dict[int, np.ndarray]:
    """Per-block dictionaries: block_idx -> (L_b, d) complex codeword matrix."""
    return {b: codebook[block_to_msg_list[b]] for b in range(num_blocks)}


def esn0_db_to_noise_var(esn0_db: float, d: int) -> float:
    """Es/N0 (dB) → per-entry per-antenna noise variance. Es = 1/d."""
    esn0_lin = 10.0 ** (esn0_db / 10.0)
    return 1.0 / (d * esn0_lin)


# ═══════════════════════════════════════════════════════════════════════════════
# V3-specific: channel generation and signal synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def make_user_channels(num_users: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. Rayleigh fading channels h_u ~ CN(0, I_M). Returns (num_users, M)."""
    return (rng.standard_normal((num_users, M)) + 1j * rng.standard_normal((num_users, M))) / np.sqrt(2)


def synthesize_received_signal_v3(
    active_msgs: np.ndarray,
    msg_to_block: dict[int, int],
    P_mats: dict[int, np.ndarray],
    codebook: np.ndarray,
    channels: np.ndarray,        # (K, M) — true channels, unknown to decoder
    noise_var: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize Y = Σ_u (P_{b_u} c_{m_u}) h_u^T + Z.

    Each user contributes a rank-1 outer product. Channels are i.i.d. CN(0, I_M).

    Returns: (Y_noisy, Y_clean)  both (n, M) complex
    """
    n = next(iter(P_mats.values())).shape[0]
    M = channels.shape[1]
    Y_clean = np.zeros((n, M), dtype=np.complex128)

    for u, m_u in enumerate(active_msgs):
        m_u = int(m_u)
        b_u = msg_to_block[m_u]
        P_u = P_mats[b_u]             # (n, d)
        c_u = codebook[m_u]           # (d,)
        h_u = channels[u]             # (M,)
        s_u = P_u @ c_u               # (n,)
        Y_clean += np.outer(s_u, h_u)

    noise = (np.sqrt(noise_var / 2) *
             (rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M))))
    return Y_clean + noise, Y_clean


# ═══════════════════════════════════════════════════════════════════════════════
# V3 graph-based decoder
# ═══════════════════════════════════════════════════════════════════════════════

def graph_based_decoder_v3(
    Y: np.ndarray,                          # (n, M) complex received signal
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],     # block_idx -> (L_b, d)
    *,
    max_iter: int = 50,
    damping: float = 0.3,
    tol: float = 1e-4,
    lambda_init: float | None = None,
    noise_var_init: float | None = None,
    gamma_h_init: float | None = None,      # initial channel power prior (scalar, isotropic)
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
    channel_update_interval: int = 1,       # how often to run channel updates (every N iters)
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Iterative message-passing decoder for V3 (unknown Rayleigh fading).

    Graph structure:
      - Resource nodes r: matrix obs y_r ∈ C^M, unknown h_k at each edge
      - Symbol/variable nodes k: scalar complex signals x_{k,r}
      - Channel nodes k: vector channel h_k ∈ C^M shared across all resources in S(k)
      - Block nodes b: discrete sparse posterior over (a_b, U_b)

    Message schedule:
      1.  Block → resource: Gaussian x-messages (init uninformative)
      2.  Resource x-update: mean-field Gaussian using current channel beliefs G_t
      3.  Extrinsic x-messages (Gaussian division)
      4.  Channel update: Gaussian update for h_k using current x-beliefs
      5.  Block decode: augmented (a_b, U_b) posterior, integrate out U_b
      6.  Extrinsic block → resource x-messages
      7.  EM updates: λ, σ², γ_h

    Active symbol nodes are implicit: each edge (b,j) represents one (user-message, resource)
    pair; the channel belief lives at the level of (block, local-msg-index).
    """
    n, M = Y.shape
    var_floor = 1e-10
    tau_floor = 1e-10
    cov_floor = 1e-10 * np.eye(M, dtype=np.complex128)

    # ────────────────────────────── helpers ──────────────────────────────────

    def logsumexp(v: np.ndarray) -> float:
        vmax = float(np.max(v))
        return vmax + float(np.log(np.sum(np.exp(v - vmax)))) if np.isfinite(vmax) else vmax

    def poisson_pmf_vec(lam: float) -> np.ndarray:
        """Truncated Poisson PMF p(0),...,p(c_max) normalised to sum 1."""
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
        """Largest support size k with tail mass P(Bin(L_b, p_nz) > k) > support_tail_tol."""
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

    # ────────────────────────────── graph structure ───────────────────────────

    # block_supports[b] : (d,) array of global resource indices for block b
    block_supports = {b: np.argmax(P_mats[b], axis=0).astype(int) for b in P_mats}

    # resource_to_edges[r] = list of (b, j): block b, local position j → resource r
    resource_to_edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b, S_b in block_supports.items():
        for j, r in enumerate(S_b):
            resource_to_edges[r].append((b, j))

    # ────────────────────────────── EM parameter init ────────────────────────

    total_L = float(sum(C_b.shape[0] for C_b in block_dicts.values()))

    lambda_est  = float(lambda_init)    if lambda_init    is not None else 1.0 / total_L
    noise_var   = float(noise_var_init) if noise_var_init is not None else (
        float(np.real(np.vdot(Y.ravel(), Y.ravel()))) / (n * M))
    # Channel power: E[||h_u||²] = M for CN(0,I_M), so γ_h = 1 per component.
    gamma_h     = float(gamma_h_init)   if gamma_h_init   is not None else 1.0

    # ────────────────────────────── message initialisation ───────────────────

    # x-messages: block → resource  (block_out) and resource → block (block_in)
    # Indexed by (b, j): scalar complex mean, scalar real variance
    block_out_mu  = {b: np.zeros(C_b.shape[1], dtype=np.complex128) for b, C_b in block_dicts.items()}
    block_out_var = {b: np.ones(C_b.shape[1],  dtype=np.float64)    for b, C_b in block_dicts.items()}
    block_in_mu   = {b: np.zeros(C_b.shape[1], dtype=np.complex128) for b, C_b in block_dicts.items()}
    block_in_var  = {b: np.ones(C_b.shape[1],  dtype=np.float64)    for b, C_b in block_dicts.items()}

    # Channel beliefs: for each (b, j) edge, h_{b,j} ∈ C^M
    # Initialise to prior: CN(0, γ_h I_M)
    ch_mu  = {b: np.zeros((C_b.shape[1], M), dtype=np.complex128) for b, C_b in block_dicts.items()}
    ch_cov = {b: [gamma_h * np.eye(M, dtype=np.complex128) for _ in range(C_b.shape[1])]
              for b, C_b in block_dicts.items()}

    # Posterior x-moments (used for channel update): keyed by (b, j)
    # mu_post[b][j] = E[x_{b,j,r}] at each resource r (we store dict per resource)
    # We store last resource-level posterior means/vars for channel update
    # These are updated in the resource x-update step.
    # Shape: post_mu_res[r] = array (num_edges_at_r,) of posterior means
    post_mu_res  = [None] * n
    post_var_res = [None] * n

    # Final block coefficient estimates
    coeffs_hat = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}
    coeffs_map  = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}

    converged = False
    it_used = 0
    history: list[dict] = []

    # ════════════════════════════════════════════════════════════════════════
    # Block decode function: augmented (a_b, U_b) posterior, U_b integrated out
    # ════════════════════════════════════════════════════════════════════════

    def decode_block_v3(
        C_b: np.ndarray,       # (L_b, d)
        r_b: np.ndarray,       # (d, M) — matrix pseudo-observation (all antennas)
        v_b: np.ndarray,       # (d,)   — scalar variance of the x-messages (antenna-0 count channel)
        lam: float,
        gamma_h_est: float,
        sigma2: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact enumeration over count states a_b; U_b integrated out analytically.

        Signal model at block level:
            r_b[:, 0] = C_b^T a_b + w_b       (count channel, w_b ~ CN(0, diag(v_b)))
            r_b[:, m] = C_b^T u_b^(m) + w_b^(m)   (fading channels, m=1..M-1)
              where u_b^(m)[i] | a_{b,i}=c ~ CN(0, c·γ_h)  i.i.d.

        For the channel-term, marginalise u_b^(m) for each m:
            r_b[:, m] | a_b ~ CN(0, C_b^T diag(a_b·γ_h) C_b^* + diag(v_b))

        Log-likelihood for candidate a_b:
            ll_count  = -(r_b[:,0] - C_b^T a_b)^H diag(v_b)^{-1} (r_b[:,0] - C_b^T a_b)
            ll_fading = Σ_{m=1}^{M-1} log CN(r_b[:,m]; 0, Σ_{fad}(a_b))
              where Σ_{fad}(a_b) = γ_h · C_b^T diag(a_b) C_b^* + diag(v_b)

        For the count term we keep using the extrinsic scalar messages (diagonal V_b).
        For the fading term, we use the full per-column of Y projected onto the block
        (stored in r_b[:, 1:]) with the correct integrated-out covariance.

        Returns:
            a_mean : (L_b,)  E[a_b | r_b]
            x_mean : (d,)    E[x_b[:,0] | r_b] = C_b^T a_mean
            x_var  : (d,)    Var(x_{b,j,0} | r_b)
            a_map  : (L_b,)  MAP integer state
        """
        pmf = poisson_pmf_vec(lam)
        c_max = len(pmf) - 1
        L_b, d_b = C_b.shape
        p_nz = float(1.0 - pmf[0])
        k_max = max_active_per_block(L_b, p_nz)
        log_pmf = np.log(pmf + 1e-300)

        # Precompute block-level observation columns
        r_count = r_b[:, 0]             # (d,) count channel
        r_fading = r_b[:, 1:] if M > 1 else None   # (d, M-1) or None

        # Enumerate all support-size ≤ k_max states
        states: list[np.ndarray] = [np.zeros(L_b, dtype=np.float64)]
        log_prior: list[float]   = [L_b * log_pmf[0]]

        for k in range(1, k_max + 1):
            lp_zeros = (L_b - k) * log_pmf[0]
            for idxs in combinations(range(L_b), k):
                for cnts in iproduct(range(1, c_max + 1), repeat=k):
                    a = np.zeros(L_b, dtype=np.float64)
                    a[list(idxs)] = np.array(cnts, dtype=np.float64)
                    states.append(a)
                    log_prior.append(lp_zeros + sum(log_pmf[c] for c in cnts))

        A = np.array(states, dtype=np.float64)     # (S, L_b)
        X = A @ C_b                                 # (S, d) : x_b = C_b^T a_b per state

        # ── Count-channel log-likelihood ──────────────────────────────────
        err_count = X - r_count[None, :]            # (S, d)
        ll_count = -np.real(
            np.sum((np.abs(err_count) ** 2) / v_b[None, :], axis=1)
        )                                            # (S,)

        # ── Fading-channel log-likelihood (integrate out U_b) ────────────
        # For each state a_b, Σ_fad = γ_h · C_b^T diag(a_b) C_b^* + diag(v_b)
        # log p(r_b[:,m] | a_b) = -r^H Σ^{-1} r - log|Σ| - d·log(π)
        # We accumulate over antennas m=1..M-1.
        if r_fading is not None and M > 1:
            ll_fading = np.zeros(len(states), dtype=np.float64)
            Ct = C_b.conj().T                        # (d, L_b)  — note: C_b is (L_b, d), rows=msgs
            # C_b^T has shape (d, L_b) (embed from msg→resource space)
            # Codeword subspace in resource space: C_b.T gives (d, L_b)
            # diag(a_b) acts on (L_b,) → Gram = C_b.T @ diag(a_b) @ C_b = (d, d)
            for s_idx, a_b_s in enumerate(states):
                # Σ_fad(a_b) = γ_h · Ct @ diag(a_b) @ Ct^H + diag(v_b)
                # Ct = C_b.T   shape (d, L_b)
                # weighted gram: (d, d)
                weighted_gram = (gamma_h_est * Ct) * a_b_s[None, :]  @ Ct.conj().T  # (d, d)
                Sigma_fad = weighted_gram + np.diag(v_b)              # (d, d)
                try:
                    L_chol = np.linalg.cholesky(Sigma_fad)
                    log_det = 2.0 * np.sum(np.log(np.maximum(np.abs(np.diag(L_chol)), 1e-300)))
                    # r^H Σ^{-1} r via Cholesky: solve L_chol x = r, then ||x||²
                    for m_ant in range(M - 1):
                        r_m = r_fading[:, m_ant]     # (d,)
                        z_m = np.linalg.solve(L_chol, r_m)
                        ll_fading[s_idx] -= np.real(np.dot(z_m.conj(), z_m)) + log_det
                except np.linalg.LinAlgError:
                    # Fallback: use diagonal approximation (Σ_fad ≈ diag)
                    diag_sig = np.real(np.diag(Sigma_fad))
                    for m_ant in range(M - 1):
                        r_m = r_fading[:, m_ant]
                        ll_fading[s_idx] -= np.sum(np.abs(r_m) ** 2 / diag_sig) + np.sum(np.log(diag_sig))
        else:
            ll_fading = np.zeros(len(states), dtype=np.float64)

        # ── Posterior ─────────────────────────────────────────────────────
        log_post = ll_count + ll_fading + np.array(log_prior, dtype=np.float64)
        log_post -= logsumexp(log_post)
        w = np.exp(log_post)                        # (S,)

        a_mean = w @ A                              # (L_b,)
        x_mean = w @ X                              # (d,)
        x_var  = np.maximum(
            np.real(w @ (np.abs(X) ** 2)) - np.abs(x_mean) ** 2,
            var_floor,
        )
        a_map = A[int(np.argmax(log_post))]
        return a_mean, x_mean, x_var, a_map

    # ════════════════════════════════════════════════════════════════════════
    # Main iteration loop
    # ════════════════════════════════════════════════════════════════════════

    for it in range(1, max_iter + 1):
        it_used = it

        # ══════════════════════════════════════════════════════════════════
        # Step A + B: Resource node mean-field x-update + extrinsic messages
        # ══════════════════════════════════════════════════════════════════
        #
        # At resource r:  y_r = Σ_k x_{k,r} h_k + z_r,  z_r ~ CN(0, σ²I_M)
        #
        # Prior: x_{k,r} ~ CN(μ_k^in, v_k^in)  (scalar, Gaussian)
        #        h_k      ~ CN(m_k, R_k)         (vector M, Gaussian)
        #
        # Mean-field symbol update (integrate out h):
        #   G_t[i,i] = tr(R_i) + |m_i|²
        #   G_t[i,j] = m_i^H m_j   (i≠j)
        #   H̄_t      = [m_k] cols → (M, d_t)
        #   Σ̂_t      = (V_t^{-1} + σ^{-2} G_t)^{-1}
        #   μ̂_t      = Σ̂_t (V_t^{-1} μ_t^in + σ^{-2} H̄_t^H y_r)

        for r in range(n):
            edges = resource_to_edges[r]
            if not edges:
                continue

            d_r = len(edges)
            mu_in  = np.array([block_out_mu[b][j]  for b, j in edges], dtype=np.complex128)
            var_in = np.maximum([block_out_var[b][j] for b, j in edges], var_floor)
            # Channel means and covs at this resource
            m_ks  = np.array([ch_mu[b][j]  for b, j in edges])   # (d_r, M)
            R_ks  = [ch_cov[b][j] for b, j in edges]              # list of (M,M)

            # Build G_t (d_r × d_r): G_t[i,j] = m_i^H m_j  (off-diag), G_t[i,i] = tr(R_i) + |m_i|²
            # m_ks shape: (d_r, M) — row i is m_i
            G_t = m_ks @ m_ks.conj().T      # (d_r, d_r): [G_t]_{ij} = m_i · m_j^H ... wrong order
            # Correct: [G_t]_{ij} = m_i^H m_j = conj(m_i) · m_j (inner product in C^M)
            # = (m_ks.conj() @ m_ks.T)[i,j]  since row i of m_ks.conj() is conj(m_i)
            G_t = m_ks.conj() @ m_ks.T      # (d_r, d_r), off-diagonal correct
            # Fix diagonal: tr(R_i) + |m_i|^2
            for i in range(d_r):
                G_t[i, i] = float(np.real(np.trace(R_ks[i]))) + float(np.real(np.dot(m_ks[i].conj(), m_ks[i])))

            H_bar = m_ks.T      # (M, d_r) — each column is m_k

            # Mean-field x-update: Σ̂_t = (V^{-1} + σ^{-2} G_t)^{-1}
            # G_t is Hermitian (G_t = G_t^H) since [G_t]_{ij} = m_i^H m_j
            V_inv = np.diag(1.0 / var_in)                          # (d_r, d_r) real diagonal
            Omega = V_inv + (1.0 / noise_var) * G_t                # (d_r, d_r) Hermitian
            Omega = 0.5 * (Omega + Omega.conj().T)                  # enforce Hermitian numerically
            try:
                Sigma_hat = np.linalg.inv(Omega)                    # (d_r, d_r)
            except np.linalg.LinAlgError:
                Sigma_hat = np.linalg.pinv(Omega)

            Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.conj().T)     # symmetrise

            info_in = mu_in / var_in                                # (d_r,)  η_in
            matched = H_bar.conj().T @ Y[r]                        # (d_r,)  H̄^H y_r
            mu_hat = Sigma_hat @ (info_in + (1.0 / noise_var) * matched)   # (d_r,)

            # Store posterior moments for channel update step
            post_mu_res[r]  = mu_hat         # (d_r,)
            post_var_res[r] = np.maximum(np.real(np.diag(Sigma_hat)), var_floor)  # (d_r,)

            # Extrinsic x-messages (Gaussian division in information domain)
            hat_tau = np.maximum(1.0 / post_var_res[r], tau_floor)
            hat_eta = mu_hat / post_var_res[r]
            tau_in  = np.maximum(1.0 / var_in, tau_floor)
            eta_in  = mu_in / var_in

            tau_ext = np.maximum(hat_tau - tau_in, tau_floor)
            eta_ext = hat_eta - eta_in

            for idx, (b, j) in enumerate(edges):
                block_in_mu[b][j]  = eta_ext[idx] / tau_ext[idx]
                block_in_var[b][j] = max(1.0 / float(tau_ext[idx]), var_floor)

        # ══════════════════════════════════════════════════════════════════
        # Step C: Channel node update (Gaussian, pools over S(k))
        # ══════════════════════════════════════════════════════════════════
        #
        # For each (b, j) symbol-node (representing a "potential user slot"):
        #   Pooling over all resources r ∈ S(b) where position j appears:
        #     ν_{b,j,r} = |μ̂_{b,j,r}|² + v̂_{b,j,r}   (E[|x|²] at that resource)
        #     resid_r   = y_r - Σ_{(b',j')≠(b,j)} m_{b',j'} μ̂_{b',j',r}
        #   Channel update:
        #     Λ_{b,j}^post = (γ_h^{-1} I + σ^{-2} Σ_r ν_{b,j,r} I)^{-1}
        #     m_{b,j}^post = Λ^post (σ^{-2} Σ_r μ̂_{b,j,r}^* resid_r)
        #
        # Note: each (b,j) corresponds to exactly one resource per position in the block.

        if it % channel_update_interval == 0:
            for b, C_b in block_dicts.items():
                S_b = block_supports[b]          # (d,) global resource indices
                d_b = len(S_b)

                for j in range(d_b):
                    r = int(S_b[j])
                    edges_r = resource_to_edges[r]
                    if post_mu_res[r] is None:
                        continue

                    # Find the local index of (b, j) in edges_r
                    local_idx = None
                    for idx_r, (bb, jj) in enumerate(edges_r):
                        if bb == b and jj == j:
                            local_idx = idx_r
                            break
                    if local_idx is None:
                        continue

                    mu_hat_bj = post_mu_res[r][local_idx]      # scalar
                    v_hat_bj  = post_var_res[r][local_idx]     # scalar

                    nu_bj = float(abs(mu_hat_bj) ** 2) + float(v_hat_bj)

                    # Residual at resource r excluding (b,j):
                    # resid = y_r - Σ_{(b',j') ≠ (b,j)} m_{b',j'} μ̂_{b',j',r}
                    mu_all  = post_mu_res[r]                    # (d_r,)
                    m_ks_all = np.array([ch_mu[bb][jj] for bb, jj in edges_r])  # (d_r, M)
                    contrib_all = (m_ks_all.T * mu_all[None, :]).sum(axis=1)     # (M,)
                    contrib_self = ch_mu[b][j] * mu_hat_bj      # (M,)
                    resid_r = Y[r] - contrib_all + contrib_self  # (M,)

                    # Channel update (isotropic prior γ_h I)
                    lam_post_scalar = 1.0 / (1.0 / gamma_h + nu_bj / noise_var)
                    m_post = lam_post_scalar * (np.conj(mu_hat_bj) / noise_var) * resid_r  # (M,)

                    # Damp channel update
                    alpha_ch = 1.0 - damping
                    ch_mu[b][j]  = alpha_ch * m_post + damping * ch_mu[b][j]
                    ch_cov[b][j] = lam_post_scalar * np.eye(M, dtype=np.complex128)

        # ══════════════════════════════════════════════════════════════════
        # Step D + E: Block nodes — augmented discrete posterior + extrinsic x-messages
        # ══════════════════════════════════════════════════════════════════

        # Collect matrix pseudo-observation for each block:
        # R_b ∈ C^{d, M}: column 0 from extrinsic x-messages; columns 1..M-1
        # from the actual Y projected onto the block resources.

        delta = 0.0
        total_mean_count  = 0.0
        total_x_var_post  = 0.0
        total_ch_var_post = 0.0   # Σ tr(R_k) for σ² EM update

        for b, C_b in block_dicts.items():
            S_b = block_supports[b]   # (d,) global resource indices
            d_b = len(S_b)

            # Column 0: extrinsic x-messages (count channel proxy)
            r_b_count = block_in_mu[b]                             # (d,) complex
            v_b       = np.maximum(block_in_var[b], var_floor)     # (d,) real

            # Columns 1..M-1: direct Y columns at block resources
            # Shape: (d_b, M-1) — these carry fading information
            Y_block = Y[S_b, :]                                     # (d_b, M)

            # Build R_b: (d_b, M) — col 0 is count proxy, rest from Y
            R_b = np.empty((d_b, M), dtype=np.complex128)
            R_b[:, 0] = r_b_count
            if M > 1:
                R_b[:, 1:] = Y_block[:, 1:]

            a_mean, x_mean, x_var, a_map = decode_block_v3(
                C_b, R_b, v_b, lambda_est, gamma_h, noise_var
            )
            coeffs_hat[b] = a_mean
            coeffs_map[b] = a_map
            total_mean_count += float(np.sum(a_mean))
            total_x_var_post += float(np.sum(x_var))
            # Channel variance contribution: E_q[Σ_i a_{b,i}] * γ_h * M
            # (expected total channel energy from active transmissions in this block)
            total_ch_var_post += float(np.sum(a_mean)) * gamma_h * M

            # Extrinsic block→resource x-messages
            tau_post = np.maximum(1.0 / x_var, tau_floor)
            tau_in_b = np.maximum(1.0 / v_b,   tau_floor)
            tau_ext  = np.maximum(tau_post - tau_in_b, tau_floor)
            eta_ext  = x_mean * tau_post - r_b_count * tau_in_b

            mu_ext  = eta_ext / tau_ext
            var_ext = 1.0 / tau_ext

            # Damp in information domain
            tau_old = np.maximum(1.0 / np.maximum(block_out_var[b], var_floor), tau_floor)
            eta_old = block_out_mu[b] * tau_old

            tau_damp = (1.0 - damping) * tau_ext + damping * tau_old
            eta_damp = (1.0 - damping) * eta_ext + damping * eta_old

            mu_new  = eta_damp / tau_damp
            var_new = np.maximum(1.0 / tau_damp, var_floor)

            delta = max(delta, float(np.max(np.abs(mu_new - block_out_mu[b]))))
            block_out_mu[b]  = mu_new
            block_out_var[b] = var_new

        # ══════════════════════════════════════════════════════════════════
        # EM updates
        # ══════════════════════════════════════════════════════════════════

        # λ: Poisson rate
        lambda_est = max(total_mean_count / total_L, 1e-12)

        # σ²: noise variance — E_q[||Y - Σ_b P_b C_b^T U_b||²_F / (nM)]
        # E[||Y - Ŷ||²_F] ≈ ||Y - Ê[Y]||²_F + uncertainty terms
        # Estimated signal: Ŷ[:,0] from count channel; Ŷ[:,m] ≈ 0 (zero channel mean → zero signal mean)
        y_hat_col0 = np.zeros(n, dtype=np.complex128)
        for b, C_b in block_dicts.items():
            y_hat_col0[block_supports[b]] += C_b.T @ coeffs_hat[b]
        # For fading antennas: mean channel is zero → mean signal is zero;
        # uncertainty is captured by total_ch_var_post and total_x_var_post.
        resid_col0 = Y[:, 0] - y_hat_col0
        resid_energy = float(np.real(np.dot(resid_col0.conj(), resid_col0)))
        if M > 1:
            resid_fading = Y[:, 1:]                                 # (n, M-1) — mean is 0
            resid_energy += float(np.real(np.vdot(resid_fading.ravel(), resid_fading.ravel())))
        # Uncertainty: Var(x) term for col 0 + Var(U_b) for fading cols
        uncertainty = total_x_var_post + total_ch_var_post
        noise_var = max((resid_energy + uncertainty) / (n * M), var_floor)

        # γ_h: channel power per component, EM update.
        # E[||h_u||²] = M · γ_h  under h_u ~ CN(0, γ_h I_M).
        # We estimate γ_h as the average second moment per component across expected-active slots.
        # Only slots with E[a_{b,j}] > threshold contribute meaningfully.
        #
        # γ_h_new = Σ_{b,j} E[a_{b,j}] · (||m_{b,j}||² + tr(R_{b,j})) / (M · Σ_{b,j} E[a_{b,j}])
        #
        # This correctly weights by expected activity so inactive (m≈0, R≈γ_h I) slots do not
        # dominate and drive γ_h toward the prior (which would collapse to 0 if channels
        # haven't been estimated yet).
        activity_threshold = 1e-3
        numerator_gh   = 0.0
        denominator_gh = 0.0
        for b, C_b in block_dicts.items():
            a_b_mean = coeffs_hat[b]    # (L_b,)
            L_b_size = C_b.shape[0]
            for j in range(L_b_size):
                w_j = float(a_b_mean[j])
                if w_j < activity_threshold:
                    continue
                ch_second_mom = (float(np.real(np.dot(ch_mu[b][j].conj(), ch_mu[b][j]))) +
                                 float(np.real(np.trace(ch_cov[b][j]))))
                numerator_gh   += w_j * ch_second_mom
                denominator_gh += w_j * M
        if denominator_gh > 0:
            gamma_h = max(numerator_gh / denominator_gh, 1e-4)
        # else: keep gamma_h from previous iteration (no update if no active slots found)

        history.append({
            "delta": delta,
            "lambda": lambda_est,
            "noise_var": noise_var,
            "gamma_h": gamma_h,
            "k_est": total_mean_count,
        })
        print(
            f"[iter {it:03d}] delta={delta:.3e} k_est={total_mean_count:.2f} "
            f"lambda={lambda_est:.3e} sigma2={noise_var:.3e}",
            flush=True,
        )

        if delta < tol:
            converged = True
            break

    return coeffs_hat, coeffs_map, {
        "converged": converged,
        "history": history,
        "iterations": it_used,
        "tol": tol,
        "damping": damping,
        "lambda_est": lambda_est,
        "noise_var_est": noise_var,
        "gamma_h_est": gamma_h,
        "lambda_init": lambda_init,
        "noise_var_init": noise_var_init,
        "gamma_h_init": gamma_h_init,
        "poisson_tail_tol": poisson_tail_tol,
        "support_tail_tol": support_tail_tol,
        "channel_update_interval": channel_update_interval,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Assembly and evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_global_counts(
    block_coeffs: dict[int, np.ndarray],
    block_to_msg_list: dict[int, list[int]],
    num_codewords: int,
) -> np.ndarray:
    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, a_b in block_coeffs.items():
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_b[local_idx]
    return counts


def evaluate_counts(
    counts_true: np.ndarray,
    counts_soft: np.ndarray,
    counts_hard: np.ndarray,
) -> dict:
    supp_true = counts_true > 0
    supp_hard = counts_hard > 0
    tp = int(np.sum(supp_true & supp_hard))
    fp = int(np.sum(~supp_true & supp_hard))
    fn = int(np.sum(supp_true & ~supp_hard))
    return dict(
        l1_soft=float(np.sum(np.abs(counts_true - counts_soft))),
        mse_soft=float(np.mean((counts_true - counts_soft) ** 2)),
        support_true=int(np.sum(supp_true)),
        support_map=int(np.sum(supp_hard)),
        tp=tp, fp=fp, fn=fn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        count_errors=int(np.sum(counts_hard[supp_true] != counts_true[supp_true].astype(int))),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_diagnostics(
    codebook, P_mats, msg_to_block, block_to_msg_list,
    active_msgs, channels, message_counts, block_coeffs,
    Y_clean, Y_noisy, noise_var, esn0_db,
):
    num_blocks = len(P_mats)
    n, d = next(iter(P_mats.values())).shape
    M = Y_noisy.shape[1]
    K = len(active_msgs)

    print("\n" + "=" * 60)
    print("ODMA + URA Simulation — V3 (multi-antenna, unknown Rayleigh fading)")
    print("=" * 60)
    print(f"\nCodebook          : {codebook.shape}  (num_codewords × d)  [complex]")
    print(f"Resource grid     : n = {n}")
    print(f"Block size        : d = {d}")
    print(f"Num blocks        : {num_blocks}")
    print(f"Num antennas      : M = {M}")
    print(f"Active devices    : K = {K}")
    print(f"Channel model     : h_u ~ CN(0, I_M),  fully unknown to decoder")
    print(f"Es/N0             : {esn0_db:.1f} dB")
    print(f"Noise variance    : σ² = {noise_var:.6f}")

    ch_norms = np.linalg.norm(channels, axis=1)
    print(f"\nChannel statistics (true):")
    print(f"  ||h_u||  min/mean/max : {ch_norms.min():.3f} / {ch_norms.mean():.3f} / {ch_norms.max():.3f}")

    print(f"\nActive messages (per user) : {active_msgs.tolist()}")
    print(f"Active blocks (per user)   : {[msg_to_block[int(m)] for m in active_msgs]}")

    num_unique = int(np.count_nonzero(message_counts))
    print(f"\nDecoder target:")
    print(f"  Unique messages : {num_unique} / {len(message_counts)}")
    print(f"  Max multiplicity: {int(message_counts.max())}")
    ai = np.nonzero(message_counts)[0]
    print(f"  Active msg→count: {dict(zip(ai.tolist(), message_counts[ai].astype(int).tolist()))}")

    print(f"\nPer-block coefficient summary:")
    for b in range(num_blocks):
        a_b = block_coeffs[b]
        nnz = int(np.count_nonzero(a_b))
        print(f"  block {b:3d}: {len(block_to_msg_list[b]):4d} msgs, nnz={nnz:3d}, total={int(a_b.sum()):3d}")

    resource_usage = np.zeros(n, dtype=int)
    for b in range(num_blocks):
        resource_usage += np.any(P_mats[b], axis=1).astype(int)
    print(f"\nResource usage  max blocks/resource: {resource_usage.max()},  unused: {int((resource_usage==0).sum())}/{n}")
    print(f"\nReceived signal:")
    print(f"  Y shape         : {Y_noisy.shape}")
    print(f"  ||Y_clean||_F   : {np.linalg.norm(Y_clean):.4f}")
    print(f"  ||Y_noisy||_F   : {np.linalg.norm(Y_noisy):.4f}")
    print(f"  ||Z||_F         : {np.linalg.norm(Y_noisy - Y_clean):.4f}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Results saving
# ═══════════════════════════════════════════════════════════════════════════════

def make_slug(args) -> str:
    return (
        f"v3_n{args.n}_d{args.d}_B{args.num_blocks}"
        f"_M{args.num_codewords}_K{args.num_devices_active}"
        f"_ant{args.num_antennas}_snr{args.esn0_db:.0f}dB_s{args.seed}"
    )


def save_results(
    out_dir: Path, args, meta: dict, metrics: dict,
    message_counts: np.ndarray,
    counts_soft: np.ndarray,
    counts_map: np.ndarray,
    P_mats: dict,
    Y_clean: np.ndarray,
    Y_noisy: np.ndarray,
    channels: np.ndarray,
    noise_var_true: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = meta.get("history", [])
    iters   = list(range(1, len(history) + 1))
    n_res   = Y_clean.shape[0]
    M       = Y_clean.shape[1]
    num_blocks = len(P_mats)
    C = {"blue":   "#4C78C8", "orange": "#E07B2A", "green":  "#3BAA5C",
         "red":    "#C84C4C", "grey":   "#888888", "purple": "#8B5CF6",
         "teal":   "#2A9D8F"}

    # ── Fig 1: convergence (4 panels: delta, lambda, sigma2, gamma_h) ──────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.suptitle("V3 Convergence — per-iteration EM diagnostics", fontsize=11, y=1.01)

    ax = axes[0]
    ax.semilogy(iters, [h["delta"] for h in history], color=C["blue"], lw=2, marker="o", ms=4)
    ax.axhline(meta["tol"], color=C["red"], lw=1, ls="--", label=f"tol={meta['tol']:.0e}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Max msg Δ (log)")
    ax.set_title("Message convergence"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, [h["lambda"] for h in history], color=C["orange"], lw=2, marker="o", ms=4, label="λ_est")
    ax.axhline(args.num_devices_active / args.num_codewords, color=C["grey"],
               lw=1.5, ls="--", label="λ_true = K/M")
    ax.set_xlabel("Iteration"); ax.set_ylabel("λ")
    ax.set_title("Poisson rate λ"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(iters, [h["noise_var"] for h in history], color=C["purple"], lw=2, marker="o", ms=4, label="σ²_est")
    ax.axhline(noise_var_true, color=C["grey"], lw=1.5, ls="--", label=f"σ²_true={noise_var_true:.4f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("σ²")
    ax.set_title("Noise variance σ²"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(iters, [h["gamma_h"] for h in history], color=C["teal"], lw=2, marker="o", ms=4, label="γ_h_est")
    ax.axhline(1.0, color=C["grey"], lw=1.5, ls="--", label="γ_h_true = 1.0")
    ax.set_xlabel("Iteration"); ax.set_ylabel("γ_h")
    ax.set_title("Channel power γ_h"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: count estimates ──────────────────────────────────────────────
    active_idx = np.nonzero(message_counts)[0]
    n_active = len(active_idx)
    x = np.arange(n_active)
    w = 0.28

    fig, ax = plt.subplots(figsize=(max(8, n_active * 0.7 + 2), 4))
    ax.bar(x - w, message_counts[active_idx], w, label="True",      color=C["blue"],   alpha=0.85)
    ax.bar(x,     counts_soft[active_idx],    w, label="MMSE soft", color=C["orange"], alpha=0.85)
    ax.bar(x + w, counts_map[active_idx],     w, label="MAP hard",  color=C["green"],  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in active_idx], fontsize=8)
    ax.set_xlabel("Message index"); ax.set_ylabel("Count")
    ax.set_title("True vs estimated counts (active messages only)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_dir / "count_estimates.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: ODMA pattern heatmap ─────────────────────────────────────────
    resource_usage = np.zeros(n_res, dtype=int)
    pattern_mat = np.zeros((num_blocks, n_res), dtype=np.float32)
    for b, P in P_mats.items():
        mask = np.any(P, axis=1)
        pattern_mat[b] = mask.astype(float)
        resource_usage += mask.astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), layout="constrained",
                             gridspec_kw={"height_ratios": [num_blocks, 1]})
    axes[0].imshow(pattern_mat, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_ylabel("Block"); axes[0].set_title("ODMA pattern matrix")
    axes[0].set_yticks(range(num_blocks))
    axes[1].bar(range(n_res), resource_usage, color=C["blue"], alpha=0.7, width=1.0)
    axes[1].set_xlim(-0.5, n_res - 0.5)
    axes[1].set_ylabel("# blocks"); axes[1].set_xlabel("Resource index")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "odma_patterns.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 4: received signal (magnitude of matched-filtered col 0) ────────
    y_clean_col0 = np.real(Y_clean[:, 0])
    y_noisy_col0_mf = np.real(Y_noisy[:, 0])   # col 0 of received
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, layout="constrained")
    r_idx = np.arange(n_res)
    for m in range(min(M, 4)):
        axes[0].plot(r_idx, np.real(Y_noisy[:, m]), color=C["orange"], lw=0.4, alpha=0.2)
    axes[0].plot(r_idx, y_clean_col0, color=C["blue"],   lw=1.3, label="Y_clean col 0 (Re)")
    axes[0].plot(r_idx, y_noisy_col0_mf, color=C["orange"], lw=0.9, label="Y_noisy col 0 (Re)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Received signal — col 0 (faint = other antennas)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(r_idx, np.real(Y_noisy[:, 0]) - y_clean_col0, color=C["grey"], lw=0.8, label="Noise col 0 (Re)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Resource index"); axes[1].set_ylabel("Noise (Re)")
    axes[1].set_title(f"Noise  σ²_true={noise_var_true:.4f},  σ²_est={meta['noise_var_est']:.4f}")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.savefig(out_dir / "received_signal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: channel norm histogram ───────────────────────────────────────
    ch_norms = np.linalg.norm(channels, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ch_norms, bins=max(5, len(ch_norms) // 2), color=C["teal"], alpha=0.75, edgecolor="white")
    ax.axvline(np.sqrt(M), color=C["red"], lw=1.5, ls="--", label=f"E[||h||]≈√M={np.sqrt(M):.2f}")
    ax.set_xlabel("||h_u||"); ax.set_ylabel("Count")
    ax.set_title(f"Channel norms  (h_u ~ CN(0, I_{M}))"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "channel_norms.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 6: per-block counts ─────────────────────────────────────────────
    block_ids = sorted(P_mats.keys())
    true_counts_pb = [float(np.sum(message_counts[[m for m in range(args.num_codewords) if m % args.num_blocks == b]])) for b in block_ids]
    soft_counts_pb = [float(np.sum(counts_soft[[m for m in range(args.num_codewords) if m % args.num_blocks == b]])) for b in block_ids]
    map_counts_pb  = [float(np.sum(counts_map[[m for m in range(args.num_codewords) if m % args.num_blocks == b]])) for b in block_ids]

    x = np.arange(num_blocks); ww = 0.28
    fig, ax = plt.subplots(figsize=(max(7, num_blocks * 0.8 + 2), 4))
    ax.bar(x - ww, true_counts_pb, ww, label="True",      color=C["blue"],   alpha=0.85)
    ax.bar(x,      soft_counts_pb, ww, label="MMSE soft", color=C["orange"], alpha=0.85)
    ax.bar(x + ww, map_counts_pb,  ww, label="MAP hard",  color=C["green"],  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in block_ids], fontsize=9)
    ax.set_xlabel("Block"); ax.set_ylabel("Total count")
    ax.set_title("Per-block total active count (true vs estimated)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "per_block_counts.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Markdown summary ────────────────────────────────────────────────────
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    prec, rec  = metrics["precision"], metrics["recall"]
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    active_map_dict  = {int(m): int(counts_map[m])      for m in np.nonzero(counts_map)[0]}
    active_true_dict = {int(m): int(message_counts[m])  for m in np.nonzero(message_counts)[0]}

    md = textwrap.dedent(f"""
    # ODMA + URA V3 — Run Results
    **Slug:** `{out_dir.name}`
    **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## Setup
    | Parameter | Value |
    |-----------|-------|
    | Resource grid n | {args.n} |
    | Codeword length d | {args.d} |
    | Num blocks B | {args.num_blocks} |
    | Num codewords M | {args.num_codewords} |
    | Active devices K | {args.num_devices_active} |
    | Es/N0 | {args.esn0_db:.1f} dB |
    | Num antennas | {args.num_antennas} |
    | Channel model | CN(0, I_M) i.i.d. Rayleigh, fully unknown |
    | Seed | {args.seed} |
    | Max iterations | {args.max_iter} |
    | Damping | {meta['damping']} |
    | Channel update interval | {meta['channel_update_interval']} |

    ## Decoder Convergence & EM Estimates
    | | Value |
    |--|--|
    | Converged | {meta['converged']} |
    | Iterations used | {meta['iterations']} / {args.max_iter} |
    | λ_true = K/M | {args.num_devices_active / args.num_codewords:.4f} |
    | λ_est (final) | {meta['lambda_est']:.4f} |
    | σ²_true | {noise_var_true:.6f} |
    | σ²_est (final) | {meta['noise_var_est']:.6f} |
    | σ²_est / σ²_true | {meta['noise_var_est'] / noise_var_true:.3f} |
    | γ_h_true | 1.0 (CN(0,I_M) prior) |
    | γ_h_est (final) | {meta['gamma_h_est']:.4f} |

    ## Detection Metrics (MAP decisions)
    | Metric | Value |
    |--------|-------|
    | Support true | {metrics['support_true']} |
    | Support estimated | {metrics['support_map']} |
    | TP | {tp} |
    | FP | {fp} |
    | FN | {fn} |
    | Precision | {prec:.4f} |
    | Recall | {rec:.4f} |
    | F1 | {f1:.4f} |
    | Count errors (on TP) | {metrics['count_errors']} |

    ## Soft Metrics (MMSE)
    | Metric | Value |
    |--------|-------|
    | L1 error | {metrics['l1_soft']:.4e} |
    | MSE | {metrics['mse_soft']:.4e} |

    ## Ground Truth Counts
    ```
    {active_true_dict}
    ```

    ## MAP Estimated Counts
    ```
    {active_map_dict}
    ```

    ## Plots
    - `convergence.png` — per-iteration delta, λ, σ², γ_h
    - `count_estimates.png` — true vs MMSE vs MAP per active message
    - `odma_patterns.png` — block×resource pattern matrix
    - `received_signal.png` — received signal and noise trace
    - `channel_norms.png` — histogram of true channel norms
    - `per_block_counts.png` — per-block total count
    """).strip()

    (out_dir / "results.md").write_text(md)

    raw = {
        "args": vars(args),
        "meta": {k: v for k, v in meta.items() if k != "history"},
        "metrics": metrics,
        "noise_var_true": noise_var_true,
        "gamma_h_true": 1.0,
        "history": history,
    }
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ODMA + URA Decoder — V3 (multi-antenna, unknown Rayleigh fading)")
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--n",                    type=int,   default=128, help="resource grid size")
    parser.add_argument("--d",                    type=int,   default=16,  help="codeword length / block size")
    parser.add_argument("--num-blocks",           type=int,   default=8)
    parser.add_argument("--num-codewords",        type=int,   default=64)
    parser.add_argument("--num-devices-active",   type=int,   default=10)
    parser.add_argument("--esn0-db",              type=float, default=10.0)
    parser.add_argument("--num-antennas",         type=int,   default=4,   help="M receive antennas")
    parser.add_argument("--max-iter",             type=int,   default=50)
    parser.add_argument("--damping",              type=float, default=0.3)
    parser.add_argument("--lambda-init",          type=float, default=None)
    parser.add_argument("--noise-var-init",       type=float, default=None)
    parser.add_argument("--gamma-h-init",         type=float, default=None,
                        help="Initial channel power per component (true=1.0 for CN(0,I))")
    parser.add_argument("--channel-update-interval", type=int, default=1,
                        help="Run channel node update every N iterations (1=every iter)")
    parser.add_argument("--poisson-tail-tol",     type=float, default=1e-4)
    parser.add_argument("--support-tail-tol",     type=float, default=1e-4)
    parser.add_argument("--results-dir",          type=str,   default="results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Setup ──────────────────────────────────────────────────────────────
    codebook          = make_codebook(args.num_codewords, args.d, rng)
    blocks            = make_odma_blocks(args.num_blocks, args.n, args.d, rng)
    P_mats            = build_pattern_matrices(blocks, args.n)
    msg_to_block, block_to_msg_list = make_message_block_mapping(args.num_codewords, args.num_blocks)
    block_dicts       = build_block_dictionaries(codebook, block_to_msg_list, args.num_blocks)

    active_msgs       = sample_active_messages(args.num_devices_active, args.num_codewords, rng)
    channels          = make_user_channels(args.num_devices_active, args.num_antennas, rng)
    message_counts    = build_message_counts(active_msgs, args.num_codewords)
    block_coeffs      = build_block_coefficients(active_msgs, msg_to_block, block_to_msg_list, args.num_blocks)
    noise_var         = esn0_db_to_noise_var(args.esn0_db, args.d)

    Y_noisy, Y_clean = synthesize_received_signal_v3(
        active_msgs, msg_to_block, P_mats, codebook, channels, noise_var, rng)

    # ── Decode ─────────────────────────────────────────────────────────────
    coeffs_hat, coeffs_map, meta = graph_based_decoder_v3(
        Y_noisy, P_mats, block_dicts,
        max_iter=args.max_iter,
        damping=args.damping,
        lambda_init=args.lambda_init,
        noise_var_init=args.noise_var_init,
        gamma_h_init=args.gamma_h_init,
        poisson_tail_tol=args.poisson_tail_tol,
        support_tail_tol=args.support_tail_tol,
        channel_update_interval=args.channel_update_interval,
    )

    counts_soft = assemble_global_counts(coeffs_hat, block_to_msg_list, args.num_codewords)
    counts_map  = assemble_global_counts(coeffs_map,  block_to_msg_list, args.num_codewords)
    metrics     = evaluate_counts(message_counts, counts_soft, counts_map)

    # ── Terminal output ─────────────────────────────────────────────────────
    print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list,
                      active_msgs, channels, message_counts, block_coeffs,
                      Y_clean, Y_noisy, noise_var, args.esn0_db)
    print(f"Decoder meta: { {k: v for k, v in meta.items() if k != 'history'} }")
    print(f"Decoder eval: {metrics}")
    active_map = {int(m): int(counts_map[m]) for m in np.nonzero(counts_map)[0]}
    print(f"Counts MAP (active): {active_map}")

    # ── Save ───────────────────────────────────────────────────────────────
    slug    = make_slug(args)
    out_dir = Path(args.results_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(out_dir, args, meta, metrics, message_counts,
                 counts_soft, counts_map, P_mats, Y_clean, Y_noisy,
                 channels, noise_var)
    print(f"\nResults saved → {out_dir}/")

    return dict(
        codebook=codebook, P_mats=P_mats,
        msg_to_block=msg_to_block, block_to_msg_list=block_to_msg_list,
        block_dicts=block_dicts,
        active_msgs=active_msgs, channels=channels,
        message_counts=message_counts, block_coeffs=block_coeffs,
        Y_clean=Y_clean, Y_noisy=Y_noisy, noise_var=noise_var,
    )


if __name__ == "__main__":
    main()