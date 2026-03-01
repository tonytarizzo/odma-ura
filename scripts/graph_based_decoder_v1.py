"""
ODMA + URA Decoder Testbed — V1: No fading, single receive stream
=================================================================
Simplest signal model for developing the message-count decoder.

Signal model:
    y = sum_b P_b C_b^T a_b + z       ∈ R^n  (or C^n)

  - Global codebook of unit-norm codewords, shape (num_codewords, d).
  - Each global message index m deterministically selects codebook[m]
    and ODMA pattern msg_to_block[m].
  - P_b ∈ {0,1}^{n×d}, P_b^T P_b = I_d.
  - No antennas, no fading, no channels.
  - Multiple devices choosing the same message add as integer counts
    in the coefficient vector a_b.  The physical signal is fully
    determined by the blockwise multiplicity vectors {a_b}.
  - z is i.i.d. AWGN.

Decoder target:
    message_counts ∈ Z_+^{num_codewords}, directly aligned with
    the physical signal's sparse coefficient structure.

Run:
    python graph_based_decoder_v1.py --seed 42 --n 128 --d 16 --num-blocks 8 \
        --num-codewords 64 --num-devices-active 10 --esn0-db 10
"""

from __future__ import annotations
import argparse
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
import numpy as np


def make_codebook(num_codewords: int, d: int, rng: np.random.Generator, complex_valued: bool = False) -> np.ndarray:
    """Random Gaussian codebook with unit-normalized rows.  Returns (num_codewords, d)."""
    if complex_valued:
        raw = rng.standard_normal((num_codewords, d)) + 1j * rng.standard_normal((num_codewords, d))
    else:
        raw = rng.standard_normal((num_codewords, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_odma_blocks(num_blocks: int, n: int, d: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random ODMA blocks. Each block is a sorted array of d resource indices drawn without replacement from {0, ..., n-1}."""
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(num_blocks)]


def build_pattern_matrices(blocks: list[np.ndarray], n: int) -> dict[int, np.ndarray]:
    """Build per-block embedding matrices P_b ∈ {0,1}^{n×d}. Column j of P_b is e_{S_b[j]}. Satisfies P_b^T P_b = I_d."""
    P_mats: dict[int, np.ndarray] = {}
    for b, S_b in enumerate(blocks):
        d = len(S_b)
        P = np.zeros((n, d), dtype=np.float64)
        P[S_b, np.arange(d)] = 1.0
        P_mats[b] = P
    return P_mats


def make_message_block_mapping(num_codewords: int, num_blocks: int):
    """Deterministic mapping: message m → pattern msg_to_block[m] = m % num_blocks."""
    msg_to_block: dict[int, int] = {}
    block_to_msg_list: dict[int, list[int]] = {b: [] for b in range(num_blocks)}
    for m in range(num_codewords):
        b = m % num_blocks
        msg_to_block[m] = b
        block_to_msg_list[b].append(m)
    return msg_to_block, block_to_msg_list


def sample_active_messages(num_devices_active: int, num_codewords: int, rng: np.random.Generator) -> np.ndarray:
    """Each active device independently picks a message index uniformly at random. Duplicates are allowed."""
    return rng.integers(0, num_codewords, size=num_devices_active)


def build_message_counts(active_msgs: np.ndarray, num_codewords: int) -> np.ndarray:
    """Global message count vector — the direct decoder target.  Returns (num_codewords,)."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for m in active_msgs:
        counts[int(m)] += 1.0
    return counts


def build_block_coefficients(active_msgs: np.ndarray, msg_to_block: dict[int, int], block_to_msg_list: dict[int, list[int]], num_blocks: int) -> dict[int, np.ndarray]:
    """Blockwise view of the decoder target: for each block b, a sparse vector of message multiplicities."""
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


def build_block_dictionaries(codebook: np.ndarray, block_to_msg_list: dict[int, list[int]], num_blocks: int) -> dict[int, np.ndarray]:
    """Gather global codewords for each block's assigned messages.  Returns dict block_idx -> (L_b, d)."""
    return {b: codebook[block_to_msg_list[b]] for b in range(num_blocks)}


def esn0_db_to_noise_var(esn0_db: float, d: int) -> float:
    """Convert Es/N0 (dB) to per-entry noise variance. Es = 1/d (unit-norm codewords over d symbols). N0 = 1/(d * esn0_lin)."""
    esn0_lin = 10.0 ** (esn0_db / 10.0)
    return 1.0 / (d * esn0_lin)


def synthesize_received_signal(P_mats: dict[int, np.ndarray], block_dicts: dict[int, np.ndarray], block_coeffs: dict[int, np.ndarray], noise_var: float, rng: np.random.Generator, complex_valued: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize y = sum_b P_b C_b^T a_b + z.  Per-block, no per-user channels.

    Returns:  (y_noisy, y_clean)  both (n,)
    """
    n = next(iter(P_mats.values())).shape[0]
    dtype = np.complex128 if complex_valued else np.float64
    y_clean = np.zeros(n, dtype=dtype)

    for b in P_mats:
        a_b = block_coeffs[b]                  # (L_b,)
        if np.any(a_b):
            C_b = block_dicts[b]               # (L_b, d)
            x_b = C_b.T @ a_b                  # (d,)  block signal in local coords
            y_clean += P_mats[b] @ x_b         # (n,)  embedded into resource grid

    if complex_valued:
        noise = np.sqrt(noise_var / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    else:
        noise = np.sqrt(noise_var) * rng.standard_normal(n)

    return y_clean + noise, y_clean


def graph_based_decoder(
    y: np.ndarray,
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
) -> tuple[dict[int, np.ndarray], dict]:
    """Iterative Gaussian resource update (LMMSE) + exact blockwise discrete Poisson posterior,
    with EM updates for noise variance σ² and Poisson rate λ.

    Graph structure:
      - Resource nodes r: scalar observations y[r] = sum_{(b,j): S_b[j]=r} x_{b,j} + z[r]
      - Block nodes b: discrete sparse posterior over a_b, with x_b = C_b^T a_b

    Message schedule:  block->resource (Gaussian) → resource LMMSE → extrinsic resource->block
                       → block discrete posterior → extrinsic block->resource  (repeat)
                       → EM updates for (λ, σ²)

    All edge messages are Gaussian, parameterised by (mu, var) or equivalently (eta=mu/v, tau=1/v).
    Extrinsic messages are formed by Gaussian division (precision subtraction).

    Neither noise_var nor λ need to be known in advance; both are estimated from the data.
    Provide noise_var_init / lambda_init only to seed the first iteration with a better guess.
    """
    from itertools import combinations, product

    n = y.shape[0]
    dtype = y.dtype
    var_floor = 1e-10
    tau_floor = 1e-10

    # ------------------------------------------------------------------ helpers
    def logsumexp(v: np.ndarray) -> float:
        vmax = float(np.max(v))
        return vmax + float(np.log(np.sum(np.exp(v - vmax)))) if np.isfinite(vmax) else vmax

    def poisson_pmf_vec(lam: float) -> np.ndarray:
        """Truncated Poisson PMF p(0), p(1), ..., p(c_max) normalised to sum to 1."""
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
        """Largest support size k with binomial tail P(Bin(L_b, p_nz) > k) > support_tail_tol."""
        p_nz = float(np.clip(p_nz, 1e-12, 1.0 - 1e-12))
        p0 = 1.0 - p_nz
        pk = p0 ** L_b          # P(support = 0)
        cdf = pk
        for k in range(L_b):
            pk = pk * ((L_b - k) / (k + 1)) * (p_nz / p0)
            cdf += pk
            if 1.0 - cdf <= support_tail_tol:
                return k + 1   # loop body added mass for support size k+1
        return L_b

    # --------------------------------------------------------- decode one block
    def decode_block(
        C_b: np.ndarray,       # (L_b, d)  local dictionary
        r_b: np.ndarray,       # (d,)      incoming mean from resources
        v_b: np.ndarray,       # (d,)      incoming variance from resources
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact enumeration of discrete Poisson posterior over a_b.

        Prior: a_i ~ Poisson(lam) i.i.d., observation: r_b = C_b^T a_b + w, w ~ N(0, diag(v_b))

        Returns:
            a_mean:  (L_b,)  posterior E[a_b | r_b]  — MMSE, used during iterations
            x_mean:  (d,)    posterior E[x_b | r_b] = C_b^T a_mean
            x_var:   (d,)    posterior marginal variances Var(x_{b,j} | r_b)
            a_map:   (L_b,)  argmax_s p(a_b = s | r_b) — MAP integer state, for final decision
        """
        pmf = poisson_pmf_vec(lam)
        c_max = len(pmf) - 1
        L_b = C_b.shape[0]
        p_nz = float(1.0 - pmf[0])
        k_max = max_active_per_block(L_b, p_nz)

        # log p(a_i = c) for c = 0, 1, ..., c_max  (correct Poisson log-probs)
        log_pmf = np.log(pmf + 1e-300)

        # Enumerate all states consistent with support size <= k_max
        states: list[np.ndarray] = [np.zeros(L_b, dtype=np.float64)]
        log_prior: list[float] = [L_b * log_pmf[0]]

        for k in range(1, k_max + 1):
            lp_zeros = (L_b - k) * log_pmf[0]   # contribution of inactive messages
            for idxs in combinations(range(L_b), k):
                for cnts in product(range(1, c_max + 1), repeat=k):
                    a = np.zeros(L_b, dtype=np.float64)
                    a[list(idxs)] = np.array(cnts, dtype=np.float64)
                    states.append(a)
                    log_prior.append(lp_zeros + sum(log_pmf[c] for c in cnts))

        A = np.array(states, dtype=np.float64)        # (S, L_b)
        X = A @ C_b                                    # (S, d):  x_b = C_b^T a_b per state

        # log-likelihood: -||r_b - x_b||^2_{V^{-1}}  (Gaussian, diagonal V)
        err = X - r_b[None, :]                         # (S, d)
        ll = -np.real(np.sum((np.abs(err) ** 2) / v_b[None, :], axis=1))

        log_post = ll + np.array(log_prior, dtype=np.float64)
        log_post -= logsumexp(log_post)                # normalise in log domain
        w = np.exp(log_post)                           # (S,) posterior weights

        a_mean = w @ A                                 # (L_b,)
        x_mean = w @ X                                 # (d,)
        # Diagonal posterior covariance of x_b via law of total variance:
        # Var(x_j) = E[x_j^2] - (E[x_j])^2  (works for real and complex)
        x_var = np.maximum(
            np.real(w @ (np.abs(X) ** 2)) - np.abs(x_mean) ** 2,
            var_floor,
        )
        # MAP: highest posterior probability integer state (free — log_post already computed)
        a_map = A[int(np.argmax(log_post))]            # (L_b,) integer-valued
        return a_mean, x_mean, x_var, a_map

    # ----------------------------------------- build resource → edge adjacency
    # block_supports[b][j] = global resource index for local position j
    block_supports = {b: np.argmax(P_mats[b], axis=0).astype(int) for b in P_mats}

    # resource_to_edges[r] = list of (b, j) pairs: block b, local position j maps to resource r
    resource_to_edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b, S_b in block_supports.items():
        for j, r in enumerate(S_b):
            resource_to_edges[r].append((b, j))

    # ----------------------------------------- initialise edge messages
    # block_out_{mu,var}[b][j]: message from block b to resource S_b[j]  (shape (d,))
    # block_in_{mu,var}[b][j]:  message from resource S_b[j] to block b  (shape (d,))
    # Initialise block->resource with zero-mean, unit variance (uninformative)
    block_out_mu  = {b: np.zeros(C_b.shape[1], dtype=dtype) for b, C_b in block_dicts.items()}
    block_out_var = {b: np.ones(C_b.shape[1], dtype=np.float64) for b, C_b in block_dicts.items()}
    block_in_mu   = {b: np.zeros(C_b.shape[1], dtype=dtype) for b, C_b in block_dicts.items()}
    block_in_var  = {b: np.ones(C_b.shape[1], dtype=np.float64) for b, C_b in block_dicts.items()}

    coeffs_hat = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}
    coeffs_map  = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}

    M = float(sum(C_b.shape[0] for C_b in block_dicts.values()))
    # λ init: prior-free — one expected active device spread across all messages
    lambda_est = float(lambda_init) if lambda_init is not None else 1.0 / M
    # σ² init: received energy / n is a conservative upper bound (pure-noise assumption)
    noise_var = float(noise_var_init) if noise_var_init is not None else float(np.real(np.vdot(y, y))) / n

    converged = False
    it_used = 0
    history: list[dict] = []  # per-iteration diagnostics for analysis

    for it in range(1, max_iter + 1):
        it_used = it

        # =====================================================================
        # Step A+B: Resource nodes — LMMSE update + extrinsic message formation
        # =====================================================================
        # At resource r: y[r] = sum_{(b,j)} x_{b,j} + z[r],  z[r] ~ N(0, noise_var)
        # Incoming prior on each x_{b,j}: CN(mu_{b,j}, v_{b,j})  independent
        # LMMSE (Sherman-Morrison for H=1^T):
        #   var_sum = sum_k v_k + noise_var
        #   hat_mu_k = mu_k + v_k * (y[r] - sum_k mu_k) / var_sum
        #   hat_v_k  = v_k  - v_k^2 / var_sum
        # Extrinsic (divide out incoming prior in information form):
        #   tau_ext_k = 1/hat_v_k - 1/v_k  = 1/(var_sum - v_k)   -- see below
        #   eta_ext_k = hat_mu_k/hat_v_k - mu_k/v_k

        for r in range(n):
            edges = resource_to_edges[r]
            if not edges:
                continue

            mu_in  = np.array([block_out_mu[b][j]  for b, j in edges], dtype=dtype)
            var_in = np.maximum([block_out_var[b][j] for b, j in edges], var_floor)

            var_sum = float(np.sum(var_in)) + noise_var
            resid   = y[r] - np.sum(mu_in)

            # Posterior marginals
            hat_mu  = mu_in + (var_in / var_sum) * resid
            hat_var = var_in - (var_in ** 2) / var_sum    # always < var_in > 0

            # Extrinsic precision: 1/hat_v - 1/v_in = 1/(var_sum - v_in)
            # (exact identity from Sherman-Morrison; always positive since var_sum > v_in)
            tau_ext = np.maximum(1.0 / hat_var - 1.0 / var_in, tau_floor)
            eta_ext = hat_mu / hat_var - mu_in / var_in

            for idx, (b, j) in enumerate(edges):
                block_in_mu[b][j]  = eta_ext[idx] / tau_ext[idx]
                block_in_var[b][j] = 1.0 / tau_ext[idx]

        # =====================================================================
        # Step C+D+E: Block nodes — discrete posterior + extrinsic messages
        # =====================================================================
        delta = 0.0
        total_mean_count = 0.0
        total_x_var_post = 0.0          # sum of posterior Var_q(x_{b,j}) — needed for σ² EM update

        for b, C_b in block_dicts.items():
            r_b = block_in_mu[b]                                # (d,) pseudo-observation means
            v_b = np.maximum(block_in_var[b], var_floor)        # (d,) pseudo-observation variances

            a_mean, x_mean, x_var, a_map = decode_block(C_b, r_b, v_b, lambda_est)
            coeffs_hat[b] = a_mean
            coeffs_map[b] = a_map
            total_mean_count += float(np.sum(a_mean))
            total_x_var_post += float(np.sum(x_var))

            # Extrinsic block->resource: divide posterior marginal by incoming message
            # tau_b->r = 1/x_var - 1/v_b;  eta_b->r = x_mean/x_var - r_b/v_b
            tau_post = np.maximum(1.0 / x_var, tau_floor)
            tau_in   = np.maximum(1.0 / v_b,   tau_floor)
            tau_ext  = np.maximum(tau_post - tau_in, tau_floor)
            eta_ext  = x_mean * tau_post - r_b * tau_in

            mu_ext  = eta_ext / tau_ext
            var_ext = 1.0 / tau_ext

            # Damp in information domain to aid convergence
            tau_old = np.maximum(1.0 / np.maximum(block_out_var[b], var_floor), tau_floor)
            eta_old = block_out_mu[b] * tau_old

            tau_damp = (1.0 - damping) * tau_ext + damping * tau_old
            eta_damp = (1.0 - damping) * eta_ext + damping * eta_old

            mu_new  = eta_damp / tau_damp
            var_new = np.maximum(1.0 / tau_damp, var_floor)

            delta = max(delta, float(np.max(np.abs(mu_new - block_out_mu[b]))))
            block_out_mu[b]  = mu_new
            block_out_var[b] = var_new

        # EM update for λ: E[K_total] / M  (already correct, derive: d/dλ E_q[log p(a|λ)] = 0)
        lambda_est = max(total_mean_count / M, 1e-12)

        # EM update for σ²:  σ²_new = (1/n) * E_q[||y - sum_b P_b x_b||²]
        #   = (1/n) * ( ||y - ŷ||² + sum_b sum_j Var_q(x_{b,j}) )
        # where ŷ = sum_b P_b C_b^T â_b is the posterior-mean signal reconstruction,
        # and the variance sum corrects for using the posterior mean instead of the true signal.
        # x_var_b was accumulated from decode_block above; block_out_var is the extrinsic
        # message (which omits incoming noise), so we must use the direct posterior variances.
        y_hat = np.zeros(n, dtype=dtype)
        for b, C_b in block_dicts.items():
            y_hat[block_supports[b]] += C_b.T @ coeffs_hat[b]
        resid_energy = float(np.real(np.vdot(y - y_hat, y - y_hat)))
        noise_var = max((resid_energy + total_x_var_post) / n, var_floor)

        history.append({"delta": delta, "lambda": lambda_est, "noise_var": noise_var,
                         "k_est": total_mean_count})
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
        "lambda_init": lambda_init,
        "noise_var_init": noise_var_init,
        "poisson_tail_tol": poisson_tail_tol,
        "support_tail_tol": support_tail_tol,
    }


def assemble_global_counts(block_coeffs: dict[int, np.ndarray], block_to_msg_list: dict[int, list[int]], num_codewords: int) -> np.ndarray:
    """Convert blockwise coefficient vectors back to a single global message count vector (num_codewords,)."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, a_b in block_coeffs.items():
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_b[local_idx]
    return counts


def evaluate_counts(counts_true: np.ndarray, counts_soft: np.ndarray, counts_hard: np.ndarray) -> dict:
    """Compare true vs estimated global message count vectors.

    counts_soft: continuous MMSE estimates (for l1/mse soft metrics)
    counts_hard: integer MAP estimates (for support detection and count accuracy)
    """
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


def print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list, active_msgs, message_counts, block_coeffs, y_clean, y_noisy, noise_var, esn0_db):
    num_blocks = len(P_mats)
    n, d = next(iter(P_mats.values())).shape
    K = len(active_msgs)

    print("\n" + "=" * 60)
    print("ODMA + URA Simulation — V1 (no fading, single stream)")
    print("=" * 60)

    print(f"\nCodebook          : {codebook.shape}  (num_codewords × d)")
    print(f"Resource grid     : n = {n}")
    print(f"Block size        : d = {d}")
    print(f"Num blocks        : {num_blocks}")
    print(f"Pattern matrices  : {num_blocks} × ({n}, {d})")
    print(f"Active devices    : K = {K}")
    print(f"Es/N0             : {esn0_db:.1f} dB")
    print(f"Noise variance    : {noise_var:.6f}")

    print(f"\nActive messages (per user) : {active_msgs.tolist()}")
    active_blocks = [msg_to_block[int(m)] for m in active_msgs]
    print(f"Active blocks (per user)  : {active_blocks}")

    num_unique = int(np.count_nonzero(message_counts))
    max_mult = int(message_counts.max())
    print(f"\nDecoder target (global message counts):")
    print(f"  Unique messages    : {num_unique} / {len(message_counts)}")
    print(f"  Max multiplicity   : {max_mult}")
    active_indices = np.nonzero(message_counts)[0]
    print(f"  Active msg → count : {dict(zip(active_indices.tolist(), message_counts[active_indices].astype(int).tolist()))}")

    print(f"\nPer-block coefficient summary:")
    for b in range(num_blocks):
        a_b = block_coeffs[b]
        nnz = int(np.count_nonzero(a_b))
        total = int(a_b.sum())
        max_mult_b = int(a_b.max()) if nnz > 0 else 0
        print(f"  block {b:3d}: {len(block_to_msg_list[b]):4d} msgs, nnz={nnz:3d}, total_users={total:3d}, max_mult={max_mult_b}")

    supports = {b: np.any(P_mats[b], axis=1) for b in range(num_blocks)}
    overlap_counts = []
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            ov = int(np.sum(supports[i] & supports[j]))
            if ov > 0:
                overlap_counts.append((i, j, ov))
    resource_usage = np.zeros(n, dtype=int)
    for b in range(num_blocks):
        resource_usage += supports[b].astype(int)

    print(f"\nBlock overlap statistics:")
    print(f"  Pairs with overlap : {len(overlap_counts)} / {num_blocks * (num_blocks - 1) // 2}")
    if overlap_counts:
        ovs = [c for _, _, c in overlap_counts]
        print(f"  Overlap sizes      : min={min(ovs)}, max={max(ovs)}, mean={np.mean(ovs):.1f}")
    print(f"  Max blocks/resource: {resource_usage.max()}")
    print(f"  Resources unused   : {int((resource_usage == 0).sum())} / {n}")

    print(f"\nReceived signal:")
    print(f"  y shape           : {y_noisy.shape}")
    print(f"  ||y_clean||       : {np.linalg.norm(y_clean):.4f}")
    print(f"  ||y_noisy||       : {np.linalg.norm(y_noisy):.4f}")
    print(f"  ||noise||         : {np.linalg.norm(y_noisy - y_clean):.4f}")
    print("=" * 60 + "\n")


def make_slug(args) -> str:
    """Short human-readable run identifier from key params."""
    cx = "cx" if args.complex_valued else "re"
    return (
        f"v1_{cx}_n{args.n}_d{args.d}_B{args.num_blocks}"
        f"_M{args.num_codewords}_K{args.num_devices_active}"
        f"_snr{args.esn0_db:.0f}dB_s{args.seed}"
    )


def save_results(out_dir: Path, args, meta: dict, metrics: dict,
                 message_counts: np.ndarray, counts_soft: np.ndarray,
                 counts_map: np.ndarray, P_mats: dict, y_clean: np.ndarray,
                 y_noisy: np.ndarray, noise_var_true: float) -> None:
    """Save markdown summary + all analysis plots to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    history = meta.get("history", [])
    iters = list(range(1, len(history) + 1))
    n_resources = y_clean.shape[0]
    num_blocks = len(P_mats)

    # ---------------------------------------------------------------- colour palette
    C = {"blue": "#4C78C8", "orange": "#E07B2A", "green": "#3BAA5C",
         "red": "#C84C4C", "grey": "#888888", "purple": "#8B5CF6"}

    # ============================================================ Fig 1: convergence
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("Convergence — per-iteration EM diagnostics", fontsize=11, y=1.01)

    ax = axes[0]
    ax.semilogy(iters, [h["delta"] for h in history], color=C["blue"], lw=2, marker="o", ms=4)
    ax.axhline(meta["tol"], color=C["red"], lw=1, ls="--", label=f"tol={meta['tol']:.0e}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Max message Δ (log)")
    ax.set_title("Message convergence"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, [h["lambda"] for h in history], color=C["orange"], lw=2, marker="o", ms=4,
            label="λ_est")
    ax.axhline(args.num_devices_active / args.num_codewords, color=C["grey"],
               lw=1.5, ls="--", label="λ_true = K/M")
    ax.set_xlabel("Iteration"); ax.set_ylabel("λ")
    ax.set_title("Poisson rate λ"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(iters, [h["noise_var"] for h in history], color=C["purple"], lw=2, marker="o", ms=4,
            label="σ²_est")
    ax.axhline(noise_var_true, color=C["grey"], lw=1.5, ls="--", label=f"σ²_true={noise_var_true:.4f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("σ²")
    ax.set_title("Noise variance σ²"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ Fig 2: count estimates
    active_idx = np.nonzero(message_counts)[0]
    n_active = len(active_idx)
    x = np.arange(n_active)
    w = 0.28

    fig, ax = plt.subplots(figsize=(max(8, n_active * 0.7 + 2), 4))
    ax.bar(x - w, message_counts[active_idx], w, label="True", color=C["blue"], alpha=0.85)
    ax.bar(x,     counts_soft[active_idx],    w, label="MMSE soft", color=C["orange"], alpha=0.85)
    ax.bar(x + w, counts_map[active_idx],     w, label="MAP hard", color=C["green"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in active_idx], fontsize=8)
    ax.set_xlabel("Message index"); ax.set_ylabel("Count")
    ax.set_title("True vs estimated counts (active messages only)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_dir / "count_estimates.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ Fig 3: ODMA pattern heatmap
    resource_usage = np.zeros(n_resources, dtype=int)
    pattern_mat = np.zeros((num_blocks, n_resources), dtype=np.float32)
    for b, P in P_mats.items():
        mask = np.any(P, axis=1)
        pattern_mat[b] = mask.astype(float)
        resource_usage += mask.astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), layout="constrained",
                             gridspec_kw={"height_ratios": [num_blocks, 1], "hspace": 0.08})
    im = axes[0].imshow(pattern_mat, aspect="auto", interpolation="nearest",
                        cmap="Blues", vmin=0, vmax=1)
    axes[0].set_ylabel("Block"); axes[0].set_xlabel("")
    axes[0].set_title("ODMA pattern matrix  (blue = resource used by block)")
    axes[0].set_yticks(range(num_blocks))

    axes[1].bar(range(n_resources), resource_usage, color=C["blue"], alpha=0.7, width=1.0)
    axes[1].set_xlim(-0.5, n_resources - 0.5)
    axes[1].set_ylabel("# blocks"); axes[1].set_xlabel("Resource index")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "odma_patterns.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ Fig 4: received signal
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, layout="constrained",
                             gridspec_kw={"hspace": 0.1})
    r_idx = np.arange(n_resources)
    axes[0].plot(r_idx, np.real(y_clean), color=C["blue"], lw=1, label="y_clean (Re)")
    axes[0].plot(r_idx, np.real(y_noisy), color=C["orange"], lw=0.7, alpha=0.7, label="y_noisy (Re)")
    axes[0].set_ylabel("Amplitude"); axes[0].set_title("Received signal (real part)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    noise_trace = np.real(y_noisy) - np.real(y_clean)
    axes[1].plot(r_idx, noise_trace, color=C["grey"], lw=0.8, alpha=0.9, label="noise (Re)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Resource index"); axes[1].set_ylabel("Noise")
    axes[1].set_title(f"Noise realisation  (σ²_true={noise_var_true:.4f},  "
                      f"σ²_est={meta['noise_var_est']:.4f})")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.savefig(out_dir / "received_signal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ Fig 5: per-block summary
    block_ids = sorted(P_mats.keys())
    true_counts_per_block  = [float(np.sum(message_counts[
        [m for m in range(args.num_codewords) if m % args.num_blocks == b]
    ])) for b in block_ids]
    soft_counts_per_block  = [float(np.sum(counts_soft[
        [m for m in range(args.num_codewords) if m % args.num_blocks == b]
    ])) for b in block_ids]
    map_counts_per_block   = [float(np.sum(counts_map[
        [m for m in range(args.num_codewords) if m % args.num_blocks == b]
    ])) for b in block_ids]

    x = np.arange(num_blocks); w = 0.28
    fig, ax = plt.subplots(figsize=(max(7, num_blocks * 0.8 + 2), 4))
    ax.bar(x - w, true_counts_per_block, w, label="True",       color=C["blue"],   alpha=0.85)
    ax.bar(x,     soft_counts_per_block, w, label="MMSE soft",  color=C["orange"], alpha=0.85)
    ax.bar(x + w, map_counts_per_block,  w, label="MAP hard",   color=C["green"],  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in block_ids], fontsize=9)
    ax.set_xlabel("Block"); ax.set_ylabel("Total count")
    ax.set_title("Per-block total active count (true vs estimated)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "per_block_counts.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ Markdown summary
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    prec, rec = metrics["precision"], metrics["recall"]
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    active_map_dict = {int(m): int(counts_map[m]) for m in np.nonzero(counts_map)[0]}
    active_true_dict = {int(m): int(message_counts[m]) for m in np.nonzero(message_counts)[0]}

    md = textwrap.dedent(f"""
    # ODMA + URA V1 — Run Results
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
    | Complex-valued | {args.complex_valued} |
    | Seed | {args.seed} |
    | Max iterations | {args.max_iter} |
    | Damping | {meta['damping']} |

    ## Decoder Convergence
    | | Value |
    |--|--|
    | Converged | {meta['converged']} |
    | Iterations used | {meta['iterations']} / {args.max_iter} |
    | λ_true = K/M | {args.num_devices_active / args.num_codewords:.4f} |
    | λ_est (final) | {meta['lambda_est']:.4f} |
    | σ²_true | {noise_var_true:.6f} |
    | σ²_est (final) | {meta['noise_var_est']:.6f} |
    | σ²_est / σ²_true | {meta['noise_var_est'] / noise_var_true:.3f} |

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
    - `convergence.png` — per-iteration delta, λ, σ²
    - `count_estimates.png` — true vs MMSE vs MAP per active message
    - `odma_patterns.png` — block×resource pattern matrix + resource usage histogram
    - `received_signal.png` — clean vs noisy signal and noise trace
    - `per_block_counts.png` — per-block total count (true vs estimated)
    """).strip()

    (out_dir / "results.md").write_text(md)

    # Also dump raw numbers as JSON for programmatic use
    raw = {
        "args": vars(args),
        "meta": {k: v for k, v in meta.items() if k != "history"},
        "metrics": metrics,
        "noise_var_true": noise_var_true,
        "history": history,
    }
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2))


def main():
    parser = argparse.ArgumentParser(description="ODMA + URA scaffold — V1 (no fading, single stream)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=128, help="total resource grid size")
    parser.add_argument("--d", type=int, default=16, help="codeword length / block size")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-codewords", type=int, default=64)
    parser.add_argument("--num-devices-active", type=int, default=10)
    parser.add_argument("--esn0-db", type=float, default=10.0, help="Es/N0 in dB")
    parser.add_argument("--complex-valued", action="store_true")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--lambda-init", type=float, default=None, help="initial Poisson mean per message (if omitted: 1/M — one device across all messages)")
    parser.add_argument("--noise-var-init", type=float, default=None, help="initial noise variance (if omitted: ||y||²/n — pure-noise upper bound)")
    parser.add_argument("--poisson-tail-tol", type=float, default=1e-4, help="truncation tolerance for Poisson count tail mass")
    parser.add_argument("--support-tail-tol", type=float, default=1e-4, help="truncation tolerance for active-support tail mass")
    parser.add_argument("--results-dir", type=str, default="results", help="root directory for saved results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    codebook = make_codebook(args.num_codewords, args.d, rng, complex_valued=args.complex_valued)
    blocks = make_odma_blocks(args.num_blocks, args.n, args.d, rng)
    P_mats = build_pattern_matrices(blocks, args.n)
    msg_to_block, block_to_msg_list = make_message_block_mapping(args.num_codewords, args.num_blocks)
    block_dicts = build_block_dictionaries(codebook, block_to_msg_list, args.num_blocks)

    active_msgs = sample_active_messages(args.num_devices_active, args.num_codewords, rng)
    message_counts = build_message_counts(active_msgs, args.num_codewords)
    block_coeffs = build_block_coefficients(active_msgs, msg_to_block, block_to_msg_list, args.num_blocks)
    noise_var = esn0_db_to_noise_var(args.esn0_db, args.d)

    # y = sum_b P_b C_b^T a_b + z
    y_noisy, y_clean = synthesize_received_signal(P_mats, block_dicts, block_coeffs, noise_var, rng, complex_valued=args.complex_valued)

    coeffs_hat, coeffs_map, meta = graph_based_decoder(
        y_noisy, P_mats, block_dicts,
        max_iter=args.max_iter, damping=args.damping,
        lambda_init=args.lambda_init, noise_var_init=args.noise_var_init,
        poisson_tail_tol=args.poisson_tail_tol, support_tail_tol=args.support_tail_tol,
    )
    counts_soft = assemble_global_counts(coeffs_hat, block_to_msg_list, args.num_codewords)
    counts_map  = assemble_global_counts(coeffs_map,  block_to_msg_list, args.num_codewords)
    metrics = evaluate_counts(message_counts, counts_soft, counts_map)

    # ----------------------------------------------------------------- terminal output
    print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list, active_msgs, message_counts, block_coeffs, y_clean, y_noisy, noise_var, args.esn0_db)
    print(f"Decoder meta: { {k: v for k, v in meta.items() if k != 'history'} }")
    print(f"Decoder eval: {metrics}")
    active_map = {int(m): int(counts_map[m]) for m in np.nonzero(counts_map)[0]}
    print(f"Counts MAP (active): {active_map}")

    # ----------------------------------------------------------------- save results
    slug = make_slug(args)
    out_dir = Path(args.results_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(out_dir, args, meta, metrics, message_counts, counts_soft, counts_map,
                 P_mats, y_clean, y_noisy, noise_var)
    print(f"\nResults saved → {out_dir}/")

    ground_truth = dict(
        codebook=codebook,
        P_mats=P_mats,
        msg_to_block=msg_to_block,
        block_to_msg_list=block_to_msg_list,
        block_dicts=block_dicts,
        active_msgs=active_msgs,
        active_blocks=np.array([msg_to_block[int(m)] for m in active_msgs]),
        message_counts=message_counts,
        block_coeffs=block_coeffs,
        y_clean=y_clean,
        y_noisy=y_noisy,
        noise_var=noise_var,
    )
    return ground_truth


if __name__ == "__main__":
    main()