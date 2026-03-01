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
    noise_var: float,
    *,
    max_iter: int = 50,
    damping: float = 0.3,
    tol: float = 1e-4,
    lambda_init: float | None = None,
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
) -> tuple[dict[int, np.ndarray], dict]:
    """Iterative Gaussian resource update (LMMSE) + exact blockwise discrete Poisson posterior.

    Graph structure:
      - Resource nodes r: scalar observations y[r] = sum_{(b,j): S_b[j]=r} x_{b,j} + z[r]
      - Block nodes b: discrete sparse posterior over a_b, with x_b = C_b^T a_b

    Message schedule:  block->resource (Gaussian) → resource LMMSE → extrinsic resource->block
                       → block discrete posterior → extrinsic block->resource  (repeat)

    All edge messages are Gaussian, parameterised by (mu, var) or equivalently (eta=mu/v, tau=1/v).
    Extrinsic messages are formed by Gaussian division (precision subtraction).
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
            a_mean:  (L_b,)  posterior E[a_b | r_b]
            x_mean:  (d,)    posterior E[x_b | r_b] = C_b^T a_mean
            x_var:   (d,)    posterior marginal variances Var(x_{b,j} | r_b)
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

        A = np.array(states, dtype=np.float64)         # (S, L_b)
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
        return a_mean, x_mean, x_var

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

    # Initial lambda estimate from received energy (matched to model)
    M = float(sum(C_b.shape[0] for C_b in block_dicts.values()))
    if lambda_init is not None:
        lambda_est = float(lambda_init)
    else:
        # E[||y||^2] ≈ lambda * sum_b ||C_b^T 1||^2 * ||P_b||_F^2 + n*sigma^2
        # Simpler: E[||y||^2] ≈ lambda * M + n*noise_var  (rough, unit-norm codewords)
        lambda_est = max((float(np.real(np.vdot(y, y))) - n * noise_var) / M, 1e-6)

    converged = False
    it_used = 0

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

        for b, C_b in block_dicts.items():
            r_b = block_in_mu[b]                                # (d,) pseudo-observation means
            v_b = np.maximum(block_in_var[b], var_floor)        # (d,) pseudo-observation variances

            a_mean, x_mean, x_var = decode_block(C_b, r_b, v_b, lambda_est)
            coeffs_hat[b] = a_mean
            total_mean_count += float(np.sum(a_mean))

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

        # Update Poisson rate: lambda = E[total active] / num_messages
        lambda_est = max(total_mean_count / M, 1e-12)

        if delta < tol:
            converged = True
            break

    return coeffs_hat, {
        "converged": converged,
        "iterations": it_used,
        "tol": tol,
        "damping": damping,
        "lambda_est": lambda_est,
        "lambda_init": lambda_init,
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


def evaluate_counts(counts_true: np.ndarray, counts_hat: np.ndarray, threshold: float = 0.5) -> dict:
    """Compare true vs estimated global message count vectors.

    Returns soft metrics (l1, mse) and hard detection metrics after rounding.
    """
    diff = counts_true - counts_hat
    counts_hard = np.round(counts_hat).astype(int)
    counts_hard = np.maximum(counts_hard, 0)

    supp_true = counts_true > 0
    supp_hard = counts_hard > 0
    tp = int(np.sum(supp_true & supp_hard))
    fp = int(np.sum(~supp_true & supp_hard))
    fn = int(np.sum(supp_true & ~supp_hard))

    return dict(
        l1=float(np.sum(np.abs(diff))),
        mse=float(np.mean(diff ** 2)),
        support_true=int(np.sum(supp_true)),
        support_hard=int(np.sum(supp_hard)),
        tp=tp, fp=fp, fn=fn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        count_errors=int(np.sum(counts_hard[supp_true] != counts_true[supp_true])),
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
    parser.add_argument("--lambda-init", type=float, default=None, help="initial Poisson mean per message (if omitted: energy-based estimate)")
    parser.add_argument("--poisson-tail-tol", type=float, default=1e-4, help="truncation tolerance for Poisson count tail mass")
    parser.add_argument("--support-tail-tol", type=float, default=1e-4, help="truncation tolerance for active-support tail mass")
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

    coeffs_hat, meta = graph_based_decoder(
        y_noisy, P_mats, block_dicts, noise_var,
        max_iter=args.max_iter, damping=args.damping,
        lambda_init=args.lambda_init, poisson_tail_tol=args.poisson_tail_tol, support_tail_tol=args.support_tail_tol,
    )
    counts_hat = assemble_global_counts(coeffs_hat, block_to_msg_list, args.num_codewords)
    metrics = evaluate_counts(message_counts, counts_hat)

    print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list, active_msgs, message_counts, block_coeffs, y_clean, y_noisy, noise_var, args.esn0_db)
    print(f"Decoder meta: {meta}")
    print(f"Decoder eval: {metrics}")

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