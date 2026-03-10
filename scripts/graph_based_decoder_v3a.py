"""
ODMA + URA Decoder Testbed — V3a: first-antenna count + support-coupled nuisance
===============================================================================
V3a keeps the V2 outer loop (resource Gaussian update + block discrete posterior)
and adds fading-aware support evidence from antennas 1..M-1.

Signal model after transmitter pre-equalisation:
    Y = sum_u (P_{b_u} c_{m_u}) g_u^T + Z,   g_u = h_u / h_u[0],   g_u[0] = 1

Decoder model:
  - antenna 0 is count-bearing (same role as V2),
  - antennas 1..M-1 are nuisance-valued but share support with antenna 0.
"""

from __future__ import annotations
import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path
import numpy as np


def make_codebook(num_codewords: int, d: int, rng: np.random.Generator, complex_valued: bool = False) -> np.ndarray:
    """Random Gaussian codebook with unit-normalized rows. Returns (num_codewords, d)."""
    if complex_valued:
        raw = rng.standard_normal((num_codewords, d)) + 1j * rng.standard_normal((num_codewords, d))
    else:
        raw = rng.standard_normal((num_codewords, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_odma_blocks(num_blocks: int, n: int, d: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random ODMA blocks. Each block is sorted d-subsample of {0,...,n-1}."""
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(num_blocks)]


def build_pattern_matrices(blocks: list[np.ndarray], n: int) -> dict[int, np.ndarray]:
    """Build P_b ∈ {0,1}^{n×d} with P_b^T P_b = I_d."""
    P_mats: dict[int, np.ndarray] = {}
    for b, S_b in enumerate(blocks):
        d = len(S_b)
        P = np.zeros((n, d), dtype=np.float64)
        P[S_b, np.arange(d)] = 1.0
        P_mats[b] = P
    return P_mats


def make_message_block_mapping(num_codewords: int, num_blocks: int):
    """Deterministic mapping: m -> block m % num_blocks."""
    msg_to_block: dict[int, int] = {}
    block_to_msg_list: dict[int, list[int]] = {b: [] for b in range(num_blocks)}
    for m in range(num_codewords):
        b = m % num_blocks
        msg_to_block[m] = b
        block_to_msg_list[b].append(m)
    return msg_to_block, block_to_msg_list


def sample_active_messages(num_devices_active: int, num_codewords: int, rng: np.random.Generator) -> np.ndarray:
    """Each active device independently picks one message. Duplicates allowed."""
    return rng.integers(0, num_codewords, size=num_devices_active)


def build_message_counts(active_msgs: np.ndarray, num_codewords: int) -> np.ndarray:
    """Global message-count target."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for m in active_msgs:
        counts[int(m)] += 1.0
    return counts


def build_block_coefficients(active_msgs: np.ndarray, msg_to_block: dict[int, int], block_to_msg_list: dict[int, list[int]], num_blocks: int) -> dict[int, np.ndarray]:
    """Blockwise sparse multiplicity vectors a_b."""
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
    """Per-block dictionary C_b with shape (L_b, d)."""
    return {b: codebook[block_to_msg_list[b]] for b in range(num_blocks)}


def make_user_channels(num_users: int, num_antennas: int, rng: np.random.Generator, complex_valued: bool = False) -> np.ndarray:
    """Raw per-user channel before inversion."""
    if complex_valued:
        return (rng.standard_normal((num_users, num_antennas)) + 1j * rng.standard_normal((num_users, num_antennas))) / np.sqrt(2)
    return rng.standard_normal((num_users, num_antennas))


def apply_channel_inversion(channels_raw: np.ndarray, min_first_ant_power: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Apply g_u = h_u / h_u[0], with deep-fade exclusion on antenna 0."""
    first_ant_power = np.abs(channels_raw[:, 0]) ** 2
    survive_mask = first_ant_power >= min_first_ant_power
    ch_survive = channels_raw[survive_mask]
    channels_eff = ch_survive / ch_survive[:, 0:1]
    return channels_eff, survive_mask


def esn0_db_to_noise_var(esn0_db: float, d: int) -> float:
    """Es/N0 (dB) to per-entry AWGN variance."""
    esn0_lin = 10.0 ** (esn0_db / 10.0)
    return 1.0 / (d * esn0_lin)


def synthesize_received_signal(active_msgs: np.ndarray, msg_to_block: dict[int, int], P_mats: dict[int, np.ndarray], codebook: np.ndarray, channels_eff: np.ndarray, noise_var: float, rng: np.random.Generator, complex_valued: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Y = sum_u (P_{b_u} c_{m_u}) g_u^T + Z."""
    n = next(iter(P_mats.values())).shape[0]
    num_antennas = channels_eff.shape[1]
    dtype = np.complex128 if complex_valued else np.float64
    Y_clean = np.zeros((n, num_antennas), dtype=dtype)

    for u in range(len(active_msgs)):
        m_u = int(active_msgs[u])
        P_u = P_mats[msg_to_block[m_u]]
        c_u = codebook[m_u]
        g_u = channels_eff[u]
        s_u = P_u @ c_u
        Y_clean += np.outer(s_u, g_u)

    if complex_valued:
        noise = np.sqrt(noise_var / 2) * (rng.standard_normal((n, num_antennas)) + 1j * rng.standard_normal((n, num_antennas)))
    else:
        noise = np.sqrt(noise_var) * rng.standard_normal((n, num_antennas))
    return Y_clean + noise, Y_clean


def graph_based_decoder_v3a(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    max_iter: int = 50,
    damping: float = 0.3,
    tol: float = 1e-4,
    lambda_init: float | None = None,
    noise_var_init: float | None = None,
    nuisance_var_init: float | None = None,
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
    support_weight: float = 1.0,
    max_support_size: int | None = None,
    max_count_value: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """V3a decoder: V2 scalar resource update + support-coupled block posterior."""
    from itertools import combinations, product

    n, num_antennas = Y.shape
    dtype = Y.dtype
    y0 = Y[:, 0]
    is_complex_model = np.iscomplexobj(Y)
    nuisance_ll_scale = 1.0 if is_complex_model else 0.5
    var_floor = 1e-10
    tau_floor = 1e-10

    def logsumexp(v: np.ndarray) -> float:
        vmax = float(np.max(v))
        return vmax + float(np.log(np.sum(np.exp(v - vmax)))) if np.isfinite(vmax) else vmax

    def poisson_pmf_vec(lam: float) -> np.ndarray:
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

    count_combo_cache: dict[tuple[int, int], np.ndarray] = {}
    state_struct_cache: dict[tuple[int, int, int], dict] = {}

    def get_count_combos(k: int, c_max: int) -> np.ndarray:
        key = (k, c_max)
        cached = count_combo_cache.get(key)
        if cached is not None:
            return cached
        arr = np.array(list(product(range(1, c_max + 1), repeat=k)), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        count_combo_cache[key] = arr
        return arr

    def get_state_struct(L_b: int, c_max: int, k_max: int) -> dict:
        key = (L_b, c_max, k_max)
        cached = state_struct_cache.get(key)
        if cached is not None:
            return cached

        rows: list[np.ndarray] = [np.zeros((1, L_b), dtype=np.float64)]
        support_keys: list[tuple[int, ...]] = [tuple()]
        state_support_ids: list[int] = [0]

        sid = 1
        for k in range(1, k_max + 1):
            cnt_arr = get_count_combos(k, c_max)  # (c_max^k, k)
            ncnt = cnt_arr.shape[0]
            for idxs in combinations(range(L_b), k):
                blk = np.zeros((ncnt, L_b), dtype=np.float64)
                blk[:, list(idxs)] = cnt_arr
                rows.append(blk)
                support_keys.append(idxs)
                state_support_ids.extend([sid] * ncnt)
                sid += 1

        A = np.vstack(rows)
        struct = {
            "A": A,
            "A_int": A.astype(np.int32, copy=False),
            "support_keys": support_keys,
            "state_support_ids": np.array(state_support_ids, dtype=np.int64),
        }
        state_struct_cache[key] = struct
        return struct

    def decode_block(
        C_b: np.ndarray,      # (L_b, d)
        r0_b: np.ndarray,     # (d,)
        v0_b: np.ndarray,     # (d,)
        y_nuis_b: np.ndarray, # (d, M-1)
        lam: float,
        sigma2: float,
        nu_var: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact Poisson enumeration; nuisance evidence is support-only."""
        pmf = poisson_pmf_vec(lam)
        if max_count_value is not None:
            c_keep = max(int(max_count_value), 1)
            pmf = pmf[: min(len(pmf), c_keep + 1)]
            pmf = pmf / np.sum(pmf)
        c_max = len(pmf) - 1
        L_b = C_b.shape[0]
        p_nz = float(1.0 - pmf[0])
        k_max = max_active_per_block(L_b, p_nz)
        if max_support_size is not None:
            k_max = min(k_max, max(int(max_support_size), 0))
        log_pmf = np.log(pmf + 1e-300)

        struct = get_state_struct(L_b, c_max, k_max)
        A = struct["A"]
        A_int = struct["A_int"]
        support_keys = struct["support_keys"]
        state_support_ids = struct["state_support_ids"]
        log_prior = np.sum(log_pmf[A_int], axis=1)
        X = A @ C_b

        err = X - r0_b[None, :]
        inv_v0 = 1.0 / v0_b
        ll_count = -np.real(np.sum((np.abs(err) ** 2) * inv_v0[None, :], axis=1))

        ll_nuis = np.zeros(A.shape[0], dtype=np.float64)
        if y_nuis_b.shape[1] > 0:
            # Nuisance evidence is support-coupled in v3a, so evaluate it once
            # per unique support and reuse for all count states on that support.
            Ct = C_b.conj().T
            d = C_b.shape[1]
            sigma2_safe = max(float(sigma2), var_floor)
            nu_safe = max(float(nu_var), 0.0)
            y_energy = float(np.real(np.vdot(y_nuis_b, y_nuis_b)))
            base_log_det = d * float(np.log(max(sigma2_safe, 1e-300)))
            ll_nuis_support = np.zeros(len(support_keys), dtype=np.float64)

            for sid, idxs in enumerate(support_keys):
                k = len(idxs)
                if k == 0 or nu_safe == 0.0:
                    quad = y_energy / sigma2_safe
                    log_det = base_log_det
                else:
                    Cs = Ct[:, idxs]  # (d, k)
                    gram_small = Cs.conj().T @ Cs
                    B = np.eye(k, dtype=gram_small.dtype) + (nu_safe / sigma2_safe) * gram_small
                    B = 0.5 * (B + B.conj().T)
                    try:
                        Lb = np.linalg.cholesky(B)
                    except np.linalg.LinAlgError:
                        Lb = np.linalg.cholesky(B + 1e-10 * np.eye(k, dtype=B.dtype))
                    proj = Cs.conj().T @ y_nuis_b
                    z = np.linalg.solve(Lb, proj)
                    corr = (nu_safe / (sigma2_safe * sigma2_safe)) * float(np.real(np.vdot(z, z)))
                    quad = max(y_energy / sigma2_safe - corr, 0.0)
                    log_det = base_log_det + 2.0 * float(np.sum(np.log(np.maximum(np.abs(np.diag(Lb)), 1e-300))))
                ll_nuis_support[sid] = -support_weight * nuisance_ll_scale * (quad + y_nuis_b.shape[1] * log_det)

            ll_nuis = ll_nuis_support[state_support_ids]

        log_post = ll_count + ll_nuis + log_prior
        log_post -= logsumexp(log_post)
        w = np.exp(log_post)

        a_mean = w @ A
        x_mean = w @ X
        x_var = np.maximum(np.real(w @ (np.abs(X) ** 2)) - np.abs(x_mean) ** 2, var_floor)
        a_map = A[int(np.argmax(log_post))]
        return a_mean, x_mean, x_var, a_map

    block_supports = {b: np.argmax(P_mats[b], axis=0).astype(int) for b in P_mats}
    resource_to_edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b, S_b in block_supports.items():
        for j, r in enumerate(S_b):
            resource_to_edges[r].append((b, j))

    block_out_mu = {b: np.zeros(C_b.shape[1], dtype=dtype) for b, C_b in block_dicts.items()}
    block_out_var = {b: np.ones(C_b.shape[1], dtype=np.float64) for b, C_b in block_dicts.items()}
    block_in_mu = {b: np.zeros(C_b.shape[1], dtype=dtype) for b, C_b in block_dicts.items()}
    block_in_var = {b: np.ones(C_b.shape[1], dtype=np.float64) for b, C_b in block_dicts.items()}

    coeffs_hat = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}
    coeffs_map = {b: np.zeros(C_b.shape[0], dtype=np.float64) for b, C_b in block_dicts.items()}

    total_msgs = float(sum(C_b.shape[0] for C_b in block_dicts.values()))
    lambda_est = float(lambda_init) if lambda_init is not None else 1.0 / total_msgs
    noise_var = float(noise_var_init) if noise_var_init is not None else float(np.mean(np.abs(y0) ** 2))

    if nuisance_var_init is not None:
        nuisance_var = float(nuisance_var_init)
    elif num_antennas > 1:
        obs_nuis = float(np.mean(np.abs(Y[:, 1:]) ** 2))
        nuisance_var = max((obs_nuis - noise_var) / max(float(np.mean(np.abs(y0) ** 2)), 1e-12), 1e-6)
    else:
        nuisance_var = 0.0

    converged = False
    it_used = 0
    history: list[dict] = []

    for it in range(1, max_iter + 1):
        it_used = it

        # Step A+B: scalar resource update on antenna 0
        for r in range(n):
            edges = resource_to_edges[r]
            if not edges:
                continue

            mu_in = np.array([block_out_mu[b][j] for b, j in edges], dtype=dtype)
            var_in = np.maximum([block_out_var[b][j] for b, j in edges], var_floor)

            v_sum = float(np.sum(var_in))
            mu_s = np.sum(mu_in)
            denom = noise_var + v_sum
            innov = y0[r] - mu_s

            hat_mu = mu_in + (var_in * innov) / denom
            hat_var = var_in - (var_in ** 2) / denom

            denom_ext = denom - var_in
            tau_ext = np.maximum(1.0 / denom_ext, tau_floor)
            eta_ext = hat_mu / hat_var - mu_in / var_in

            for idx, (b, j) in enumerate(edges):
                block_in_mu[b][j] = eta_ext[idx] / tau_ext[idx]
                block_in_var[b][j] = 1.0 / tau_ext[idx]

        # Step C+D+E: block posterior + extrinsic back to resources
        delta = 0.0
        total_mean_count = 0.0
        total_x_var_post = 0.0

        for b, C_b in block_dicts.items():
            S_b = block_supports[b]
            r0_b = block_in_mu[b]
            v0_b = np.maximum(block_in_var[b], var_floor)
            y_nuis_b = Y[S_b, 1:] if num_antennas > 1 else np.zeros((len(S_b), 0), dtype=dtype)

            a_mean, x_mean, x_var, a_map = decode_block(C_b, r0_b, v0_b, y_nuis_b, lambda_est, noise_var, nuisance_var)
            coeffs_hat[b] = a_mean
            coeffs_map[b] = a_map
            total_mean_count += float(np.sum(a_mean))
            total_x_var_post += float(np.sum(x_var))

            tau_post = np.maximum(1.0 / x_var, tau_floor)
            tau_in = np.maximum(1.0 / v0_b, tau_floor)
            tau_ext = np.maximum(tau_post - tau_in, tau_floor)
            eta_ext = x_mean * tau_post - r0_b * tau_in

            tau_old = np.maximum(1.0 / np.maximum(block_out_var[b], var_floor), tau_floor)
            eta_old = block_out_mu[b] * tau_old
            tau_damp = (1.0 - damping) * tau_ext + damping * tau_old
            eta_damp = (1.0 - damping) * eta_ext + damping * eta_old

            mu_new = eta_damp / tau_damp
            var_new = np.maximum(1.0 / tau_damp, var_floor)

            delta = max(delta, float(np.max(np.abs(mu_new - block_out_mu[b]))))
            block_out_mu[b] = mu_new
            block_out_var[b] = var_new

        lambda_est = max(total_mean_count / total_msgs, 1e-12)

        y_hat = np.zeros(n, dtype=dtype)
        for b, C_b in block_dicts.items():
            y_hat[block_supports[b]] += C_b.T @ coeffs_hat[b]
        resid0 = y0 - y_hat
        resid_energy = float(np.real(np.vdot(resid0, resid0)))
        noise_var = max((resid_energy + total_x_var_post) / n, var_floor)

        if num_antennas > 1:
            signal_power = float(np.mean(np.abs(y_hat) ** 2))
            obs_nuis = float(np.mean(np.abs(Y[:, 1:]) ** 2))
            nuisance_var = float(np.clip((obs_nuis - noise_var) / max(signal_power, 1e-12), 1e-6, 1e6))
        else:
            nuisance_var = 0.0

        history.append({
            "delta": delta,
            "lambda": lambda_est,
            "noise_var": noise_var,
            "nuisance_var": nuisance_var,
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
        "nuisance_var_est": nuisance_var,
        "lambda_init": lambda_init,
        "noise_var_init": noise_var_init,
        "nuisance_var_init": nuisance_var_init,
        "poisson_tail_tol": poisson_tail_tol,
        "support_tail_tol": support_tail_tol,
        "support_weight": support_weight,
        "nuisance_ll_scale": nuisance_ll_scale,
        "max_support_size": max_support_size,
        "max_count_value": max_count_value,
    }


def assemble_global_counts(block_coeffs: dict[int, np.ndarray], block_to_msg_list: dict[int, list[int]], num_codewords: int) -> np.ndarray:
    """Convert blockwise estimates to global message-count vector."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, a_b in block_coeffs.items():
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_b[local_idx]
    return counts


def evaluate_counts(counts_true: np.ndarray, counts_soft: np.ndarray, counts_hard: np.ndarray) -> dict:
    """Soft + hard count metrics."""
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


def print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list, active_msgs, channels_raw, channels_eff, num_sampled, message_counts, block_coeffs, Y_clean, Y_noisy, noise_var, esn0_db, min_first_ant_power):
    num_blocks = len(P_mats)
    n, d = next(iter(P_mats.values())).shape
    num_antennas = Y_noisy.shape[1]
    K = len(active_msgs)

    print("\n" + "=" * 60)
    print("ODMA + URA Simulation — V3a (first-antenna inversion)")
    print("=" * 60)
    print(f"\nCodebook          : {codebook.shape}  (num_codewords x d)")
    print(f"Resource grid     : n = {n}")
    print(f"Block size        : d = {d}")
    print(f"Num blocks        : {num_blocks}")
    print(f"Pattern matrices  : {num_blocks} x ({n}, {d})")
    print(f"Num antennas      : M = {num_antennas}")
    print(f"Sampled devices   : {num_sampled}")
    print(f"Surviving devices : K = {K}  (dropped {num_sampled - K}, threshold |h[0]|^2 >= {min_first_ant_power})")
    print(f"Es/N0             : {esn0_db:.1f} dB")
    print(f"Noise variance    : sigma^2 = {noise_var:.6f}")
    print(f"g_u[0] all ones   : {np.allclose(channels_eff[:, 0], 1.0)}")

    print(f"\nActive messages (per user) : {active_msgs.tolist()}")
    active_blocks = [msg_to_block[int(m)] for m in active_msgs]
    print(f"Active blocks (per user)  : {active_blocks}")

    num_unique = int(np.count_nonzero(message_counts))
    max_mult = int(message_counts.max()) if num_unique > 0 else 0
    print("\nDecoder target (global message counts):")
    print(f"  Unique messages    : {num_unique} / {len(message_counts)}")
    print(f"  Max multiplicity   : {max_mult}")
    active_indices = np.nonzero(message_counts)[0]
    print(f"  Active msg -> count: {dict(zip(active_indices.tolist(), message_counts[active_indices].astype(int).tolist()))}")

    print("\nPer-block coefficient summary:")
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

    print("\nBlock overlap statistics:")
    print(f"  Pairs with overlap : {len(overlap_counts)} / {num_blocks * (num_blocks - 1) // 2}")
    if overlap_counts:
        ovs = [c for _, _, c in overlap_counts]
        print(f"  Overlap sizes      : min={min(ovs)}, max={max(ovs)}, mean={np.mean(ovs):.1f}")
    print(f"  Max blocks/resource: {resource_usage.max()}")
    print(f"  Resources unused   : {int((resource_usage == 0).sum())} / {n}")

    print("\nReceived signal:")
    print(f"  Y shape           : {Y_noisy.shape}")
    print(f"  ||Y_clean||_F     : {np.linalg.norm(Y_clean):.4f}")
    print(f"  ||Y_noisy||_F     : {np.linalg.norm(Y_noisy):.4f}")
    print(f"  ||noise||_F       : {np.linalg.norm(Y_noisy - Y_clean):.4f}")

    if len(channels_raw) > 0:
        first_power = np.abs(channels_raw[:, 0]) ** 2
        ratio_energy = np.mean(np.abs(channels_eff[:, 1:]) ** 2) if channels_eff.shape[1] > 1 else 0.0
        print("\nEffective channel stats:")
        print(f"  mean |h_raw[0]|^2 : {float(np.mean(first_power)):.4f}")
        print(f"  mean |g[:,1:]|^2  : {float(ratio_energy):.4f}")
    print("=" * 60 + "\n")


def make_slug(args) -> str:
    """Short run identifier."""
    cx = "cx" if args.complex_valued else "re"
    return (
        f"v3a_{cx}_n{args.n}_d{args.d}_B{args.num_blocks}"
        f"_M{args.num_codewords}_K{args.num_devices_active}"
        f"_ant{args.num_antennas}_snr{args.esn0_db:.0f}dB_s{args.seed}"
    )


def save_results(out_dir: Path, args, meta: dict, metrics: dict, message_counts: np.ndarray, counts_soft: np.ndarray, counts_map: np.ndarray, P_mats: dict, y_clean: np.ndarray, y_noisy: np.ndarray, noise_var_true: float, channels_eff: np.ndarray) -> None:
    """Save markdown summary + plots to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = meta.get("history", [])
    iters = list(range(1, len(history) + 1))
    n_resources = y_clean.shape[0]
    num_blocks = len(P_mats)
    num_antennas = y_noisy.shape[1]
    k_effective = float(np.sum(message_counts))

    C = {"blue": "#4C78C8", "orange": "#E07B2A", "green": "#3BAA5C",
         "red": "#C84C4C", "grey": "#888888", "purple": "#8B5CF6", "teal": "#2A9D8F"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.suptitle("Convergence - per-iteration diagnostics", fontsize=11, y=1.01)

    ax = axes[0]
    ax.semilogy(iters, [h["delta"] for h in history], color=C["blue"], lw=2, marker="o", ms=4)
    ax.axhline(meta["tol"], color=C["red"], lw=1, ls="--", label=f"tol={meta['tol']:.0e}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Max message delta (log)")
    ax.set_title("Message convergence"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, [h["lambda"] for h in history], color=C["orange"], lw=2, marker="o", ms=4, label="lambda_est")
    ax.axhline(k_effective / args.num_codewords, color=C["grey"], lw=1.5, ls="--", label="lambda_true=K_eff/M")
    ax.set_xlabel("Iteration"); ax.set_ylabel("lambda")
    ax.set_title("Poisson rate"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(iters, [h["noise_var"] for h in history], color=C["purple"], lw=2, marker="o", ms=4, label="sigma2_est")
    ax.axhline(noise_var_true, color=C["grey"], lw=1.5, ls="--", label=f"sigma2_true={noise_var_true:.4f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("sigma2")
    ax.set_title("Noise variance"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(iters, [h["nuisance_var"] for h in history], color=C["teal"], lw=2, marker="o", ms=4, label="nu_var_est")
    ax.set_xlabel("Iteration"); ax.set_ylabel("nuisance scale")
    ax.set_title("Nuisance-column scale"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    active_idx = np.nonzero(message_counts)[0]
    n_active = len(active_idx)
    x = np.arange(n_active)
    w = 0.28
    fig, ax = plt.subplots(figsize=(max(8, n_active * 0.7 + 2), 4))
    ax.bar(x - w, message_counts[active_idx], w, label="True", color=C["blue"], alpha=0.85)
    ax.bar(x, counts_soft[active_idx], w, label="MMSE soft", color=C["orange"], alpha=0.85)
    ax.bar(x + w, counts_map[active_idx], w, label="MAP hard", color=C["green"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in active_idx], fontsize=8)
    ax.set_xlabel("Message index"); ax.set_ylabel("Count")
    ax.set_title("True vs estimated counts (active messages only)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_dir / "count_estimates.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    resource_usage = np.zeros(n_resources, dtype=int)
    pattern_mat = np.zeros((num_blocks, n_resources), dtype=np.float32)
    for b, P in P_mats.items():
        mask = np.any(P, axis=1)
        pattern_mat[b] = mask.astype(float)
        resource_usage += mask.astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), layout="constrained", gridspec_kw={"height_ratios": [num_blocks, 1]})
    axes[0].imshow(pattern_mat, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_ylabel("Block"); axes[0].set_title("ODMA pattern matrix")
    axes[0].set_yticks(range(num_blocks))
    axes[1].bar(range(n_resources), resource_usage, color=C["blue"], alpha=0.7, width=1.0)
    axes[1].set_xlim(-0.5, n_resources - 0.5)
    axes[1].set_ylabel("# blocks"); axes[1].set_xlabel("Resource index")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "odma_patterns.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    y_sc = np.real(y_clean[:, 0])
    y0 = np.real(y_noisy[:, 0])
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, layout="constrained")
    r_idx = np.arange(n_resources)
    n_show = min(num_antennas, 6)
    for m in range(n_show):
        axes[0].plot(r_idx, np.real(y_noisy[:, m]), color=C["orange"], lw=0.4, alpha=0.2)
    axes[0].plot(r_idx, y_sc, color=C["blue"], lw=1.3, label="Y_clean[:,0] (Re)")
    axes[0].plot(r_idx, y0, color=C["orange"], lw=1.0, label="Y_noisy[:,0] (Re)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Received signal (faint: per-antenna traces)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(r_idx, y0 - y_sc, color=C["grey"], lw=0.8, alpha=0.9, label="Antenna-0 noise (Re)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Resource index"); axes[1].set_ylabel("Noise")
    axes[1].set_title(
        f"sigma2_true={noise_var_true:.4f}, sigma2_est={meta['noise_var_est']:.4f}, "
        f"nuisance_est={meta['nuisance_var_est']:.3f}"
    )
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.savefig(out_dir / "received_signal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if channels_eff.shape[1] > 1 and channels_eff.shape[0] > 0:
        ratio_pwr = np.abs(channels_eff[:, 1:]).ravel() ** 2
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ratio_pwr, bins=max(10, len(ratio_pwr) // 4), color=C["teal"], alpha=0.75, edgecolor="white")
        ax.set_xlabel("|g_u[m]|^2, m>=1"); ax.set_ylabel("Count")
        ax.set_title("Effective nuisance-channel energy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "nuisance_channel_hist.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    prec, rec = metrics["precision"], metrics["recall"]
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    active_map_dict = {int(m): int(counts_map[m]) for m in np.nonzero(counts_map)[0]}
    active_true_dict = {int(m): int(message_counts[m]) for m in np.nonzero(message_counts)[0]}

    md = textwrap.dedent(f"""
    # ODMA + URA V3a - Run Results
    **Slug:** `{out_dir.name}`
    **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## Setup
    | Parameter | Value |
    |-----------|-------|
    | Resource grid n | {args.n} |
    | Codeword length d | {args.d} |
    | Num blocks B | {args.num_blocks} |
    | Num codewords M | {args.num_codewords} |
    | Active devices sampled | {args.num_devices_active} |
    | Active devices survived | {int(k_effective)} |
    | Es/N0 | {args.esn0_db:.1f} dB |
    | Antennas | {args.num_antennas} |
    | Deep-fade threshold | {args.min_first_ant_power} |
    | Complex-valued | {args.complex_valued} |
    | Seed | {args.seed} |
    | Max iterations | {args.max_iter} |
    | Damping | {meta['damping']} |
    | Support weight | {meta['support_weight']} |
    | Nuisance ll scale | {meta['nuisance_ll_scale']} |
    | Max support size cap | {meta['max_support_size']} |
    | Max count value cap | {meta['max_count_value']} |

    ## Decoder Convergence
    | | Value |
    |--|--|
    | Converged | {meta['converged']} |
    | Iterations used | {meta['iterations']} / {args.max_iter} |
    | lambda_true = K_eff/M | {k_effective / args.num_codewords:.4f} |
    | lambda_est (final) | {meta['lambda_est']:.4f} |
    | sigma2_true | {noise_var_true:.6f} |
    | sigma2_est (final) | {meta['noise_var_est']:.6f} |
    | nuisance_var_est (final) | {meta['nuisance_var_est']:.4f} |

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
    - `convergence.png` - delta, lambda, sigma2, nuisance scale
    - `count_estimates.png` - true vs MMSE vs MAP
    - `odma_patterns.png` - ODMA usage map
    - `received_signal.png` - antenna-0 clean/noisy traces
    - `nuisance_channel_hist.png` - histogram of |g_u[m]|^2 for m>=1
    """).strip()
    (out_dir / "results.md").write_text(md)

    raw = {
        "args": vars(args),
        "meta": {k: v for k, v in meta.items() if k != "history"},
        "metrics": metrics,
        "noise_var_true": noise_var_true,
        "history": history,
    }
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2))


def main():
    parser = argparse.ArgumentParser(description="ODMA + URA decoder - V3a (pre-equalised fading)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=128, help="resource grid size")
    parser.add_argument("--d", type=int, default=16, help="codeword length / block size")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-codewords", type=int, default=64)
    parser.add_argument("--num-devices-active", type=int, default=10)
    parser.add_argument("--num-antennas", type=int, default=4)
    parser.add_argument("--esn0-db", type=float, default=10.0)
    parser.add_argument("--min-first-ant-power", type=float, default=0.0, help="drop users with |h[0]|^2 below threshold")
    parser.add_argument("--complex-valued", action="store_true")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--lambda-init", type=float, default=None)
    parser.add_argument("--noise-var-init", type=float, default=None)
    parser.add_argument("--nuisance-var-init", type=float, default=None)
    parser.add_argument("--support-weight", type=float, default=1.0, help="weight of nuisance-column support evidence")
    parser.add_argument("--poisson-tail-tol", type=float, default=1e-4)
    parser.add_argument("--support-tail-tol", type=float, default=1e-4)
    parser.add_argument("--max-support-size", type=int, default=None, help="hard cap on active local messages per block")
    parser.add_argument("--max-count-value", type=int, default=None, help="hard cap on per-message multiplicity")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    codebook = make_codebook(args.num_codewords, args.d, rng, complex_valued=args.complex_valued)
    blocks = make_odma_blocks(args.num_blocks, args.n, args.d, rng)
    P_mats = build_pattern_matrices(blocks, args.n)
    msg_to_block, block_to_msg_list = make_message_block_mapping(args.num_codewords, args.num_blocks)
    block_dicts = build_block_dictionaries(codebook, block_to_msg_list, args.num_blocks)

    num_sampled = args.num_devices_active
    active_msgs_all = sample_active_messages(num_sampled, args.num_codewords, rng)
    channels_raw_all = make_user_channels(num_sampled, args.num_antennas, rng, complex_valued=args.complex_valued)
    channels_eff_all, survive_mask = apply_channel_inversion(channels_raw_all, args.min_first_ant_power)

    active_msgs = active_msgs_all[survive_mask]
    channels_raw = channels_raw_all[survive_mask]
    channels_eff = channels_eff_all

    message_counts = build_message_counts(active_msgs, args.num_codewords)
    block_coeffs = build_block_coefficients(active_msgs, msg_to_block, block_to_msg_list, args.num_blocks)
    noise_var = esn0_db_to_noise_var(args.esn0_db, args.d)

    Y_noisy, Y_clean = synthesize_received_signal(
        active_msgs, msg_to_block, P_mats, codebook, channels_eff, noise_var, rng, complex_valued=args.complex_valued
    )

    coeffs_hat, coeffs_map, meta = graph_based_decoder_v3a(
        Y_noisy, P_mats, block_dicts,
        max_iter=args.max_iter, damping=args.damping,
        lambda_init=args.lambda_init, noise_var_init=args.noise_var_init, nuisance_var_init=args.nuisance_var_init,
        poisson_tail_tol=args.poisson_tail_tol, support_tail_tol=args.support_tail_tol,
        support_weight=args.support_weight,
        max_support_size=args.max_support_size,
        max_count_value=args.max_count_value,
    )
    counts_soft = assemble_global_counts(coeffs_hat, block_to_msg_list, args.num_codewords)
    counts_map = assemble_global_counts(coeffs_map, block_to_msg_list, args.num_codewords)
    metrics = evaluate_counts(message_counts, counts_soft, counts_map)

    print_diagnostics(
        codebook, P_mats, msg_to_block, block_to_msg_list,
        active_msgs, channels_raw, channels_eff, num_sampled,
        message_counts, block_coeffs, Y_clean, Y_noisy, noise_var,
        args.esn0_db, args.min_first_ant_power,
    )
    print(f"Decoder meta: { {k: v for k, v in meta.items() if k != 'history'} }")
    print(f"Decoder eval: {metrics}")
    active_map = {int(m): int(counts_map[m]) for m in np.nonzero(counts_map)[0]}
    print(f"Counts MAP (active): {active_map}")

    slug = make_slug(args)
    out_dir = Path(args.results_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(out_dir, args, meta, metrics, message_counts, counts_soft, counts_map, P_mats, Y_clean, Y_noisy, noise_var, channels_eff)
    print(f"\nResults saved -> {out_dir}/")

    ground_truth = dict(
        codebook=codebook,
        P_mats=P_mats,
        msg_to_block=msg_to_block,
        block_to_msg_list=block_to_msg_list,
        block_dicts=block_dicts,
        active_msgs=active_msgs,
        active_blocks=np.array([msg_to_block[int(m)] for m in active_msgs]),
        channels_raw=channels_raw,
        channels_eff=channels_eff,
        num_sampled=num_sampled,
        num_dropped=int(num_sampled - survive_mask.sum()),
        message_counts=message_counts,
        block_coeffs=block_coeffs,
        Y_clean=Y_clean,
        Y_noisy=Y_noisy,
        noise_var=noise_var,
    )
    return ground_truth


if __name__ == "__main__":
    main()