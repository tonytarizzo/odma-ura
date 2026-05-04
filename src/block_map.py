"""Block-MAP scoring with precomputed enumeration cache (Poisson count prior)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product as iproduct

import numpy as np
from scipy.special import gammaln


@dataclass
class BlockStateCache:
    A: np.ndarray
    X: np.ndarray
    count_sum: np.ndarray
    log_factorial_sum: np.ndarray


def build_block_state_cache(C_b: np.ndarray, lam_design: float = 0.4,
                            poisson_tail_tol: float = 1e-4,
                            support_tail_tol: float = 1e-4,
                            c_max_cap: int = 4,
                            k_max_cap: int = 6) -> BlockStateCache:
    """Enumerate count vectors with conservative tail bounds + hard caps."""
    L_b = C_b.shape[0]
    lam_d = float(np.clip(lam_design, 1e-12, 1.0 - 1e-12))

    probs = [np.exp(-lam_d)]
    total = probs[0]
    c = 0
    while 1.0 - total > poisson_tail_tol and probs[-1] > 0.0 and c < c_max_cap:
        c += 1
        probs.append(probs[-1] * lam_d / c)
        total += probs[-1]
    c_max = max(1, min(c_max_cap, len(probs) - 1))

    p_nz = float(np.clip(1.0 - np.exp(-lam_d), 1e-12, 1.0 - 1e-12))
    p0_b = 1.0 - p_nz
    pk = p0_b ** L_b
    cdf_b = pk
    k_max = L_b
    for k in range(L_b):
        pk = pk * ((L_b - k) / (k + 1)) * (p_nz / p0_b)
        cdf_b += pk
        if 1.0 - cdf_b <= support_tail_tol:
            k_max = k + 1
            break
    k_max = min(k_max, k_max_cap, L_b)

    states: list[np.ndarray] = [np.zeros(L_b, dtype=np.float64)]
    for k in range(1, k_max + 1):
        for idxs in combinations(range(L_b), k):
            for cnts in iproduct(range(1, c_max + 1), repeat=k):
                a = np.zeros(L_b, dtype=np.float64)
                a[list(idxs)] = cnts
                states.append(a)

    A = np.asarray(states, dtype=np.float64)
    X = A @ C_b
    count_sum = A.sum(axis=1)
    log_fact = gammaln(A + 1.0).sum(axis=1)
    return BlockStateCache(A=A, X=X, count_sum=count_sum, log_factorial_sum=log_fact)


def block_map_from_cache(cache: BlockStateCache, r_b: np.ndarray,
                         quad_coeff: float, lam: float) -> tuple[np.ndarray, float]:
    """MAP search over the precomputed cache for residual r_b."""
    err = cache.X - r_b[None, :]
    log_prior = cache.count_sum * np.log(max(lam, 1e-300)) - cache.log_factorial_sum
    log_lik = -quad_coeff * np.sum(np.abs(err) ** 2, axis=1)
    score = log_prior + log_lik
    idx = int(np.argmax(score))
    return cache.A[idx], float(score[idx])
