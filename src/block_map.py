"""Block-state enumeration cache for count decoders.

The cache is the discrete optimization domain A_b that block-MAP / ADMM /
BlockCD decoders search over. As emphasised in the decoder report, A_b is
not just a speed cache; if the true block state is not contained in A_b,
the decoder can fail even when its objective is otherwise correct.

This module exposes:
  - design_caps(L_b, K_per_block_target, *, margin, max_states): pick caps
    (c_max, k_max) adapted to the expected per-block count.
  - build_block_state_cache(C_b, *, c_max, k_max, max_states): enumerate
    nonnegative integer count vectors of length L_b with at most `k_max`
    nonzero entries each bounded by `c_max`. Raises if the resulting cache
    would exceed `max_states` (caller must shrink caps or change strategy).

Note: the (c_max, k_max) form gives states = sum_{k=0..k_max} C(L_b,k) c_max^k
which grows fast. For very high K/L_b regimes the cache cannot be made
exhaustive; that limitation is fundamental and must be reasoned about by
the caller (we surface meta info but do not silently truncate the truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product as iproduct

import numpy as np
from scipy.special import gammaln


DEFAULT_MAX_STATES = 200_000


@dataclass
class BlockStateCache:
    A: np.ndarray
    X: np.ndarray
    count_sum: np.ndarray
    log_factorial_sum: np.ndarray
    c_max: int
    k_max: int


def _state_count(L_b: int, c_max: int, k_max: int) -> int:
    """Number of states enumerated under (c_max, k_max) for a length-L_b block."""
    from math import comb
    total = 1
    for k in range(1, k_max + 1):
        total += comb(L_b, k) * (c_max ** k)
    return total


def design_caps(L_b: int, K_per_block_target: float, *,
                margin: float = 1.5, max_c: int = 8, max_k: int | None = None,
                max_states: int = DEFAULT_MAX_STATES) -> tuple[int, int]:
    """Pick (c_max, k_max) covering the expected per-block count with margin.

    Strategy: start from k_max = ceil(K_target * margin) (number of nonzero
    slots an all-ones-Poisson block would typically need) capped by L_b,
    and c_max = ceil(K_target / k_max * margin) capped at `max_c`. Then
    shrink k_max while the resulting state count exceeds `max_states`.
    """
    K = max(1.0, float(K_per_block_target))
    k_max = int(min(L_b if max_k is None else min(L_b, max_k), max(1, np.ceil(K * margin))))
    c_max = int(max(2, np.ceil(K / max(k_max, 1) * margin)))
    c_max = min(c_max, max_c)

    while k_max > 1 and _state_count(L_b, c_max, k_max) > max_states:
        k_max -= 1
    if _state_count(L_b, c_max, k_max) > max_states and c_max > 2:
        while c_max > 2 and _state_count(L_b, c_max, k_max) > max_states:
            c_max -= 1
    return c_max, k_max


def build_block_state_cache(C_b: np.ndarray, *, c_max: int, k_max: int,
                            max_states: int = DEFAULT_MAX_STATES
                            ) -> BlockStateCache:
    """Enumerate count vectors with caps (c_max, k_max) and precompute X = A C_b."""
    L_b = int(C_b.shape[0])
    c_max = int(max(1, c_max))
    k_max = int(min(L_b, max(0, k_max)))
    n_states = _state_count(L_b, c_max, k_max)
    if n_states > max_states:
        raise RuntimeError(
            f"BlockStateCache would have {n_states} states (L_b={L_b}, "
            f"c_max={c_max}, k_max={k_max}); exceeds budget {max_states}.")

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
    return BlockStateCache(A=A, X=X, count_sum=count_sum, log_factorial_sum=log_fact,
                           c_max=c_max, k_max=k_max)


def block_map_from_cache(cache: BlockStateCache, r_b: np.ndarray,
                         quad_coeff: float, lam: float) -> tuple[np.ndarray, float]:
    """Per-block Poisson MAP search over the cached states for residual r_b.

    Score is log p(a|r_b) = -quad_coeff ||X - r_b||^2 + a log(lam) - log(a!).
    """
    err = cache.X - r_b[None, :]
    log_prior = cache.count_sum * np.log(max(lam, 1e-300)) - cache.log_factorial_sum
    log_lik = -quad_coeff * np.sum(np.abs(err) ** 2, axis=1)
    score = log_prior + log_lik
    idx = int(np.argmax(score))
    return cache.A[idx], float(score[idx])
