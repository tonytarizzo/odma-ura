"""Per-block exact discrete MAP baseline."""

from __future__ import annotations

import numpy as np

from ..block_map import DEFAULT_MAX_STATES, block_map_from_cache, build_block_state_cache, design_caps
from ..estimators import estimate_noise_var, initial_k_prior, initial_lambda
from ..scenario import Scenario


def run_poisson(scenario: Scenario, *, cache_margin: float = 1.5, cache_max_c: int = 8,
                cache_max_k: int | None = None,
                cache_max_states: int = DEFAULT_MAX_STATES) -> tuple[np.ndarray, dict]:
    """Independent per-block Poisson MAP.

    This baseline ignores cross-block resource interference. It is useful as a
    local block-MAP comparator, but it is not an AMP/VAMP method.
    """
    sigma2 = estimate_noise_var(scenario.Y)
    M_total = sum(C_b.shape[0] for C_b in scenario.block_dicts.values())
    lam = initial_lambda(M_total)
    mu_K, _ = initial_k_prior(M_total)
    K_per_block = max(1.0, mu_K / max(scenario.num_blocks, 1))

    h = np.ones(scenario.num_antennas, dtype=scenario.Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    y_mf = scenario.Y @ h.conj() / gamma
    quad_coeff = gamma / max(2.0 * sigma2, 1e-12)

    counts = np.zeros(scenario.num_codewords, dtype=np.float64)
    cache_size = 0
    for b, C_b in scenario.block_dicts.items():
        c_max, k_max = design_caps(C_b.shape[0], K_per_block, margin=cache_margin,
                                   max_c=cache_max_c, max_k=cache_max_k,
                                   max_states=cache_max_states)
        cache = build_block_state_cache(C_b, c_max=c_max, k_max=k_max,
                                        max_states=cache_max_states)
        cache_size += cache.A.shape[0]
        a_map, _ = block_map_from_cache(cache, scenario.P_mats[b].T @ y_mf, quad_coeff, lam)
        for local_idx, global_msg in enumerate(scenario.block_to_msg_list[b]):
            counts[global_msg] = a_map[local_idx]

    return counts, {"noise_var_est": sigma2, "lam": lam, "K_target": mu_K,
                    "cache_size": cache_size}
