"""Block coordinate MAP decoders for the V2 ODMA+URA count objective.

Implements the BlockCD-OracleK and BlockCD-SoftK variants described in the
"Block Coordinate MAP Search" section of the decoder report. The score
J_b(s) carries the full variable-K multinomial penalty:

    J_b(s) = (beta/2) ||P_b^T r^{(-b)} - X_b(s)||^2
           + sum_l log(s_l!)
           - log((K_{-b} + k_b(s))!)
           + (K_{-b} + k_b(s)) log M
           + 0.5 ((K_{-b} + k_b(s) - mu_K) / sigma_K)^2

For BlockCD-OracleK we set mu_K = K_true and sigma_K very small so the
softened total-count constraint approaches a hard equality. For
BlockCD-SoftK we initialise mu_K from a weak uninformative prior and
update it via a damped pseudo-EM rule averaged over the local conditional
distributions q_b(s).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaln

from ..block_map import build_block_state_cache, design_caps
from ..estimators import estimate_noise_var, initial_k_prior
from ..metrics import assemble_global_counts
from ..objectives import matched_filter_observation, soft_k_map_objective
from ..scenario import Scenario


def _block_supports(scenario: Scenario) -> dict[int, np.ndarray]:
    return {b: np.argmax(scenario.P_mats[b], axis=0).astype(int) for b in scenario.block_dicts}


def _build_caches(scenario: Scenario, K_target: float, *, margin: float,
                  max_c: int, max_k: int | None, max_states: int):
    caches = {}
    K_per_block = max(1.0, K_target / max(scenario.num_blocks, 1))
    for b, C_b in scenario.block_dicts.items():
        c_max, k_max = design_caps(C_b.shape[0], K_per_block, margin=margin,
                                   max_c=max_c, max_k=max_k, max_states=max_states)
        caches[b] = build_block_state_cache(C_b, c_max=c_max, k_max=k_max,
                                            max_states=max_states)
    return caches


def _stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -scores / max(float(temperature), 1e-12)
    scaled -= float(np.max(scaled))
    weights = np.exp(scaled)
    return weights / np.sum(weights)


def _run_block_cd(
    scenario: Scenario,
    *,
    oracle_k: bool,
    max_iter: int = 30,
    tol: float = 1e-6,
    cache_margin: float = 1.5,
    cache_max_c: int = 8,
    cache_max_k: int | None = None,
    cache_max_states: int = 200_000,
    sigma_k_frac: float = 0.5,
    sigma_k_min: float = 5.0,
    oracle_sigma_k: float = 1.0,
    k_update: bool = True,
    k_update_alpha: float = 0.05,
    k_update_burnin: int = 3,
    soft_temperature: float = 1.0,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Sequential block coordinate descent on the variable-K multinomial MAP."""
    ybar, gamma = matched_filter_observation(scenario.Y)
    sigma2 = estimate_noise_var(scenario.Y)
    beta = gamma / max(sigma2, 1e-12)
    log_M = float(np.log(max(int(scenario.num_codewords), 1)))

    if oracle_k:
        mu_K = float(scenario.num_devices_active)
        sigma_K = max(float(oracle_sigma_k), 1e-9)
    else:
        mu_K_init, sigma_K_init = initial_k_prior(scenario.num_codewords,
                                               sigma_frac=sigma_k_frac,
                                               sigma_min=sigma_k_min)
        mu_K = mu_K_init
        sigma_K = sigma_K_init

    caches = _build_caches(scenario, K_target=max(mu_K, 1.0), margin=cache_margin,
                           max_c=cache_max_c, max_k=cache_max_k,
                           max_states=cache_max_states)
    block_supports = _block_supports(scenario)
    block_keys = list(scenario.block_dicts.keys())
    a_b = {b: np.zeros(scenario.block_dicts[b].shape[0]) for b in block_keys}
    cond_K = {b: 0.0 for b in block_keys}
    cond_var_K = {b: 0.0 for b in block_keys}
    y_hat = np.zeros_like(ybar)
    history: list[dict] = []
    converged = False
    timed_out = False
    wall_start = time.time()

    for it in range(1, max_iter + 1):
        if max_wall_seconds is not None and time.time() - wall_start > max_wall_seconds:
            timed_out = True
            break

        max_delta = 0.0
        for b in block_keys:
            old_a = a_b[b]
            old_x = scenario.block_dicts[b].T @ old_a
            residual_without = ybar - y_hat + scenario.P_mats[b] @ old_x
            r_b = residual_without[block_supports[b]]
            K_other = sum(float(np.sum(a_b[j])) for j in block_keys if j != b)

            cache = caches[b]
            err = cache.X - r_b[None, :]
            data_cost = 0.5 * beta * np.sum(np.abs(err) ** 2, axis=1)
            total_count = K_other + cache.count_sum
            multinom_K = total_count * log_M - gammaln(total_count + 1.0)
            k_prior = 0.5 * ((total_count - mu_K) / sigma_K) ** 2
            costs = data_cost + cache.log_factorial_sum + multinom_K + k_prior
            idx = int(np.argmin(costs))

            new_a = cache.A[idx].copy()
            new_x = scenario.block_dicts[b].T @ new_a
            a_b[b] = new_a
            y_hat += scenario.P_mats[b] @ (new_x - old_x)

            q_b = _stable_softmax(costs, soft_temperature)
            cond_K[b] = float(K_other + np.dot(q_b, cache.count_sum))
            local_var = float(np.dot(q_b, cache.count_sum ** 2)
                              - np.dot(q_b, cache.count_sum) ** 2)
            cond_var_K[b] = max(local_var, 0.0)
            max_delta = max(max_delta, float(np.max(np.abs(new_a - old_a))))

        K_post = float(np.mean(list(cond_K.values()))) if cond_K else 0.0
        if (not oracle_k) and k_update and it > k_update_burnin:
            mu_K = (1.0 - k_update_alpha) * mu_K + k_update_alpha * K_post

        counts = assemble_global_counts(a_b, scenario.block_to_msg_list, scenario.num_codewords)
        objective = soft_k_map_objective(scenario, counts, mu_K=mu_K, sigma_K=sigma_K, noise_var=sigma2)
        K_hard = float(counts.sum())
        history.append({
            "iter": it, "delta": max_delta, "objective": objective,
            "K_hat": K_hard, "K_post": K_post, "K_prior": mu_K, "sigma_K": sigma_K,
        })
        if verbose:
            print(f"  [iter {it:03d}] delta={max_delta:.3e} obj={objective:.3e} "
                  f"K={K_hard:.1f} prior={mu_K:.1f} sigK={sigma_K:.2f}", flush=True)
        if max_delta < tol:
            converged = True
            break

    counts = assemble_global_counts(a_b, scenario.block_to_msg_list, scenario.num_codewords)
    cache_caps = {b: (caches[b].c_max, caches[b].k_max) for b in block_keys}
    meta = {
        "converged": converged,
        "timed_out": timed_out,
        "iterations": len(history),
        "history": history,
        "noise_var_est": sigma2,
        "K_hat": float(counts.sum()),
        "K_prior": float(mu_K),
        "sigma_K": float(sigma_K),
        "oracle_K": bool(oracle_k),
        "cache_size": int(sum(caches[b].A.shape[0] for b in block_keys)),
        "cache_caps": cache_caps,
        "wall_s": time.time() - wall_start,
    }
    return counts, meta


def run_oracle_k(scenario: Scenario, *, max_iter: int = 30, tol: float = 1e-6,
                 oracle_sigma_k: float = 1.0, cache_margin: float = 1.5,
                 cache_max_c: int = 8, cache_max_k: int | None = None,
                 cache_max_states: int = 200_000,
                 max_wall_seconds: float | None = None,
                 verbose: bool = False) -> tuple[np.ndarray, dict]:
    return _run_block_cd(
        scenario, oracle_k=True, max_iter=max_iter, tol=tol,
        oracle_sigma_k=oracle_sigma_k, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        max_wall_seconds=max_wall_seconds, verbose=verbose)


def run_soft_k(scenario: Scenario, *, max_iter: int = 30, tol: float = 1e-6,
               sigma_k_frac: float = 0.5, sigma_k_min: float = 5.0,
               k_update: bool = True, k_update_alpha: float = 0.05,
               k_update_burnin: int = 3, soft_temperature: float = 1.0,
               cache_margin: float = 1.5, cache_max_c: int = 8,
               cache_max_k: int | None = None, cache_max_states: int = 200_000,
               max_wall_seconds: float | None = None,
               verbose: bool = False) -> tuple[np.ndarray, dict]:
    return _run_block_cd(
        scenario, oracle_k=False, max_iter=max_iter, tol=tol,
        sigma_k_frac=sigma_k_frac, sigma_k_min=sigma_k_min,
        k_update=k_update, k_update_alpha=k_update_alpha,
        k_update_burnin=k_update_burnin, soft_temperature=soft_temperature,
        cache_margin=cache_margin, cache_max_c=cache_max_c,
        cache_max_k=cache_max_k, cache_max_states=cache_max_states,
        max_wall_seconds=max_wall_seconds, verbose=verbose)
