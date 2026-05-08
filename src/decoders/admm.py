"""MAP-ADMM family for ODMA + URA (V2 common-signature model).

Includes:
  - ADMM-Poisson: independent-Poisson count prior with EM lambda updates.
  - ADMM-Multinom: multinomial count prior with a fixed soft total-K prior.
  - Residual-MAP: block coordinate descent on the variable-K multinomial MAP.
  - ADMM-KDP-OracleK / SoftK: ADMM with fixed-K or soft-K DP a-step.
  - ADMM-KDP-{SpectralRho, Anderson, SRA}: penalty/acceleration variants.

All decoders match the V2 derivation in `docs/current_decoder_research_plan.tex`:
  - sigma^2 from the orthogonal antenna subspace (M_ant >= 2).
  - K initialized from a weak/uninformative prior, never an energy estimate.
  - State cache caps adapted to the expected per-block count, with explicit
    memory budget; cache is the optimization domain A_b, not just a speed cache.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaln

from ..block_map import (
    DEFAULT_MAX_STATES,
    build_block_state_cache,
    design_caps,
    block_map_from_cache,
)
from ..estimators import estimate_noise_var, initial_k_prior, initial_lambda
from ..metrics import assemble_global_counts
from ..scenario import Scenario


def _matched_filter_observation(Y: np.ndarray) -> tuple[np.ndarray, float]:
    M_ant = Y.shape[1]
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    return (Y @ h.conj()) / gamma, gamma


def _block_supports(P_mats: dict[int, np.ndarray], block_keys: list[int]) -> dict[int, np.ndarray]:
    return {b: np.argmax(P_mats[b], axis=0).astype(int) for b in block_keys}


def _resource_multiplicity(n: int, block_supports: dict[int, np.ndarray]) -> np.ndarray:
    m_t = np.zeros(n, dtype=np.float64)
    for S_b in block_supports.values():
        m_t[S_b] += 1.0
    return m_t


def _zero_block_counts(block_dicts: dict[int, np.ndarray],
                       block_keys: list[int]) -> dict[int, np.ndarray]:
    return {b: np.zeros(block_dicts[b].shape[0], dtype=np.float64) for b in block_keys}


def _rho_init(beta: float, m_t: np.ndarray, lam: float) -> tuple[float, float, float]:
    m_t_pos = m_t[m_t > 0]
    m_t_med = float(np.median(m_t_pos)) if m_t_pos.size else 1.0
    rho_min = 1e-3 * beta
    rho_max = 10.0 * beta * max(1.0, float(np.max(m_t)) if m_t.size else 1.0)
    rho0 = max(beta * m_t_med, 2.0 * abs(np.log(max(lam, 1e-12))))
    return float(np.clip(rho0, rho_min, rho_max)), rho_min, rho_max


def _build_caches(block_dicts: dict[int, np.ndarray], block_keys: list[int],
                  K_target_total: float, *, num_blocks: int,
                  cache_margin: float = 1.5, max_c: int = 8,
                  max_k: int | None = None,
                  max_states: int = DEFAULT_MAX_STATES) -> dict[int, object]:
    K_per_block = max(1.0, float(K_target_total) / max(int(num_blocks), 1))
    caches: dict[int, object] = {}
    for b in block_keys:
        C_b = block_dicts[b]
        c_max, k_max = design_caps(C_b.shape[0], K_per_block, margin=cache_margin,
                                   max_c=max_c, max_k=max_k, max_states=max_states)
        caches[b] = build_block_state_cache(C_b, c_max=c_max, k_max=k_max,
                                            max_states=max_states)
    return caches


def _state_costs_by_count(cache, r_b: np.ndarray, quad_coeff: float) -> tuple[np.ndarray, np.ndarray]:
    """Best (cost, state-index) for each within-block total count k."""
    err = cache.X - r_b[None, :]
    costs = quad_coeff * np.sum(np.abs(err) ** 2, axis=1) + cache.log_factorial_sum
    counts = np.rint(cache.count_sum).astype(int)
    max_count = int(counts.max()) if counts.size else 0
    best_cost = np.full(max_count + 1, np.inf)
    best_idx = np.full(max_count + 1, -1, dtype=np.int64)
    for idx, k in enumerate(counts):
        c = float(costs[idx])
        if c < best_cost[k]:
            best_cost[k] = c
            best_idx[k] = idx
    return best_cost, best_idx


def _total_count_penalty(K_vals: np.ndarray, K_hat: float, M_total: int,
                         k_prior_std: float) -> np.ndarray:
    """psi(K) = K log M - log(K!) + (K - K_hat)^2 / (2 sigma_K^2)."""
    K_vals = np.asarray(K_vals, dtype=np.float64)
    multinom_K = K_vals * np.log(max(M_total, 1)) - gammaln(K_vals + 1.0)
    k_prior = 0.5 * ((K_vals - float(K_hat)) / max(float(k_prior_std), 1e-9)) ** 2
    return multinom_K + k_prior


def _dp_select_states(best_by_block: dict[int, tuple[np.ndarray, np.ndarray]],
                      block_keys: list[int], K_hat: float, M_total: int,
                      k_prior_std: float) -> tuple[dict[int, int], int, float]:
    """Soft-K DP across blocks; selects per-block state with smallest D_B(K) + psi(K)."""
    max_total = int(sum(len(best_by_block[b][0]) - 1 for b in block_keys))
    dp = np.full(max_total + 1, np.inf)
    dp[0] = 0.0
    parents: list[dict[int, tuple[int, int]]] = []
    active_max = 0

    for b in block_keys:
        best_cost, best_idx = best_by_block[b]
        next_dp = np.full(max_total + 1, np.inf)
        parent: dict[int, tuple[int, int]] = {}
        valid_counts = np.flatnonzero(np.isfinite(best_cost) & (best_idx >= 0))
        for prev_total in range(active_max + 1):
            prev_cost = dp[prev_total]
            if not np.isfinite(prev_cost):
                continue
            for k in valid_counts:
                new_total = prev_total + int(k)
                new_cost = prev_cost + float(best_cost[k])
                if new_cost < next_dp[new_total]:
                    next_dp[new_total] = new_cost
                    parent[new_total] = (prev_total, int(k))
        active_max += len(best_cost) - 1
        dp = next_dp
        parents.append(parent)

    totals = np.arange(max_total + 1)
    total_scores = dp + _total_count_penalty(totals, K_hat, M_total, k_prior_std)
    K_star = int(np.argmin(total_scores))
    if not np.isfinite(total_scores[K_star]):
        raise RuntimeError("DP state selection failed: no finite total-count path.")

    selected_counts: dict[int, int] = {}
    total = K_star
    for b, parent in zip(reversed(block_keys), reversed(parents)):
        prev_total, k = parent[total]
        selected_counts[b] = k
        total = prev_total
    selected_idx = {b: int(best_by_block[b][1][selected_counts[b]]) for b in block_keys}
    return selected_idx, K_star, float(total_scores[K_star])


def _dp_select_states_kdp(
    best_by_block: dict[int, tuple[np.ndarray, np.ndarray]],
    block_keys: list[int],
    *,
    fixed_K: int | None = None,
    K_hat: float | None = None,
    M_total: int | None = None,
    k_prior_std: float | None = None,
    temperature: float = 1.0,
) -> tuple[dict[int, int], int, float, float]:
    """Either oracle fixed-K or soft-K DP with a posterior-mean K_post readout."""
    max_total = int(sum(len(best_by_block[b][0]) - 1 for b in block_keys))
    dp = np.full(max_total + 1, np.inf)
    dp[0] = 0.0
    parents: list[dict[int, tuple[int, int]]] = []
    active_max = 0

    for b in block_keys:
        best_cost, best_idx = best_by_block[b]
        next_dp = np.full(max_total + 1, np.inf)
        parent: dict[int, tuple[int, int]] = {}
        valid_counts = np.flatnonzero(np.isfinite(best_cost) & (best_idx >= 0))
        for prev_total in range(active_max + 1):
            prev_cost = dp[prev_total]
            if not np.isfinite(prev_cost):
                continue
            for q in valid_counts:
                new_total = prev_total + int(q)
                new_cost = prev_cost + float(best_cost[q])
                if new_cost < next_dp[new_total]:
                    next_dp[new_total] = new_cost
                    parent[new_total] = (prev_total, int(q))
        active_max += len(best_cost) - 1
        dp = next_dp
        parents.append(parent)

    totals = np.arange(max_total + 1)
    if fixed_K is not None:
        K_star = int(fixed_K)
        if K_star < 0 or K_star > max_total or not np.isfinite(dp[K_star]):
            raise RuntimeError(f"fixed-K DP has no feasible cached path for K={K_star}.")
        total_scores = dp
        K_post = float(K_star)
    else:
        if K_hat is None or M_total is None or k_prior_std is None:
            raise ValueError("soft-K DP requires K_hat, M_total, and k_prior_std.")
        total_scores = dp + _total_count_penalty(totals, K_hat, M_total, k_prior_std)
        finite = np.isfinite(total_scores)
        if not np.any(finite):
            raise RuntimeError("soft-K DP state selection failed: no finite total-count path.")
        finite_scores = total_scores[finite]
        finite_totals = totals[finite]
        K_star = int(finite_totals[int(np.argmin(finite_scores))])
        scaled = -finite_scores / max(float(temperature), 1e-12)
        scaled -= float(np.max(scaled))
        weights = np.exp(scaled)
        K_post = float(np.dot(weights, finite_totals) / np.sum(weights))

    selected_counts: dict[int, int] = {}
    total = K_star
    for b, parent in zip(reversed(block_keys), reversed(parents)):
        prev_total, q = parent[total]
        selected_counts[b] = q
        total = prev_total
    selected_idx = {b: int(best_by_block[b][1][selected_counts[b]]) for b in block_keys}
    return selected_idx, K_star, float(total_scores[K_star]), K_post


def _multinomial_objective(y_mf: np.ndarray, P_mats: dict[int, np.ndarray],
                           block_dicts: dict[int, np.ndarray],
                           a_b: dict[int, np.ndarray], beta: float,
                           K_hat: float, k_prior_std: float,
                           M_total: int) -> float:
    y_hat = np.zeros_like(y_mf)
    log_fact_sum = 0.0
    K_total = 0.0
    for b, a in a_b.items():
        y_hat += P_mats[b] @ (block_dicts[b].T @ a)
        log_fact_sum += float(gammaln(a + 1.0).sum())
        K_total += float(np.sum(a))
    data_term = 0.5 * beta * float(np.real(np.vdot(y_mf - y_hat, y_mf - y_hat)))
    total_penalty = float(_total_count_penalty(np.array([K_total]), K_hat, M_total, k_prior_std)[0])
    return data_term + log_fact_sum + total_penalty


def _fixed_k_objective(y_mf: np.ndarray, P_mats: dict[int, np.ndarray],
                       block_dicts: dict[int, np.ndarray],
                       a_b: dict[int, np.ndarray], beta: float,
                       K_target: int) -> float:
    y_hat = np.zeros_like(y_mf)
    log_fact_sum = 0.0
    K_total = 0.0
    for b, a in a_b.items():
        y_hat += P_mats[b] @ (block_dicts[b].T @ a)
        log_fact_sum += float(gammaln(a + 1.0).sum())
        K_total += float(np.sum(a))
    if int(round(K_total)) != int(K_target):
        return float("inf")
    data_term = 0.5 * beta * float(np.real(np.vdot(y_mf - y_hat, y_mf - y_hat)))
    return data_term + log_fact_sum


def _block_signals(block_dicts: dict[int, np.ndarray],
                   a_b: dict[int, np.ndarray],
                   block_keys: list[int]) -> dict[int, np.ndarray]:
    return {b: block_dicts[b].T @ a_b[b] for b in block_keys}


def _flatten_blocks(vals: dict[int, np.ndarray], block_keys: list[int]) -> np.ndarray:
    return np.concatenate([np.ravel(vals[b]) for b in block_keys])


def _unflatten_blocks(flat: np.ndarray, template: dict[int, np.ndarray],
                      block_keys: list[int]) -> dict[int, np.ndarray]:
    out = {}
    offset = 0
    for b in block_keys:
        size = template[b].size
        out[b] = flat[offset:offset + size].reshape(template[b].shape).copy()
        offset += size
    return out


def _norm_flat(vec: np.ndarray) -> float:
    return float(np.sqrt(max(np.real(np.vdot(vec, vec)), 0.0)))


def _real_inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.real(np.vdot(a, b)))


def _anderson_dual_update(y_old: np.ndarray, y_new: np.ndarray,
                          aa_y: list[np.ndarray], aa_g: list[np.ndarray],
                          aa_f: list[np.ndarray], memory: int,
                          max_step: float, damping: float) -> tuple[np.ndarray, bool]:
    f_new = y_new - y_old
    aa_y.append(y_old.copy())
    aa_g.append(y_new.copy())
    aa_f.append(f_new.copy())
    if len(aa_f) > memory + 1:
        aa_y.pop(0); aa_g.pop(0); aa_f.pop(0)
    if len(aa_f) < 2:
        return y_new, False

    F = np.column_stack(aa_f)
    G = np.column_stack(aa_g)
    gram = np.real(F.conj().T @ F)
    ones = np.ones((gram.shape[0], 1))
    kkt = np.block([[gram, ones], [ones.T, np.zeros((1, 1))]])
    rhs = np.zeros(gram.shape[0] + 1)
    rhs[-1] = 1.0
    try:
        gamma = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
    except np.linalg.LinAlgError:
        return y_new, False

    candidate = G @ gamma
    if not np.all(np.isfinite(candidate)):
        return y_new, False
    step = _norm_flat(candidate - y_new)
    if step > max_step * max(1.0, _norm_flat(y_new)):
        return y_new, False
    return (1.0 - damping) * y_new + damping * candidate, True


def _admm_core(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    num_blocks: int,
    max_iter: int = 50,
    tol: float = 1e-4,
    alpha_lam: float = 0.1,
    mu_res: float = 10.0,
    tau_rho: float = 2.0,
    rho_update_every: int = 5,
    rho_adapt_until: int = 25,
    cache_margin: float = 1.5,
    cache_max_c: int = 8,
    cache_max_k: int | None = None,
    cache_max_states: int = DEFAULT_MAX_STATES,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[dict[int, np.ndarray], dict]:
    n, _ = Y.shape
    dtype = Y.dtype
    y_mf, gamma = _matched_filter_observation(Y)

    block_keys = list(block_dicts.keys())
    block_supports = _block_supports(P_mats, block_keys)
    M_total = sum(block_dicts[b].shape[0] for b in block_keys)

    sigma2 = estimate_noise_var(Y)
    beta = gamma / max(sigma2, 1e-12)
    mu_K_init, _ = initial_k_prior(M_total)
    lam_init = initial_lambda(M_total)
    lam = lam_init

    m_t = _resource_multiplicity(n, block_supports)
    rho_init, rho_min, rho_max = _rho_init(beta, m_t, lam)
    rho = rho_init

    caches = _build_caches(block_dicts, block_keys, K_target_total=mu_K_init,
                           num_blocks=num_blocks, cache_margin=cache_margin,
                           max_c=cache_max_c, max_k=cache_max_k,
                           max_states=cache_max_states)
    cache_size = sum(caches[b].A.shape[0] for b in block_keys)

    x_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}
    u_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}
    a_b = {b: np.zeros(block_dicts[b].shape[0])              for b in block_keys}

    converged = False
    timed_out = False
    it_used = 0
    history: list[dict] = []
    wall_start = time.time()

    for it in range(1, max_iter + 1):
        it_used = it
        if max_wall_seconds is not None and (time.time() - wall_start) > max_wall_seconds:
            timed_out = True
            break

        a_b_prev = {b: a_b[b].copy() for b in block_keys}

        q_b = {b: block_dicts[b].T @ a_b[b] - u_b[b] for b in block_keys}
        q_sigma = np.zeros(n, dtype=dtype)
        for b in block_keys:
            q_sigma += P_mats[b] @ q_b[b]
        for b in block_keys:
            t_idx = block_supports[b]
            coeffs = beta / (rho + m_t[t_idx] * beta)
            x_b[b] = q_b[b] + coeffs * (y_mf[t_idx] - q_sigma[t_idx])

        for b in block_keys:
            r_b = x_b[b] + u_b[b]
            a_b[b], _ = block_map_from_cache(caches[b], r_b, rho / 2.0, lam)

        r_pri_sq = 0.0
        r_dual_sq = 0.0
        for b in block_keys:
            C_bT_a = block_dicts[b].T @ a_b[b]
            res = x_b[b] - C_bT_a
            u_b[b] += res
            r_pri_sq += float(np.real(np.vdot(res, res)))
            d_a = block_dicts[b].T @ (a_b[b] - a_b_prev[b])
            r_dual_sq += rho * rho * float(np.real(np.vdot(d_a, d_a)))
        r_pri = np.sqrt(r_pri_sq); r_dual = np.sqrt(r_dual_sq)

        K_hat = sum(float(np.sum(a_b[b])) for b in block_keys)
        lam_emp = K_hat / max(M_total, 1)
        lam = float(np.clip((1.0 - alpha_lam) * lam + alpha_lam * lam_emp, 1e-4, 0.95))

        Phi_a = np.zeros(n, dtype=dtype)
        for b in block_keys:
            Phi_a += P_mats[b] @ (block_dicts[b].T @ a_b[b])
        data_term = 0.5 * beta * float(np.real(np.vdot(y_mf - Phi_a, y_mf - Phi_a)))
        log_lam = np.log(max(lam, 1e-300))
        log_fact_sum = sum(float(gammaln(a_b[b] + 1.0).sum()) for b in block_keys)
        prior_term = M_total * lam - K_hat * log_lam + log_fact_sum
        objective = data_term + prior_term

        if (it % rho_update_every == 0) and (it <= rho_adapt_until):
            if r_pri > mu_res * r_dual:
                rho_new = min(tau_rho * rho, rho_max)
            elif r_dual > mu_res * r_pri:
                rho_new = max(rho / tau_rho, rho_min)
            else:
                rho_new = rho
            if rho_new != rho:
                scale = rho / rho_new
                for b in block_keys:
                    u_b[b] *= scale
                rho = rho_new

        history.append({
            "iter": it,
            "r_pri": r_pri, "r_dual": r_dual,
            "objective": objective,
            "rho": rho, "lam": lam, "K_hat": K_hat,
        })
        if verbose:
            print(f"  [iter {it:03d}] r_pri={r_pri:.3e}  r_dual={r_dual:.3e}  "
                  f"obj={objective:.3e}  rho={rho:.3f}  lam={lam:.4f}", flush=True)

        if r_pri < tol and r_dual < tol:
            converged = True
            break

    return a_b, {
        "converged":   converged,
        "timed_out":   timed_out,
        "iterations":  it_used,
        "history":     history,
        "tol":         tol,
        "rho":         rho,
        "rho_init":    rho_init,
        "noise_var_est": sigma2,
        "lam":         lam,
        "lam_init":    lam_init,
        "cache_size":  cache_size,
        "K_hat":       history[-1]["K_hat"] if history else 0.0,
        "wall_s":      time.time() - wall_start,
    }


def _admm_multinomial_core(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    num_blocks: int,
    max_iter: int = 50,
    tol: float = 1e-4,
    cache_margin: float = 1.5,
    cache_max_c: int = 8,
    cache_max_k: int | None = None,
    cache_max_states: int = DEFAULT_MAX_STATES,
    sigma_k_frac: float = 0.5,
    sigma_k_min: float = 5.0,
    mu_res: float = 10.0,
    tau_rho: float = 2.0,
    rho_update_every: int = 5,
    rho_adapt_until: int = 25,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[dict[int, np.ndarray], dict]:
    n, _ = Y.shape
    dtype = Y.dtype
    y_mf, gamma = _matched_filter_observation(Y)
    block_keys = list(block_dicts.keys())
    block_supports = _block_supports(P_mats, block_keys)
    M_total = sum(block_dicts[b].shape[0] for b in block_keys)

    sigma2 = estimate_noise_var(Y)
    beta = gamma / max(sigma2, 1e-12)
    mu_K, sigma_K = initial_k_prior(M_total, sigma_frac=sigma_k_frac, sigma_min=sigma_k_min)

    m_t = _resource_multiplicity(n, block_supports)
    rho_init, rho_min, rho_max = _rho_init(beta, m_t, max(mu_K / max(M_total, 1), 1e-4))
    rho = rho_init

    caches = _build_caches(block_dicts, block_keys, K_target_total=mu_K,
                           num_blocks=num_blocks, cache_margin=cache_margin,
                           max_c=cache_max_c, max_k=cache_max_k,
                           max_states=cache_max_states)
    cache_size = sum(caches[b].A.shape[0] for b in block_keys)

    x_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}
    u_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}
    a_b = {b: np.zeros(block_dicts[b].shape[0]) for b in block_keys}

    converged = False
    timed_out = False
    it_used = 0
    history: list[dict] = []
    wall_start = time.time()

    for it in range(1, max_iter + 1):
        it_used = it
        if max_wall_seconds is not None and (time.time() - wall_start) > max_wall_seconds:
            timed_out = True
            break

        a_b_prev = {b: a_b[b].copy() for b in block_keys}

        q_b = {b: block_dicts[b].T @ a_b[b] - u_b[b] for b in block_keys}
        q_sigma = np.zeros(n, dtype=dtype)
        for b in block_keys:
            q_sigma += P_mats[b] @ q_b[b]
        for b in block_keys:
            t_idx = block_supports[b]
            coeffs = beta / (rho + m_t[t_idx] * beta)
            x_b[b] = q_b[b] + coeffs * (y_mf[t_idx] - q_sigma[t_idx])

        best_by_block = {}
        for b in block_keys:
            r_b = x_b[b] + u_b[b]
            best_by_block[b] = _state_costs_by_count(caches[b], r_b, rho / 2.0)
        selected_idx, K_star, dp_score = _dp_select_states(
            best_by_block, block_keys, mu_K, M_total, sigma_K)
        for b in block_keys:
            a_b[b] = caches[b].A[selected_idx[b]].copy()

        r_pri_sq = 0.0
        r_dual_sq = 0.0
        for b in block_keys:
            C_bT_a = block_dicts[b].T @ a_b[b]
            res = x_b[b] - C_bT_a
            u_b[b] += res
            r_pri_sq += float(np.real(np.vdot(res, res)))
            d_a = block_dicts[b].T @ (a_b[b] - a_b_prev[b])
            r_dual_sq += rho * rho * float(np.real(np.vdot(d_a, d_a)))
        r_pri = np.sqrt(r_pri_sq); r_dual = np.sqrt(r_dual_sq)

        K_total = sum(float(np.sum(a_b[b])) for b in block_keys)
        objective = _multinomial_objective(
            y_mf, P_mats, block_dicts, a_b, beta, mu_K, sigma_K, M_total)

        if (it % rho_update_every == 0) and (it <= rho_adapt_until):
            if r_pri > mu_res * r_dual:
                rho_new = min(tau_rho * rho, rho_max)
            elif r_dual > mu_res * r_pri:
                rho_new = max(rho / tau_rho, rho_min)
            else:
                rho_new = rho
            if rho_new != rho:
                scale = rho / rho_new
                for b in block_keys:
                    u_b[b] *= scale
                rho = rho_new

        history.append({
            "iter": it,
            "r_pri": r_pri, "r_dual": r_dual,
            "objective": objective, "dp_score": dp_score,
            "rho": rho, "K_hat": K_total, "K_target": mu_K, "K_star": K_star,
        })
        if verbose:
            print(f"  [iter {it:03d}] r_pri={r_pri:.3e} r_dual={r_dual:.3e} "
                  f"obj={objective:.3e} rho={rho:.3f} K={K_total:.1f}/{mu_K:.1f}",
                  flush=True)

        if r_pri < tol and r_dual < tol:
            converged = True
            break

    return a_b, {
        "converged": converged,
        "timed_out": timed_out,
        "iterations": it_used,
        "history": history,
        "tol": tol,
        "rho": rho,
        "rho_init": rho_init,
        "noise_var_est": sigma2,
        "K_hat": history[-1]["K_hat"] if history else 0.0,
        "K_target": mu_K,
        "sigma_K": sigma_K,
        "cache_size": cache_size,
        "wall_s": time.time() - wall_start,
    }


def _residual_map_core(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    num_blocks: int,
    max_iter: int = 30,
    tol: float = 1e-6,
    cache_margin: float = 1.5,
    cache_max_c: int = 8,
    cache_max_k: int | None = None,
    cache_max_states: int = DEFAULT_MAX_STATES,
    sigma_k_frac: float = 0.5,
    sigma_k_min: float = 5.0,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[dict[int, np.ndarray], dict]:
    dtype = Y.dtype
    y_mf, gamma = _matched_filter_observation(Y)
    block_keys = list(block_dicts.keys())
    block_supports = _block_supports(P_mats, block_keys)
    M_total = sum(block_dicts[b].shape[0] for b in block_keys)

    sigma2 = estimate_noise_var(Y)
    beta = gamma / max(sigma2, 1e-12)
    mu_K, sigma_K = initial_k_prior(M_total, sigma_frac=sigma_k_frac, sigma_min=sigma_k_min)

    caches = _build_caches(block_dicts, block_keys, K_target_total=mu_K,
                           num_blocks=num_blocks, cache_margin=cache_margin,
                           max_c=cache_max_c, max_k=cache_max_k,
                           max_states=cache_max_states)
    cache_size = sum(caches[b].A.shape[0] for b in block_keys)

    a_b = {b: np.zeros(block_dicts[b].shape[0]) for b in block_keys}
    y_hat = np.zeros_like(y_mf)
    converged = False
    timed_out = False
    history: list[dict] = []
    wall_start = time.time()

    for it in range(1, max_iter + 1):
        if max_wall_seconds is not None and (time.time() - wall_start) > max_wall_seconds:
            timed_out = True
            break
        max_delta = 0.0

        for b in block_keys:
            old_a = a_b[b]
            old_x = block_dicts[b].T @ old_a
            residual_without = y_mf - y_hat + P_mats[b] @ old_x
            r_b = residual_without[block_supports[b]]
            K_other = sum(float(np.sum(a_b[j])) for j in block_keys if j != b)

            err = caches[b].X - r_b[None, :]
            data_cost = 0.5 * beta * np.sum(np.abs(err) ** 2, axis=1)
            K_total = K_other + caches[b].count_sum
            total_penalty = _total_count_penalty(K_total, mu_K, M_total, sigma_K)
            costs = data_cost + caches[b].log_factorial_sum + total_penalty
            idx = int(np.argmin(costs))

            new_a = caches[b].A[idx].copy()
            new_x = block_dicts[b].T @ new_a
            a_b[b] = new_a
            y_hat += P_mats[b] @ (new_x - old_x)
            max_delta = max(max_delta, float(np.max(np.abs(new_a - old_a))))

        objective = _multinomial_objective(
            y_mf, P_mats, block_dicts, a_b, beta, mu_K, sigma_K, M_total)
        K_total = sum(float(np.sum(a_b[b])) for b in block_keys)
        history.append({
            "iter": it, "delta": max_delta, "objective": objective,
            "K_hat": K_total, "K_target": mu_K,
        })
        if verbose:
            print(f"  [iter {it:03d}] delta={max_delta:.3e} obj={objective:.3e} "
                  f"K={K_total:.1f}/{mu_K:.1f}", flush=True)
        if max_delta < tol:
            converged = True
            break

    return a_b, {
        "converged": converged,
        "timed_out": timed_out,
        "iterations": len(history),
        "history": history,
        "noise_var_est": sigma2,
        "K_hat": history[-1]["K_hat"] if history else 0.0,
        "K_target": mu_K,
        "sigma_K": sigma_K,
        "cache_size": cache_size,
        "wall_s": time.time() - wall_start,
    }


def _admm_kdp_core(
    scenario: Scenario,
    *,
    k_mode: str,
    rho_policy: str = "fixed",
    use_anderson: bool = False,
    max_iter: int = 50,
    tol: float = 1e-4,
    cache_margin: float = 1.5,
    cache_max_c: int = 8,
    cache_max_k: int | None = None,
    cache_max_states: int = DEFAULT_MAX_STATES,
    sigma_k_frac: float = 0.5,
    sigma_k_min: float = 5.0,
    soft_k_update: bool = True,
    soft_k_alpha: float = 0.05,
    soft_k_burnin: int = 3,
    soft_temperature: float = 1.0,
    rho_update_every: int = 5,
    rho_adapt_until: int = 50,
    mu_res: float = 10.0,
    tau_rho: float = 2.0,
    sra_tau: float = 10.0,
    anderson_memory: int = 3,
    anderson_damping: float = 1.0,
    anderson_max_step: float = 5.0,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[dict[int, np.ndarray], dict]:
    Y = scenario.Y
    P_mats = scenario.P_mats
    block_dicts = scenario.block_dicts
    n, _ = Y.shape
    dtype = Y.dtype
    y_mf, gamma = _matched_filter_observation(Y)
    block_keys = list(block_dicts.keys())
    block_supports = _block_supports(P_mats, block_keys)
    M_total = sum(block_dicts[b].shape[0] for b in block_keys)

    sigma2 = estimate_noise_var(Y)
    beta = gamma / max(sigma2, 1e-12)
    if k_mode == "oracle":
        K_target = float(scenario.num_devices_active)
        sigma_K = 1.0
        K_fixed: int | None = int(round(K_target))
    elif k_mode == "soft":
        K_target, sigma_K = initial_k_prior(M_total, sigma_frac=sigma_k_frac, sigma_min=sigma_k_min)
        K_fixed = None
    else:
        raise ValueError(f"unknown k_mode {k_mode!r}")

    m_t = _resource_multiplicity(n, block_supports)
    rho_init, rho_min, rho_max = _rho_init(beta, m_t, max(K_target / max(M_total, 1), 1e-4))
    rho = rho_init

    caches = _build_caches(block_dicts, block_keys,
                           K_target_total=max(K_target, 1.0),
                           num_blocks=scenario.num_blocks,
                           cache_margin=cache_margin, max_c=cache_max_c,
                           max_k=cache_max_k, max_states=cache_max_states)
    cache_size = sum(caches[b].A.shape[0] for b in block_keys)
    cache_caps = {b: (int(caches[b].c_max), int(caches[b].k_max)) for b in block_keys}
    max_feasible_K = float(sum(np.max(caches[b].count_sum) for b in block_keys))

    wall_start = time.time()
    if k_mode == "oracle" and K_fixed is not None and K_fixed > max_feasible_K:
        return _zero_block_counts(block_dicts, block_keys), {
            "converged": False,
            "timed_out": False,
            "decoder_failure": True,
            "failure_reason": "fixed-K target is outside the restricted state domain.",
            "iterations": 0,
            "history": [],
            "tol": tol,
            "rho": rho,
            "rho_init": rho_init,
            "rho_policy": rho_policy,
            "rho_updates": 0,
            "anderson_steps": 0,
            "noise_var_est": sigma2,
            "K_hat": 0.0,
            "K_target": K_target,
            "K_star": K_fixed,
            "sigma_K": sigma_K,
            "cache_size": cache_size,
            "cache_caps": cache_caps,
            "max_feasible_K": max_feasible_K,
            "objective": float("nan"),
            "r_pri": float("nan"),
            "r_dual": float("nan"),
            "wall_s": time.time() - wall_start,
        }

    a_b = _zero_block_counts(block_dicts, block_keys)
    z_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}
    y_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}
    x_b = {b: np.zeros(block_dicts[b].shape[1], dtype=dtype) for b in block_keys}

    prev_x_flat: np.ndarray | None = None
    prev_z_flat: np.ndarray | None = None
    prev_y_flat: np.ndarray | None = None
    prev_ytilde_flat: np.ndarray | None = None
    aa_y: list[np.ndarray] = []
    aa_g: list[np.ndarray] = []
    aa_f: list[np.ndarray] = []

    converged = False
    timed_out = False
    rho_updates = 0
    anderson_steps = 0
    history: list[dict] = []

    for it in range(1, max_iter + 1):
        if max_wall_seconds is not None and (time.time() - wall_start) > max_wall_seconds:
            timed_out = True
            break

        z_old = {b: z_b[b].copy() for b in block_keys}
        y_old = {b: y_b[b].copy() for b in block_keys}
        y_old_flat = _flatten_blocks(y_old, block_keys)

        q_b = {b: z_old[b] - y_old[b] / rho for b in block_keys}
        q_sigma = np.zeros(n, dtype=dtype)
        for b in block_keys:
            q_sigma += P_mats[b] @ q_b[b]
        for b in block_keys:
            t_idx = block_supports[b]
            coeffs = beta / (rho + m_t[t_idx] * beta)
            x_b[b] = q_b[b] + coeffs * (y_mf[t_idx] - q_sigma[t_idx])

        best_by_block = {}
        for b in block_keys:
            r_b = x_b[b] + y_old[b] / rho
            best_by_block[b] = _state_costs_by_count(caches[b], r_b, rho / 2.0)

        try:
            if k_mode == "oracle":
                selected_idx, K_star, dp_score, K_post = _dp_select_states_kdp(
                    best_by_block, block_keys, fixed_K=K_fixed)
            else:
                selected_idx, K_star, dp_score, K_post = _dp_select_states_kdp(
                    best_by_block, block_keys, K_hat=K_target, M_total=M_total,
                    k_prior_std=sigma_K, temperature=soft_temperature)
        except RuntimeError as exc:
            return _zero_block_counts(block_dicts, block_keys), {
                "converged": False,
                "timed_out": False,
                "decoder_failure": True,
                "failure_reason": str(exc),
                "iterations": len(history),
                "history": history,
                "tol": tol,
                "rho": rho,
                "rho_init": rho_init,
                "rho_policy": rho_policy,
                "rho_updates": rho_updates,
                "anderson_steps": anderson_steps,
                "noise_var_est": sigma2,
                "K_hat": history[-1]["K_hat"] if history else 0.0,
                "K_target": K_target,
                "K_star": K_fixed if k_mode == "oracle" else None,
                "sigma_K": sigma_K,
                "cache_size": cache_size,
                "cache_caps": cache_caps,
                "max_feasible_K": max_feasible_K,
                "objective": float("nan"),
                "r_pri": float("nan"),
                "r_dual": float("nan"),
                "wall_s": time.time() - wall_start,
            }

        for b in block_keys:
            a_b[b] = caches[b].A[selected_idx[b]].copy()
        z_b = _block_signals(block_dicts, a_b, block_keys)

        y_tilde = {b: y_old[b] + rho * (x_b[b] - z_old[b]) for b in block_keys}
        y_new = {b: y_old[b] + rho * (x_b[b] - z_b[b]) for b in block_keys}
        x_flat = _flatten_blocks(x_b, block_keys)
        z_old_flat = _flatten_blocks(z_old, block_keys)
        z_flat = _flatten_blocks(z_b, block_keys)
        y_flat = _flatten_blocks(y_new, block_keys)
        ytilde_flat = _flatten_blocks(y_tilde, block_keys)

        r_pri = _norm_flat(x_flat - z_flat)
        r_dual = rho * _norm_flat(z_flat - z_old_flat)

        if use_anderson:
            y_acc, accepted = _anderson_dual_update(
                y_old_flat, y_flat, aa_y, aa_g, aa_f,
                anderson_memory, anderson_max_step, anderson_damping)
            if accepted:
                y_flat = y_acc
                y_new = _unflatten_blocks(y_flat, y_new, block_keys)
                anderson_steps += 1
        y_b = y_new

        rho_candidate = rho
        if (rho_policy == "spectral") and prev_x_flat is not None and it % rho_update_every == 0 and it <= rho_adapt_until:
            dx = x_flat - prev_x_flat
            dz = z_flat - prev_z_flat
            dy = y_flat - prev_y_flat
            dyt = ytilde_flat - prev_ytilde_flat
            denom1 = _real_inner(dx, dyt)
            denom2 = _real_inner(-dz, dy)
            if denom1 > 0.0 and denom2 > 0.0:
                rho_candidate = np.sqrt((_real_inner(dyt, dyt) * _real_inner(dy, dy)) / (denom1 * denom2))
        elif (rho_policy == "sra") and it % rho_update_every == 0 and it <= rho_adapt_until:
            dy_norm = _norm_flat(y_flat - y_old_flat)
            dz_norm = _norm_flat(z_flat - z_old_flat)
            if dy_norm > 0.0 and dz_norm > 0.0:
                rho_candidate = dy_norm / dz_norm
            elif dy_norm > 0.0:
                rho_candidate = sra_tau * rho
            elif dz_norm > 0.0:
                rho_candidate = rho / sra_tau
        elif (rho_policy == "residual") and it % rho_update_every == 0 and it <= rho_adapt_until:
            if r_pri > mu_res * r_dual:
                rho_candidate = tau_rho * rho
            elif r_dual > mu_res * r_pri:
                rho_candidate = rho / tau_rho

        rho_candidate = float(np.clip(rho_candidate, rho_min, rho_max))
        if rho_candidate != rho:
            rho = rho_candidate
            rho_updates += 1

        if k_mode == "soft" and soft_k_update and it > soft_k_burnin:
            K_target = (1.0 - soft_k_alpha) * K_target + soft_k_alpha * K_post

        K_total = sum(float(np.sum(a_b[b])) for b in block_keys)
        if k_mode == "oracle":
            objective = _fixed_k_objective(y_mf, P_mats, block_dicts, a_b, beta, int(K_fixed))
        else:
            objective = _multinomial_objective(
                y_mf, P_mats, block_dicts, a_b, beta, K_target, sigma_K, M_total)

        prev_x_flat = x_flat.copy()
        prev_z_flat = z_flat.copy()
        prev_y_flat = y_flat.copy()
        prev_ytilde_flat = ytilde_flat.copy()

        history.append({
            "iter": it, "r_pri": r_pri, "r_dual": r_dual,
            "objective": objective, "dp_score": dp_score,
            "rho": rho, "K_hat": K_total, "K_target": K_target,
            "K_star": K_star, "K_post": K_post,
        })
        if verbose:
            print(f"  [iter {it:03d}] r_pri={r_pri:.3e} r_dual={r_dual:.3e} "
                  f"obj={objective:.3e} rho={rho:.3f} K={K_total:.1f}/{K_target:.1f}",
                  flush=True)

        if r_pri < tol and r_dual < tol:
            converged = True
            break

    return a_b, {
        "converged": converged,
        "timed_out": timed_out,
        "iterations": len(history),
        "history": history,
        "tol": tol,
        "rho": rho,
        "rho_init": rho_init,
        "rho_policy": rho_policy,
        "rho_updates": rho_updates,
        "anderson_steps": anderson_steps,
        "noise_var_est": sigma2,
        "K_hat": history[-1]["K_hat"] if history else 0.0,
        "K_target": K_target,
        "sigma_K": sigma_K,
        "cache_size": cache_size,
        "cache_caps": cache_caps,
        "max_feasible_K": max_feasible_K,
        "objective": history[-1]["objective"] if history else float("nan"),
        "r_pri": history[-1]["r_pri"] if history else float("nan"),
        "r_dual": history[-1]["r_dual"] if history else float("nan"),
        "wall_s": time.time() - wall_start,
    }


# ----- public entry points --------------------------------------------------


def run_poisson(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                alpha_lam: float = 0.1, cache_margin: float = 1.5,
                cache_max_c: int = 8, cache_max_k: int | None = None,
                cache_max_states: int = DEFAULT_MAX_STATES,
                max_wall_seconds: float | None = None,
                verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_core(
        scenario.Y, scenario.P_mats, scenario.block_dicts,
        num_blocks=scenario.num_blocks,
        max_iter=max_iter, tol=tol, alpha_lam=alpha_lam,
        cache_margin=cache_margin, cache_max_c=cache_max_c,
        cache_max_k=cache_max_k, cache_max_states=cache_max_states,
        max_wall_seconds=max_wall_seconds, verbose=verbose,
    )
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_multinomial(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                    cache_margin: float = 1.5, cache_max_c: int = 8,
                    cache_max_k: int | None = None,
                    cache_max_states: int = DEFAULT_MAX_STATES,
                    sigma_k_frac: float = 0.5, sigma_k_min: float = 5.0,
                    max_wall_seconds: float | None = None,
                    verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_multinomial_core(
        scenario.Y, scenario.P_mats, scenario.block_dicts,
        num_blocks=scenario.num_blocks,
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        sigma_k_frac=sigma_k_frac, sigma_k_min=sigma_k_min,
        max_wall_seconds=max_wall_seconds, verbose=verbose,
    )
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_residual_map(scenario: Scenario, *, max_iter: int = 30, tol: float = 1e-6,
                     cache_margin: float = 1.5, cache_max_c: int = 8,
                     cache_max_k: int | None = None,
                     cache_max_states: int = DEFAULT_MAX_STATES,
                     sigma_k_frac: float = 0.5, sigma_k_min: float = 5.0,
                     max_wall_seconds: float | None = None,
                     verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _residual_map_core(
        scenario.Y, scenario.P_mats, scenario.block_dicts,
        num_blocks=scenario.num_blocks,
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        sigma_k_frac=sigma_k_frac, sigma_k_min=sigma_k_min,
        max_wall_seconds=max_wall_seconds, verbose=verbose,
    )
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_kdp_oracle(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                   cache_margin: float = 1.5, cache_max_c: int = 8,
                   cache_max_k: int | None = None,
                   cache_max_states: int = DEFAULT_MAX_STATES,
                   max_wall_seconds: float | None = None,
                   verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_kdp_core(
        scenario, k_mode="oracle", rho_policy="residual",
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        max_wall_seconds=max_wall_seconds, verbose=verbose)
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_kdp_soft(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                 cache_margin: float = 1.5, cache_max_c: int = 8,
                 cache_max_k: int | None = None,
                 cache_max_states: int = DEFAULT_MAX_STATES,
                 sigma_k_frac: float = 0.5, sigma_k_min: float = 5.0,
                 soft_k_alpha: float = 0.05, soft_k_burnin: int = 3,
                 max_wall_seconds: float | None = None,
                 verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_kdp_core(
        scenario, k_mode="soft", rho_policy="residual",
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        sigma_k_frac=sigma_k_frac, sigma_k_min=sigma_k_min,
        soft_k_alpha=soft_k_alpha, soft_k_burnin=soft_k_burnin,
        max_wall_seconds=max_wall_seconds, verbose=verbose)
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_kdp_spectral_rho(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                         cache_margin: float = 1.5, cache_max_c: int = 8,
                         cache_max_k: int | None = None,
                         cache_max_states: int = DEFAULT_MAX_STATES,
                         rho_update_every: int = 5,
                         max_wall_seconds: float | None = None,
                         verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_kdp_core(
        scenario, k_mode="soft", rho_policy="spectral",
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        rho_update_every=rho_update_every,
        max_wall_seconds=max_wall_seconds, verbose=verbose)
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_kdp_anderson(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                     cache_margin: float = 1.5, cache_max_c: int = 8,
                     cache_max_k: int | None = None,
                     cache_max_states: int = DEFAULT_MAX_STATES,
                     anderson_memory: int = 3, anderson_damping: float = 1.0,
                     max_wall_seconds: float | None = None,
                     verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_kdp_core(
        scenario, k_mode="soft", rho_policy="fixed", use_anderson=True,
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        anderson_memory=anderson_memory, anderson_damping=anderson_damping,
        max_wall_seconds=max_wall_seconds, verbose=verbose)
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


def run_kdp_sra(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
                cache_margin: float = 1.5, cache_max_c: int = 8,
                cache_max_k: int | None = None,
                cache_max_states: int = DEFAULT_MAX_STATES,
                rho_update_every: int = 5,
                max_wall_seconds: float | None = None,
                verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_kdp_core(
        scenario, k_mode="soft", rho_policy="sra",
        max_iter=max_iter, tol=tol, cache_margin=cache_margin,
        cache_max_c=cache_max_c, cache_max_k=cache_max_k,
        cache_max_states=cache_max_states,
        rho_update_every=rho_update_every,
        max_wall_seconds=max_wall_seconds, verbose=verbose)
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta


run = run_poisson
