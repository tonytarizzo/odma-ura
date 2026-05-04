"""Non-oracle MAP-ADMM decoder for ODMA + URA (V2 case)."""

from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaln

from ..block_map import block_map_from_cache, build_block_state_cache
from ..estimators import estimate_lambda_energy, estimate_noise_var
from ..metrics import assemble_global_counts
from ..scenario import Scenario


def _admm_core(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
    alpha_lam: float = 0.1,
    mu_res: float = 10.0,
    tau_rho: float = 2.0,
    rho_update_every: int = 5,
    rho_adapt_until: int = 25,
    lam_cache_max: float = 0.6,
    c_max_cap: int = 4,
    k_max_cap: int = 6,
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[dict[int, np.ndarray], dict]:
    n, M_ant = Y.shape
    dtype = Y.dtype
    h = np.ones(M_ant, dtype=dtype)
    gamma = float(np.real(np.vdot(h, h)))
    y_mf = (Y @ h.conj()) / gamma

    block_keys = list(block_dicts.keys())
    block_supports = {b: np.argmax(P_mats[b], axis=0).astype(int) for b in block_keys}
    M_total = sum(block_dicts[b].shape[0] for b in block_keys)

    sigma2 = estimate_noise_var(Y, P_mats)
    beta = gamma / max(sigma2, 1e-12)
    lam_init, K_hat_init = estimate_lambda_energy(Y, sigma2, M_total)
    lam = lam_init

    m_t = np.zeros(n, dtype=np.float64)
    for b in block_keys:
        m_t[block_supports[b]] += 1.0
    m_t_pos = m_t[m_t > 0]
    m_t_med = float(np.median(m_t_pos)) if m_t_pos.size else 1.0

    rho_min = 1e-3 * beta
    rho_max = 10.0 * beta * max(1.0, float(np.max(m_t)) if m_t.size else 1.0)
    rho0 = max(beta * m_t_med, 2.0 * abs(np.log(max(lam, 1e-12))))
    rho_init = float(np.clip(rho0, rho_min, rho_max))
    rho = rho_init

    lam_design = float(np.clip(1.5 * lam_init, 0.05, lam_cache_max))
    caches = {b: build_block_state_cache(
                  block_dicts[b], lam_design=lam_design,
                  poisson_tail_tol=poisson_tail_tol,
                  support_tail_tol=support_tail_tol,
                  c_max_cap=c_max_cap, k_max_cap=k_max_cap)
              for b in block_keys}
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
        lam = float(np.clip((1.0 - alpha_lam) * lam + alpha_lam * lam_emp,
                            1e-4, lam_design))

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
        "lam_design":  lam_design,
        "cache_size":  cache_size,
        "K_hat":       history[-1]["K_hat"] if history else K_hat_init,
        "K_hat_init":  K_hat_init,
        "wall_s":      time.time() - wall_start,
    }


def run(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-4,
        alpha_lam: float = 0.1, lam_cache_max: float = 0.6,
        c_max_cap: int = 4, k_max_cap: int = 6,
        max_wall_seconds: float | None = None,
        verbose: bool = False) -> tuple[np.ndarray, dict]:
    coeffs_block, meta = _admm_core(
        scenario.Y, scenario.P_mats, scenario.block_dicts,
        max_iter=max_iter, tol=tol, alpha_lam=alpha_lam,
        lam_cache_max=lam_cache_max,
        c_max_cap=c_max_cap, k_max_cap=k_max_cap,
        max_wall_seconds=max_wall_seconds, verbose=verbose,
    )
    counts = assemble_global_counts(coeffs_block, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta
