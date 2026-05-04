"""AMP-style decoders.

  - AMP-BG: per-block GAMP with Bernoulli-Gaussian prior. Oracle: sigma^2, K.
  - BlockMAP: per-block exact discrete Poisson MAP, no cross-block link. No oracle.
"""

from __future__ import annotations

import numpy as np

from ..block_map import block_map_from_cache, build_block_state_cache
from ..estimators import estimate_lambda_energy, estimate_noise_var
from ..scenario import Scenario


def run_bg(scenario: Scenario, *, sigma_x_sq: float = 1.0,
           max_iter: int = 30) -> tuple[np.ndarray, dict]:
    """Per-block GAMP with BG prior. Real-valued only."""
    Y = scenario.Y
    if np.iscomplexobj(Y) or np.iscomplexobj(next(iter(scenario.block_dicts.values()))):
        raise NotImplementedError("AMP-BG currently supports real-valued setup only.")
    n, M_ant = Y.shape
    dtype = Y.dtype
    h = np.ones(M_ant, dtype=dtype)
    h_norm_sq = float(np.real(np.dot(h.conj(), h)))
    y_mf = np.real(Y @ h.conj() / h_norm_sq)

    M_total = scenario.num_codewords
    rho = scenario.num_devices_active / M_total
    sigma_eff_sq = scenario.noise_var / M_ant

    counts = np.zeros(scenario.num_codewords, dtype=np.float64)
    for b, C_b in scenario.block_dicts.items():
        y_b = scenario.P_mats[b].T @ y_mf
        A_b = np.real(C_b).T
        d_b, L_b = A_b.shape
        x_hat = np.zeros(L_b)
        z = y_b.copy()
        for _ in range(max_iter):
            r = A_b.T @ z + x_hat
            tau = max(float(np.sum(z ** 2)) / d_b, sigma_eff_sq)
            log_p1 = (np.log(rho + 1e-300) - 0.5 * np.log1p(sigma_x_sq / tau)
                      - r ** 2 / (2.0 * (tau + sigma_x_sq)))
            log_p0 = np.log(1.0 - rho + 1e-300) - r ** 2 / (2.0 * tau)
            p_act = 1.0 / (1.0 + np.exp(np.clip(log_p0 - log_p1, -50.0, 50.0)))
            coeff = sigma_x_sq / (sigma_x_sq + tau)
            x_hat_new = p_act * coeff * r
            var_x_r = p_act * coeff * tau + p_act * (1.0 - p_act) * (coeff * r) ** 2
            xi = float(np.mean(var_x_r)) / tau
            z_new = y_b - A_b @ x_hat_new + (L_b / d_b) * xi * z
            delta = float(np.max(np.abs(x_hat_new - x_hat)))
            x_hat = x_hat_new; z = z_new
            if delta < 1e-6:
                break
        for local_idx, global_msg in enumerate(scenario.block_to_msg_list[b]):
            counts[global_msg] = max(0.0, round(float(x_hat[local_idx])))
    return counts, {}


def run_block_map(scenario: Scenario, *,
                  poisson_tail_tol: float = 1e-4,
                  support_tail_tol: float = 1e-4,
                  lam_cache_max: float = 0.6,
                  c_max_cap: int = 4,
                  k_max_cap: int = 6) -> tuple[np.ndarray, dict]:
    """Per-block exact Poisson MAP. Estimates sigma^2, lambda from data (no oracle)."""
    Y = scenario.Y
    P_mats = scenario.P_mats
    block_dicts = scenario.block_dicts
    num_codewords = scenario.num_codewords
    block_to_msg_list = scenario.block_to_msg_list

    sigma2 = estimate_noise_var(Y, P_mats)
    M_total = sum(block_dicts[b].shape[0] for b in block_dicts)
    lam, K_hat = estimate_lambda_energy(Y, sigma2, M_total)
    lam_design = float(np.clip(1.5 * lam, 0.05, lam_cache_max))

    n, M_ant = Y.shape
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    y_mf = Y @ h.conj() / gamma
    quad_coeff = (M_ant / sigma2) if np.iscomplexobj(Y) else (M_ant / (2.0 * sigma2))

    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, C_b in block_dicts.items():
        y_b = P_mats[b].T @ y_mf
        cache = build_block_state_cache(
            C_b, lam_design=lam_design,
            poisson_tail_tol=poisson_tail_tol, support_tail_tol=support_tail_tol,
            c_max_cap=c_max_cap, k_max_cap=k_max_cap)
        a_map, _ = block_map_from_cache(cache, y_b, quad_coeff, lam)
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_map[local_idx]

    return counts, {"noise_var_est": sigma2, "lam": lam, "K_hat": K_hat}
