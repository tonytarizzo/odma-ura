"""AMP-style decoders.

  - AMP-BG: per-block GAMP with Bernoulli-Gaussian prior. Oracle: sigma^2, K
    (uses the true scenario noise variance and activity rate). Treats each
    block independently (no cross-block resource consistency), so this is a
    structural baseline rather than a tight match to the V2 model.
"""

from __future__ import annotations

import numpy as np

from ..scenario import Scenario


def _failed_counts(scenario: Scenario, reason: str, **meta) -> tuple[np.ndarray, dict]:
    return np.zeros(scenario.num_codewords, dtype=np.float64), {
        "converged": False,
        "decoder_failure": True,
        "failure_reason": reason,
        **meta,
    }


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
    if not (0.0 < rho < 1.0):
        return _failed_counts(
            scenario,
            "AMP-BG Bernoulli activity rate K/M is outside (0, 1).",
            rho_activity=float(rho),
            K_target=float(scenario.num_devices_active),
            iterations=0,
        )

    sigma_eff_sq = scenario.noise_var / M_ant

    counts = np.zeros(scenario.num_codewords, dtype=np.float64)
    max_used_iter = 0
    for b, C_b in scenario.block_dicts.items():
        y_b = scenario.P_mats[b].T @ y_mf
        A_b = np.real(C_b).T
        d_b, L_b = A_b.shape
        x_hat = np.zeros(L_b)
        z = y_b.copy()
        for it in range(1, max_iter + 1):
            r = A_b.T @ z + x_hat
            tau = max(float(np.sum(z ** 2)) / d_b, sigma_eff_sq)
            if not np.isfinite(tau) or tau <= 0.0:
                return _failed_counts(
                    scenario, "AMP-BG numerical divergence: nonfinite tau.",
                    block=int(b), iterations=max_used_iter,
                    rho_activity=float(rho),
                )
            log_p1 = (np.log(rho + 1e-300) - 0.5 * np.log1p(sigma_x_sq / tau)
                      - r ** 2 / (2.0 * (tau + sigma_x_sq)))
            log_p0 = np.log(1.0 - rho + 1e-300) - r ** 2 / (2.0 * tau)
            p_act = 1.0 / (1.0 + np.exp(np.clip(log_p0 - log_p1, -50.0, 50.0)))
            coeff = sigma_x_sq / (sigma_x_sq + tau)
            x_hat_new = p_act * coeff * r
            var_x_r = p_act * coeff * tau + p_act * (1.0 - p_act) * (coeff * r) ** 2
            xi = float(np.mean(var_x_r)) / tau
            z_new = y_b - A_b @ x_hat_new + (L_b / d_b) * xi * z
            if not (np.all(np.isfinite(x_hat_new)) and np.all(np.isfinite(z_new))):
                return _failed_counts(
                    scenario, "AMP-BG numerical divergence: nonfinite iterate.",
                    block=int(b), iterations=max_used_iter,
                    rho_activity=float(rho),
                )
            delta = float(np.max(np.abs(x_hat_new - x_hat)))
            x_hat = x_hat_new
            z = z_new
            max_used_iter = max(max_used_iter, it)
            if delta < 1e-6:
                break
        for local_idx, global_msg in enumerate(scenario.block_to_msg_list[b]):
            value = float(x_hat[local_idx])
            if not np.isfinite(value):
                return _failed_counts(
                    scenario, "AMP-BG numerical divergence: nonfinite estimate.",
                    block=int(b), iterations=max_used_iter,
                    rho_activity=float(rho),
                )
            counts[global_msg] = max(0.0, round(value))
    return counts, {
        "converged": True,
        "decoder_failure": False,
        "iterations": max_used_iter,
        "rho_activity": float(rho),
    }
