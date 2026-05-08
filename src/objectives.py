"""Shared objective diagnostics for ODMA+URA count decoders.

These mirror the "Common Objective Diagnostics" section of the decoder
report. They score a candidate count vector under different generative
assumptions:

  - F_ML: Gaussian matched-filter likelihood, beta/2 ||ybar - Phi a||^2.
  - F_Pois: independent Poisson prior; constant M*lam retained so different
    lambdas can be compared.
  - F_fixedK: Gaussian likelihood + within-count multinomial penalty,
    only valid when sum(a) == K (returns None otherwise).
  - F_varK: variable-K multinomial that explicitly carries the K!, K log M,
    and -log p_K(K) terms.
  - F_softK: F_varK with a Gaussian total-count prior p_K(K).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .scenario import Scenario


def matched_filter_observation(Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ybar = Y h / ||h||^2 and gamma = ||h||^2 for h = 1_M_ant."""
    h = np.ones(Y.shape[1], dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    return (Y @ h.conj()) / gamma, gamma


def build_global_dictionary(scenario: Scenario) -> np.ndarray:
    """Global dictionary Phi[:, m] = P_b c_m used by F_ML."""
    dtype = np.complex128 if np.iscomplexobj(scenario.Y) or np.iscomplexobj(scenario.codebook) else np.float64
    Phi = np.zeros((scenario.n, scenario.num_codewords), dtype=dtype)
    for b, msg_list in scenario.block_to_msg_list.items():
        P_b = scenario.P_mats[b]
        for m in msg_list:
            Phi[:, m] = P_b @ scenario.codebook[m]
    return Phi


def ml_objective(scenario: Scenario, counts: np.ndarray, *,
                 noise_var: float | None = None) -> float:
    """F_ML(a) = beta/2 ||ybar - Phi a||^2."""
    ybar, gamma = matched_filter_observation(scenario.Y)
    sigma2 = float(scenario.noise_var if noise_var is None else noise_var)
    beta = gamma / max(sigma2, 1e-12)
    resid = ybar - build_global_dictionary(scenario) @ np.asarray(counts, dtype=np.float64)
    return 0.5 * beta * float(np.real(np.vdot(resid, resid)))


def log_factorial_penalty(counts: np.ndarray) -> float:
    """sum_m log(a_m!)."""
    return float(gammaln(np.asarray(counts, dtype=np.float64) + 1.0).sum())


def poisson_map_objective(scenario: Scenario, counts: np.ndarray,
                          lam: float | None = None, *,
                          noise_var: float | None = None) -> float:
    """F_Pois(a) = F_ML(a) + M*lam + sum_m log(a_m!) - K(a) log(lam).

    The constant M*lam is retained so objectives with different lam are
    comparable; for fixed lam it can be ignored.
    """
    counts = np.asarray(counts, dtype=np.float64)
    M = max(int(scenario.num_codewords), 1)
    lam_eff = max(float(scenario.num_devices_active / M if lam is None else lam), 1e-12)
    K = float(counts.sum())
    return (ml_objective(scenario, counts, noise_var=noise_var)
            + log_factorial_penalty(counts)
            + M * lam_eff
            - K * np.log(lam_eff))


def fixed_k_map_objective(scenario: Scenario, counts: np.ndarray,
                          K: int | None = None, *,
                          noise_var: float | None = None) -> float | None:
    """F_fixedK(a) = F_ML(a) + sum_m log(a_m!), valid only when sum(a) == K.

    The constants -log(K!) and K*log(M) are dropped because K is fixed.
    """
    counts = np.asarray(counts, dtype=np.float64)
    K_eff = int(scenario.num_devices_active if K is None else K)
    if int(round(float(counts.sum()))) != K_eff:
        return None
    return ml_objective(scenario, counts, noise_var=noise_var) + log_factorial_penalty(counts)


def var_k_map_objective(scenario: Scenario, counts: np.ndarray, *,
                        neg_log_p_K: float = 0.0,
                        noise_var: float | None = None) -> float:
    """F_varK(a) = F_ML(a) + sum_m log(a_m!) - log(K!) + K log M - log p_K(K).

    Requires the caller to pass `neg_log_p_K = -log p_K(K(a))` for the
    chosen total-count prior. The K!, K log M terms are kept because they
    are non-constant when comparing different total counts.
    """
    counts = np.asarray(counts, dtype=np.float64)
    K = float(counts.sum())
    M = max(int(scenario.num_codewords), 1)
    return (ml_objective(scenario, counts, noise_var=noise_var)
            + log_factorial_penalty(counts)
            - float(gammaln(K + 1.0))
            + K * float(np.log(M))
            + float(neg_log_p_K))


def diagnostic_soft_k_sigma(mu_K: float) -> float:
    """Broad sigma_K used only for the common synthetic-trial diagnostic."""
    return max(5.0, 0.5 * float(mu_K))


def soft_k_map_objective(scenario: Scenario, counts: np.ndarray,
                         mu_K: float, sigma_K: float, *,
                         noise_var: float | None = None) -> float:
    """F_softK(a) = F_varK(a) with -log p_K(K) = (K - mu_K)^2/(2 sigma_K^2).

    This is the variable-K multinomial objective with a Gaussian batch-level
    total-count prior; it is not an independent Poisson rate prior.
    """
    counts = np.asarray(counts, dtype=np.float64)
    K = float(counts.sum())
    neg_log_p_K = 0.5 * ((K - float(mu_K)) / max(float(sigma_K), 1e-9)) ** 2
    return var_k_map_objective(scenario, counts, neg_log_p_K=neg_log_p_K, noise_var=noise_var)


def objective_diagnostics(scenario: Scenario, counts: np.ndarray) -> dict:
    """Cross-decoder objective scores at common true generative settings."""
    counts = np.asarray(counts, dtype=np.float64)
    fixed_obj = fixed_k_map_objective(scenario, counts)
    return {
        "obj_ml": ml_objective(scenario, counts),
        "obj_poisson_true_lam": poisson_map_objective(scenario, counts),
        "obj_fixed_k_true": fixed_obj,
        "obj_soft_k_true": soft_k_map_objective(
            scenario, counts, mu_K=float(scenario.num_devices_active),
            sigma_K=diagnostic_soft_k_sigma(float(scenario.num_devices_active))),
        "obj_count_sum": float(counts.sum()),
        "obj_fixed_k_gap": float(counts.sum() - scenario.num_devices_active),
    }
