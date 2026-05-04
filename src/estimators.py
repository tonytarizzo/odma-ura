"""Non-oracle parameter estimators (noise variance, Poisson rate)."""

from __future__ import annotations

import numpy as np


def estimate_noise_var_orthogonal(Y: np.ndarray) -> float:
    """sigma^2 from antenna subspace orthogonal to h = 1_M (M_ant > 1)."""
    n, M_ant = Y.shape
    if M_ant <= 1:
        raise ValueError("Orthogonal noise estimate requires M_ant > 1.")
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    Y_perp = Y - np.outer(Y @ h.conj() / gamma, h)
    return float(np.real(np.vdot(Y_perp, Y_perp)) / (n * (M_ant - 1)))


def estimate_noise_var_unused_resources(Y: np.ndarray,
                                        P_mats: dict[int, np.ndarray]) -> float:
    """sigma^2 from resources untouched by any block (M_ant=1 fallback)."""
    n, _ = Y.shape
    m_t = np.zeros(n, dtype=np.int64)
    for b, P_b in P_mats.items():
        m_t[np.argmax(P_b, axis=0)] += 1
    unused = m_t == 0
    n_unused = int(unused.sum())
    if n_unused < 5:
        raise ValueError(
            f"M_ant=1 noise estimate needs >=5 unused resources, found {n_unused}.")
    Y_u = Y[unused]
    return float(np.real(np.vdot(Y_u, Y_u)) / (n_unused * Y.shape[1]))


def estimate_noise_var(Y: np.ndarray, P_mats: dict[int, np.ndarray]) -> float:
    """Dispatch to orthogonal-subspace (M_ant>1) or unused-resource (M_ant=1) estimator."""
    if Y.shape[1] > 1:
        return estimate_noise_var_orthogonal(Y)
    return estimate_noise_var_unused_resources(Y, P_mats)


def estimate_lambda_energy(Y: np.ndarray, sigma2: float, num_codewords: int
                           ) -> tuple[float, float]:
    """Initial Poisson rate via signal-energy moment matching.  Returns (lam_hat, K_hat)."""
    n, M_ant = Y.shape
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    y_mf = Y @ h.conj() / gamma
    E_obs = float(np.real(np.vdot(y_mf, y_mf)))
    E_sig = max(0.0, E_obs - n * sigma2 / gamma)
    M = float(num_codewords)
    K_hat = 0.5 * (-M + np.sqrt(M * M + 4.0 * M * E_sig))
    lam_hat = float(np.clip(K_hat / M, 1e-4, 0.95))
    return lam_hat, float(K_hat)
