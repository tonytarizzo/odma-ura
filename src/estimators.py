"""Non-oracle parameter estimators for the V2 common-signature model.

Assumes M_ant >= 2 throughout (single-antenna runs are not supported).

Provides:
  - estimate_noise_var(Y): orthogonal-antenna-subspace sigma^2 estimate.
  - initial_k_prior(num_codewords): broad initial (mu_K, sigma_K) batch prior.
  - initial_lambda(num_codewords): broad initial Poisson rate.
"""

from __future__ import annotations

import numpy as np


def estimate_noise_var(Y: np.ndarray) -> float:
    """sigma^2 from the antenna subspace orthogonal to h = 1_M.

    With Q_perp = I - h h^T / gamma and gamma = ||h||^2 = M_ant,
        E[||Q_perp Y_t||^2] = sigma^2 (M_ant - 1).
    Independent of a, K, codebook, and ODMA pattern overlaps.
    """
    n, M_ant = Y.shape
    if M_ant < 2:
        raise ValueError("Orthogonal-subspace noise estimate requires M_ant >= 2.")
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    Y_perp = Y - np.outer(Y @ h.conj() / gamma, h)
    return float(np.real(np.vdot(Y_perp.ravel(), Y_perp.ravel())) / (n * (M_ant - 1)))


def initial_k_prior(num_codewords: int, *, mu_frac: float = 0.25,
                    sigma_frac: float = 0.5, sigma_min: float = 5.0) -> tuple[float, float]:
    """Broad starting (mu_K, sigma_K) for decoders that update K internally.

    This is only an initialization/prior scale, not a measurement of K. Decoders
    that support EM-style K updates should let their within-iteration posterior
    quantities move away from this value.
    """
    M = max(int(num_codewords), 1)
    mu_K = float(mu_frac * M)
    sigma_K = float(max(sigma_min, sigma_frac * M))
    return mu_K, sigma_K


def initial_lambda(num_codewords: int) -> float:
    """Initial Poisson rate per slot before any iterative update.

    Uses the same broad mu_K = num_codewords/4 baseline as `initial_k_prior`
    and converts it to a per-message rate.
    """
    M = max(int(num_codewords), 1)
    mu_K, _ = initial_k_prior(M)
    return float(np.clip(mu_K / M, 0.05, 0.5))
