"""VAMP-style global sparse-regression baselines."""

from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaln

from ..estimators import estimate_noise_var
from ..objectives import build_global_dictionary, matched_filter_observation
from ..scenario import Scenario


def _failed_counts(scenario: Scenario, reason: str, **meta) -> tuple[np.ndarray, dict]:
    return np.zeros(scenario.num_codewords, dtype=np.float64), {
        "converged": False,
        "decoder_failure": True,
        "failure_reason": reason,
        **meta,
    }


def _prepare_linear_model(scenario: Scenario):
    if np.iscomplexobj(scenario.Y) or np.iscomplexobj(scenario.codebook):
        raise NotImplementedError("VAMP decoders currently support the real-valued setup only.")
    ybar, gamma_ant = matched_filter_observation(scenario.Y)
    y = np.asarray(np.real(ybar), dtype=np.float64)
    A = np.asarray(np.real(build_global_dictionary(scenario)), dtype=np.float64)
    sigma_eff_sq = max(estimate_noise_var(scenario.Y) / gamma_ant, 1e-12)
    U, sing, Vt = np.linalg.svd(A, full_matrices=False)
    return y, A, gamma_ant, sigma_eff_sq, U.T @ y, sing, Vt


def _count_moment_params(K: float, M: int) -> tuple[float, float, float]:
    """Map total device load to BG support probability and active second moment."""
    lam = max(float(K) / max(int(M), 1), 1e-12)
    rho = float(np.clip(1.0 - np.exp(-lam), 1e-8, 1.0 - 1e-8))
    sigma_x_sq = float(max((lam + lam * lam) / rho, 1e-8))
    return lam, rho, sigma_x_sq


def _initial_bg_params(y: np.ndarray, sigma_eff_sq: float, M: int) -> tuple[float, float, float]:
    """Energy-based non-oracle load initialisation, then convert to BG parameters."""
    E_sig = max(0.0, float(np.real(np.vdot(y, y))) - y.size * float(sigma_eff_sq))
    M_float = float(max(M, 1))
    K_hat = 0.5 * (-M_float + np.sqrt(M_float * M_float + 4.0 * M_float * E_sig))
    _, rho, sigma_x_sq = _count_moment_params(K_hat, M)
    return rho, sigma_x_sq, float(K_hat)


def _bg_denoise(r: np.ndarray, precision: float, rho: float,
                sigma_x_sq: float) -> tuple[np.ndarray, float, dict]:
    """Bernoulli-Gaussian denoiser and EM sufficient statistics."""
    gamma = max(float(precision), 1e-12)
    rho = float(np.clip(rho, 1e-8, 1.0 - 1e-8))
    sigma_x_sq = max(float(sigma_x_sq), 1e-12)

    active_var = 1.0 / (gamma + 1.0 / sigma_x_sq)
    active_mean = active_var * gamma * r
    logit = (
        np.log(rho) - np.log1p(-rho)
        - 0.5 * np.log1p(gamma * sigma_x_sq)
        + 0.5 * gamma * gamma * sigma_x_sq * r * r / (1.0 + gamma * sigma_x_sq)
    )
    p_act = 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))
    mean = p_act * active_mean
    second = p_act * (active_var + active_mean * active_mean)
    var = np.maximum(second - mean * mean, 0.0)
    alpha = float(np.clip(np.mean(gamma * var), 1e-6, 1.0 - 1e-6))
    return mean, alpha, {
        "p_act": p_act,
        "active_second": active_var + active_mean * active_mean,
    }


def _poisson_denoise(r: np.ndarray, precision: float, lam: float,
                     c_max: int | None) -> tuple[np.ndarray, float, dict]:
    """Discrete nonnegative Poisson-count denoiser on {0, ..., c_max}."""
    gamma = max(float(precision), 1e-12)
    lam = max(float(lam), 1e-12)
    if c_max is None:
        c_max = int(np.ceil(lam + 8.0 * np.sqrt(lam + 1.0) + 8.0))
    c_max = int(np.clip(c_max, 1, 500))
    c = np.arange(c_max + 1, dtype=np.float64)
    log_prior = c * np.log(lam) - gammaln(c + 1.0)
    log_w = log_prior[None, :] - 0.5 * gamma * (r[:, None] - c[None, :]) ** 2
    log_w -= np.max(log_w, axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= np.sum(w, axis=1, keepdims=True)
    mean = w @ c
    second = w @ (c * c)
    var = np.maximum(second - mean * mean, 0.0)
    alpha = float(np.clip(np.mean(gamma * var), 1e-6, 1.0 - 1e-6))
    return mean, alpha, {"second": second, "c_max": c_max}


def _linear_step(y: np.ndarray, Uy: np.ndarray, sing: np.ndarray, Vt: np.ndarray,
                 r2: np.ndarray, gamma2: float, sigma_eff_sq: float) -> tuple[np.ndarray, float, float]:
    M = r2.size
    rank = sing.size
    Vtr2 = Vt @ r2
    denom = sing * sing / sigma_eff_sq + gamma2
    rhs = sing * Uy / sigma_eff_sq + gamma2 * Vtr2
    x2_v = rhs / denom
    x2 = Vt.T @ x2_v + (r2 - Vt.T @ Vtr2)
    alpha2_raw = (float(np.sum(gamma2 / denom)) + (M - rank)) / M
    alpha2 = float(np.clip(alpha2_raw, 1e-6, 1.0 - 1e-6))
    trace_AcovAt = float(np.sum((sing * sing) / denom))
    return x2, alpha2, trace_AcovAt


def _run_vamp(
    scenario: Scenario,
    *,
    prior: str,
    oracle_k: bool,
    max_iter: int,
    tol: float,
    update_rho: bool = False,
    update_sigma_x: bool = False,
    update_lambda: bool = False,
    update_noise: bool = False,
    alpha_rho: float = 0.2,
    alpha_sigma_x: float = 0.1,
    alpha_lambda: float = 0.2,
    alpha_noise: float = 0.05,
    sigma_x_sq_init: float | None = None,
    rho_init: float | None = None,
    lambda_init: float | None = None,
    poisson_c_max: int | None = None,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    y, A, gamma_ant, sigma_eff_sq, Uy, sing, Vt = _prepare_linear_model(scenario)
    n, M = A.shape

    if oracle_k:
        lam0, rho0, sigma_x0 = _count_moment_params(scenario.num_devices_active, M)
    else:
        rho0, sigma_x0, K_init = _initial_bg_params(y, sigma_eff_sq, M)
        lam0 = max(K_init / max(M, 1), 1e-12)
    rho = float(np.clip(rho0 if rho_init is None else rho_init, 1e-8, 1.0 - 1e-8))
    sigma_x_sq = float(np.clip(sigma_x0 if sigma_x_sq_init is None else sigma_x_sq_init, 1e-8, 1e8))
    lam = float(max(lam0 if lambda_init is None else lambda_init, 1e-12))

    r1 = np.zeros(M, dtype=np.float64)
    if prior == "bg":
        gamma1 = 1.0 / max(sigma_x_sq, 1e-12)
    elif prior == "poisson":
        gamma1 = 1.0 / max(lam + lam * lam, 1e-12)
    else:
        raise ValueError(f"unknown VAMP prior {prior!r}")

    x_prev = np.zeros(M, dtype=np.float64)
    x_est = np.zeros(M, dtype=np.float64)
    history: list[dict] = []
    wall_start = time.time()
    converged = False
    timed_out = False

    for it in range(1, max_iter + 1):
        if max_wall_seconds is not None and time.time() - wall_start > max_wall_seconds:
            timed_out = True
            break

        if prior == "bg":
            x1, alpha1, stats = _bg_denoise(r1, gamma1, rho, sigma_x_sq)
        else:
            x1, alpha1, stats = _poisson_denoise(r1, gamma1, lam, poisson_c_max)
        gamma2 = max(gamma1 * alpha1 / (1.0 - alpha1), 1e-12)
        r2 = (x1 - alpha1 * r1) / (1.0 - alpha1)

        x2, alpha2, trace_AcovAt = _linear_step(y, Uy, sing, Vt, r2, gamma2, sigma_eff_sq)
        if not np.all(np.isfinite(x2)):
            return _failed_counts(scenario, "VAMP numerical divergence: nonfinite linear estimate.",
                                  iterations=len(history))
        gamma1_new = max(gamma2 * alpha2 / (1.0 - alpha2), 1e-12)
        r1_new = (x2 - alpha2 * r2) / (1.0 - alpha2)

        if prior == "bg":
            p_act = stats["p_act"]
            active_second = stats["active_second"]
            rho_em = float(np.clip(np.mean(p_act), 1e-8, 1.0 - 1e-8))
            sigma_x_em = float(np.sum(p_act * active_second) / max(float(np.sum(p_act)), 1e-12))
            sigma_x_em = float(np.clip(sigma_x_em, 1e-8, 1e8))
            if update_rho:
                rho = float(np.clip((1.0 - alpha_rho) * rho + alpha_rho * rho_em,
                                    1e-8, 1.0 - 1e-8))
            if update_sigma_x:
                sigma_x_sq = float(np.clip((1.0 - alpha_sigma_x) * sigma_x_sq
                                           + alpha_sigma_x * sigma_x_em, 1e-8, 1e8))
        else:
            lam_em = float(np.clip(np.mean(x1), 1e-12, 1e8))
            if update_lambda:
                lam = float(np.clip((1.0 - alpha_lambda) * lam + alpha_lambda * lam_em,
                                    1e-12, 1e8))

        residual = y - A @ x2
        rss = float(np.real(np.vdot(residual, residual)))
        sigma_eff_em = float(np.clip((rss + trace_AcovAt) / max(n, 1), 1e-12, 1e8))
        if update_noise:
            sigma_eff_sq = float((1.0 - alpha_noise) * sigma_eff_sq + alpha_noise * sigma_eff_em)

        delta = float(np.max(np.abs(x2 - x_prev)))
        x_prev = x2.copy()
        x_est = np.maximum(0.0, x2)
        counts = np.rint(x_est)
        hist = {
            "iter": it,
            "delta": delta,
            "residual": rss,
            "sigma_eff_sq": sigma_eff_sq,
            "sigma_eff_em": sigma_eff_em,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "K_hat": float(np.sum(counts)),
        }
        if prior == "bg":
            hist.update({
                "rho": rho,
                "rho_em": rho_em,
                "support_hat": rho * M,
                "sigma_x_sq": sigma_x_sq,
                "sigma_x_em": sigma_x_em,
            })
        else:
            hist.update({
                "lam": lam,
                "lambda_em": lam_em,
                "poisson_c_max": stats["c_max"],
            })
        history.append(hist)

        if verbose:
            print(f"  [iter {it:03d}] delta={delta:.3e} Khat={hist['K_hat']:.1f} "
                  f"seff2={sigma_eff_sq:.3e}", flush=True)
        if delta < tol:
            converged = True
            break
        r1 = r1_new
        gamma1 = gamma1_new

    counts = np.rint(np.maximum(0.0, x_est))
    meta = {
        "converged": converged,
        "timed_out": timed_out,
        "iterations": len(history),
        "history": history,
        "noise_var_est": float(sigma_eff_sq * gamma_ant),
        "sigma_eff_sq": float(sigma_eff_sq),
        "K_hat": float(np.sum(counts)),
        "wall_s": time.time() - wall_start,
    }
    if prior == "bg":
        meta.update({
            "rho_activity": float(rho),
            "support_hat": float(rho * M),
            "sigma_x_sq": float(sigma_x_sq),
        })
    else:
        meta.update({"lam": float(lam)})
    return counts, meta


def run_bg_oracle_k(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-5,
                    max_wall_seconds: float | None = None,
                    verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Oracle-K VAMP with BG prior parameters derived from the known load."""
    return _run_vamp(
        scenario, prior="bg", oracle_k=True, max_iter=max_iter, tol=tol,
        max_wall_seconds=max_wall_seconds, verbose=verbose)


def run_bg_em_rho(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-5,
                  max_wall_seconds: float | None = None,
                  verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Non-oracle VAMP-BG with EM activity updates and fixed noise/slab variance."""
    return _run_vamp(
        scenario, prior="bg", oracle_k=False, update_rho=True,
        update_sigma_x=False, update_noise=False, max_iter=max_iter, tol=tol,
        max_wall_seconds=max_wall_seconds, verbose=verbose)


def run_bg_em_rho_sigma(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-5,
                         max_wall_seconds: float | None = None,
                         verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Non-oracle VAMP-BG with EM activity and slab-variance updates."""
    return _run_vamp(
        scenario, prior="bg", oracle_k=False, update_rho=True,
        update_sigma_x=True, update_noise=False, max_iter=max_iter, tol=tol,
        max_wall_seconds=max_wall_seconds, verbose=verbose)


def run_bg_em_all(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-5,
                  max_wall_seconds: float | None = None,
                  verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Non-oracle VAMP-BG with EM activity, slab-variance, and noise updates."""
    return _run_vamp(
        scenario, prior="bg", oracle_k=False, update_rho=True,
        update_sigma_x=True, update_noise=True, max_iter=max_iter, tol=tol,
        max_wall_seconds=max_wall_seconds, verbose=verbose)


def run_poisson_em(scenario: Scenario, *, max_iter: int = 50, tol: float = 1e-5,
                   max_wall_seconds: float | None = None,
                   verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Non-oracle VAMP with a separable discrete Poisson count denoiser."""
    return _run_vamp(
        scenario, prior="poisson", oracle_k=False, update_lambda=True,
        update_noise=False, max_iter=max_iter, tol=tol,
        max_wall_seconds=max_wall_seconds, verbose=verbose)


run_bg = run_bg_oracle_k
