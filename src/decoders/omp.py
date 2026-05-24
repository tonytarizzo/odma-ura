"""Non-negative OMP decoders over the flattened ODMA dictionary."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ..scenario import Scenario


def _nnls_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        A_r = np.vstack([A.real, A.imag])
        b_r = np.concatenate([b.real, b.imag])
        x, _ = nnls(A_r, b_r)
    else:
        x, _ = nnls(A, b)
    return x


def _build_global_dictionary(scenario: Scenario) -> np.ndarray:
    codebook = scenario.codebook
    n = scenario.n
    num_codewords = scenario.num_codewords
    dtype = np.complex128 if np.iscomplexobj(codebook) or np.iscomplexobj(scenario.Y) else np.float64
    Phi = np.zeros((n, num_codewords), dtype=dtype)
    for b, msg_list in scenario.block_to_msg_list.items():
        P_b = scenario.P_mats[b]
        for m in msg_list:
            Phi[:, m] = P_b @ codebook[m]
    return Phi


def _matched_filter_y(scenario: Scenario) -> np.ndarray:
    Y = scenario.Y
    _, M_ant = Y.shape
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    return Y @ h.conj() / gamma


def _project_to_integer_total(x: np.ndarray, total: int) -> np.ndarray:
    """Project nonnegative amplitudes to integer counts with exact sum `total`."""
    if total < 0:
        raise ValueError(f"total must be nonnegative, got {total}")
    if x.size == 0:
        if total == 0:
            return np.zeros(0, dtype=np.float64)
        raise ValueError("cannot assign a positive total count to an empty support")
    if total == 0:
        return np.zeros_like(x, dtype=np.float64)

    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - float(total)
    idx = np.arange(1, x.size + 1, dtype=np.float64)
    active = u - cssv / idx > 0
    theta = cssv[np.nonzero(active)[0][-1]] / float(np.sum(active)) if np.any(active) else 0.0
    z = np.maximum(x - theta, 0.0)
    counts = np.floor(z).astype(np.float64)
    rem = int(total - np.sum(counts))
    if rem > 0:
        frac_order = np.argsort(-(z - counts), kind="mergesort")
        counts[frac_order[:rem]] += 1.0
    elif rem < 0:
        frac_order = np.argsort(z - counts, kind="mergesort")
        for i in frac_order:
            if rem == 0:
                break
            take = min(int(counts[i]), -rem)
            counts[i] -= take
            rem += take
    if int(np.sum(counts)) != total:
        raise RuntimeError("integer total projection failed")
    return counts


def run(scenario: Scenario, *, max_steps: int | None = None,
        bic_patience: int = 3) -> tuple[np.ndarray, dict]:
    num_codewords = scenario.num_codewords
    n = scenario.n
    y_mf = _matched_filter_y(scenario)
    Phi = _build_global_dictionary(scenario)

    residual = y_mf.copy()
    support: list[int] = []
    used = np.zeros(num_codewords, dtype=bool)
    best_support: list[int] = []
    best_x = np.zeros(0, dtype=np.float64)
    best_bic = float("inf"); worse_count = 0
    max_k = min(n, num_codewords) if max_steps is None else min(max_steps, n, num_codewords)

    for _ in range(max_k):
        corrs = np.real(Phi.conj().T @ residual)
        corrs[used] = -np.inf
        best_m = int(np.argmax(corrs))
        if not np.isfinite(corrs[best_m]) or corrs[best_m] <= 0.0:
            break
        support.append(best_m); used[best_m] = True
        Phi_s = Phi[:, support]
        x_nn = _nnls_solve(Phi_s, y_mf)
        residual = y_mf - Phi_s @ x_nn
        rss = max(float(np.real(np.vdot(residual, residual))), 1e-12)
        k = len(support)
        bic = n * np.log(rss / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic; best_support = support.copy(); best_x = x_nn.copy()
            worse_count = 0
        else:
            worse_count += 1
            if worse_count >= bic_patience:
                break

    counts = np.zeros(num_codewords, dtype=np.float64)
    for i, m in enumerate(best_support):
        counts[m] = max(0.0, round(float(best_x[i])))
    return counts, {"selected_k": len(best_support), "best_bic": best_bic}


def run_oracle_k(scenario: Scenario, *, max_steps: int | None = None) -> tuple[np.ndarray, dict]:
    """Oracle-K NNOMP with integer projection onto sum(a)=K_a."""
    num_codewords = scenario.num_codewords
    n = scenario.n
    K = int(scenario.num_devices_active)
    y_mf = _matched_filter_y(scenario)
    Phi = _build_global_dictionary(scenario)

    residual = y_mf.copy()
    support: list[int] = []
    used = np.zeros(num_codewords, dtype=bool)
    max_k = min(K if max_steps is None else int(max_steps), n, num_codewords)

    x_nn = np.zeros(0, dtype=np.float64)
    for _ in range(max_k):
        corrs = np.real(Phi.conj().T @ residual)
        corrs[used] = -np.inf
        best_m = int(np.argmax(corrs))
        if not np.isfinite(corrs[best_m]):
            break
        support.append(best_m); used[best_m] = True
        Phi_s = Phi[:, support]
        x_nn = _nnls_solve(Phi_s, y_mf)
        residual = y_mf - Phi_s @ x_nn

    counts = np.zeros(num_codewords, dtype=np.float64)
    if support:
        projected = _project_to_integer_total(x_nn, K)
        for i, m in enumerate(support):
            counts[m] = projected[i]
    return counts, {
        "selected_k": len(support),
        "K_target": K,
        "K_hat": float(np.sum(counts)),
        "projection": "nonnegative_integer_total",
    }
