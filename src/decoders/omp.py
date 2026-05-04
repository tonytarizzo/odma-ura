"""Non-negative OMP with BIC stopping (NNOMP). No oracle."""

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


def run(scenario: Scenario, *, max_steps: int | None = None,
        bic_patience: int = 3) -> tuple[np.ndarray, dict]:
    Y = scenario.Y
    P_mats = scenario.P_mats
    codebook = scenario.codebook
    num_codewords = scenario.num_codewords
    block_to_msg_list = scenario.block_to_msg_list

    n, M_ant = Y.shape
    h = np.ones(M_ant, dtype=Y.dtype)
    gamma = float(np.real(np.vdot(h, h)))
    y_mf = Y @ h.conj() / gamma

    phi_dtype = np.complex128 if np.iscomplexobj(codebook) or np.iscomplexobj(Y) else np.float64
    Phi = np.zeros((n, num_codewords), dtype=phi_dtype)
    for b, msg_list in block_to_msg_list.items():
        P_b = P_mats[b]
        for m in msg_list:
            Phi[:, m] = P_b @ codebook[m]

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
