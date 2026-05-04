"""SIC decoder — global single-pass MF cancellation with BIC stopping. No oracle."""

from __future__ import annotations

import numpy as np

from ..scenario import Scenario


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
    path_msgs: list[int] = []
    path_amps: list[int] = []
    used = np.zeros(num_codewords, dtype=bool)
    best_bic = float("inf"); best_k = 0; worse_count = 0
    max_k = min(n, num_codewords) if max_steps is None else min(max_steps, n, num_codewords)

    for k_step in range(1, max_k + 1):
        score = np.real(Phi.conj().T @ residual)
        score[used] = -np.inf
        best_m = int(np.argmax(score))
        if not np.isfinite(score[best_m]) or score[best_m] <= 0.0:
            break
        amp = max(1, int(round(score[best_m])))
        residual = residual - amp * Phi[:, best_m]
        used[best_m] = True
        path_msgs.append(best_m); path_amps.append(amp)
        rss = max(float(np.real(np.vdot(residual, residual))), 1e-12)
        bic = n * np.log(rss / n) + k_step * np.log(n)
        if bic < best_bic:
            best_bic = bic; best_k = k_step; worse_count = 0
        else:
            worse_count += 1
            if worse_count >= bic_patience:
                break

    counts = np.zeros(num_codewords, dtype=np.float64)
    for m, a in zip(path_msgs[:best_k], path_amps[:best_k]):
        counts[m] += a
    return counts, {"selected_k": best_k, "best_bic": best_bic}
