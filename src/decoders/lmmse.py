"""LMMSE decoders (oracle: K + user->block assignment).

Three pattern-aware variants from the V2 derivation:
  - approach 2: ignores ODMA pattern structure (rank-1 degenerate with h=1_M)
  - approach 3: per-resource exact-TIN scalar LMMSE (interference-aware)
  - approach 4: full joint vectorisation LMMSE
"""

from __future__ import annotations

import numpy as np

from ..metrics import nn_votes_to_counts
from ..scenario import Scenario


def _lmmse2(s: Scenario) -> np.ndarray:
    K = len(s.active_msgs)
    n, M_ant = s.Y.shape
    dtype = s.Y.dtype
    d = s.codebook.shape[1]
    p_d = 1.0 / d
    h = np.ones(M_ant, dtype=dtype)
    H = np.tile(h[None, :], (K, 1))
    HtH_M = p_d * (H.conj().T @ H)
    reg_M = s.noise_var * np.eye(M_ant, dtype=dtype)
    inv_M = np.linalg.inv(HtH_M + reg_M)
    X_prime_hat = p_d * (s.Y @ inv_M @ H.conj().T)
    x_hat_list = []
    for k, m in enumerate(s.active_msgs):
        b = s.msg_to_block[int(m)]
        x_hat_list.append(s.P_mats[b].T @ X_prime_hat[:, k])
    return nn_votes_to_counts(x_hat_list, s.codebook, s.num_codewords)


def _lmmse3(s: Scenario) -> np.ndarray:
    K = len(s.active_msgs)
    n, M_ant = s.Y.shape
    dtype = s.Y.dtype
    d = s.codebook.shape[1]
    p_d = 1.0 / d
    h = np.ones(M_ant, dtype=dtype)
    gamma = float(np.real(np.vdot(h, h)))
    device_blocks = [s.msg_to_block[int(m)] for m in s.active_msgs]
    device_S = [np.argmax(s.P_mats[b], axis=0).astype(int) for b in device_blocks]

    users_per_resource = np.zeros(n, dtype=np.int64)
    for S_k in device_S:
        users_per_resource[S_k] += 1

    Y_h = s.Y @ h.conj()
    x_hat_list = []
    for k in range(K):
        S_k = device_S[k]
        q_ki = users_per_resource[S_k] - 1
        denom = s.noise_var / p_d + gamma * (1 + q_ki)
        x_hat_list.append(Y_h[S_k] / denom)
    return nn_votes_to_counts(x_hat_list, s.codebook, s.num_codewords)


def _lmmse4(s: Scenario) -> np.ndarray:
    K = len(s.active_msgs)
    n, M_ant = s.Y.shape
    dtype = s.Y.dtype
    d = s.codebook.shape[1]
    p_d = 1.0 / d
    h = np.ones(M_ant, dtype=dtype)
    device_blocks = [s.msg_to_block[int(m)] for m in s.active_msgs]
    A = np.zeros((n * M_ant, K * d), dtype=dtype)
    for k in range(K):
        P_k = s.P_mats[device_blocks[k]]
        block_k = (P_k[:, :, None] * h[None, None, :])
        A[:, k * d:(k + 1) * d] = block_k.transpose(0, 2, 1).reshape(n * M_ant, d)
    y = s.Y.reshape(-1)
    reg = (s.noise_var / p_d) * np.eye(K * d, dtype=dtype)
    AhA = A.conj().T @ A
    x_hat_vec = np.linalg.solve(AhA + reg, A.conj().T @ y)
    x_hat_list = [x_hat_vec[k * d:(k + 1) * d] for k in range(K)]
    return nn_votes_to_counts(x_hat_list, s.codebook, s.num_codewords)


def run_2(scenario: Scenario) -> tuple[np.ndarray, dict]:
    return _lmmse2(scenario), {}


def run_3(scenario: Scenario) -> tuple[np.ndarray, dict]:
    return _lmmse3(scenario), {}


def run_4(scenario: Scenario, *, max_kd: int = 800) -> tuple[np.ndarray, dict]:
    """Joint LMMSE-4. Skipped (raises) when K*d exceeds max_kd to avoid OOM."""
    if scenario.num_devices_active * scenario.d > max_kd:
        raise RuntimeError(
            f"LMMSE-4 skipped: K*d = {scenario.num_devices_active * scenario.d} > {max_kd}")
    return _lmmse4(scenario), {}
