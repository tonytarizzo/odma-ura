"""Codebook construction and ODMA block / message bookkeeping (V2 case)."""

from __future__ import annotations

import numpy as np


def make_codebook(num_codewords: int, d: int, rng: np.random.Generator,
                  complex_valued: bool = False) -> np.ndarray:
    """Random Gaussian codebook with unit-normalised rows. Returns (num_codewords, d)."""
    if complex_valued:
        raw = (rng.standard_normal((num_codewords, d))
               + 1j * rng.standard_normal((num_codewords, d)))
    else:
        raw = rng.standard_normal((num_codewords, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_odma_blocks(num_blocks: int, n: int, d: int,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Random ODMA blocks: each is a sorted array of d resource indices from {0,...,n-1}."""
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(num_blocks)]


def build_pattern_matrices(blocks: list[np.ndarray], n: int) -> dict[int, np.ndarray]:
    """Per-block embedding matrices P_b in {0,1}^{n x d}.  P_b^T P_b = I_d."""
    P_mats: dict[int, np.ndarray] = {}
    for b, S_b in enumerate(blocks):
        d = len(S_b)
        P = np.zeros((n, d), dtype=np.float64)
        P[S_b, np.arange(d)] = 1.0
        P_mats[b] = P
    return P_mats


def make_message_block_mapping(num_codewords: int, num_blocks: int):
    """Deterministic mapping: message m -> block (m mod num_blocks)."""
    msg_to_block: dict[int, int] = {m: m % num_blocks for m in range(num_codewords)}
    block_to_msg_list: dict[int, list[int]] = {b: [] for b in range(num_blocks)}
    for m in range(num_codewords):
        block_to_msg_list[m % num_blocks].append(m)
    return msg_to_block, block_to_msg_list


def sample_active_messages(num_devices_active: int, num_codewords: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Each active device picks a message uniformly at random."""
    return rng.integers(0, num_codewords, size=num_devices_active)


def build_message_counts(active_msgs: np.ndarray, num_codewords: int) -> np.ndarray:
    """Global message-count vector — the direct decoder target."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for m in active_msgs:
        counts[int(m)] += 1.0
    return counts


def build_block_coefficients(active_msgs: np.ndarray,
                              msg_to_block: dict[int, int],
                              block_to_msg_list: dict[int, list[int]],
                              num_blocks: int) -> dict[int, np.ndarray]:
    """Blockwise sparse multiplicity vectors a_b."""
    coeffs: dict[int, np.ndarray] = {}
    for b in range(num_blocks):
        msg_list = block_to_msg_list[b]
        a_b = np.zeros(len(msg_list), dtype=np.float64)
        msg_to_local = {m: i for i, m in enumerate(msg_list)}
        for m in active_msgs:
            if msg_to_block[m] == b:
                a_b[msg_to_local[m]] += 1.0
        coeffs[b] = a_b
    return coeffs


def build_block_dictionaries(codebook: np.ndarray,
                              block_to_msg_list: dict[int, list[int]],
                              num_blocks: int) -> dict[int, np.ndarray]:
    """Codebook rows for each block's assigned messages.  block -> (L_b, d)."""
    return {b: codebook[block_to_msg_list[b]] for b in range(num_blocks)}
