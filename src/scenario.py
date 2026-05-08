"""Scenario dataclass — bundles all per-trial data passed into decoders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codebook import (
    build_block_coefficients,
    build_block_dictionaries,
    build_message_counts,
    build_pattern_matrices,
    make_codebook,
    make_message_block_mapping,
    make_odma_blocks,
    sample_active_messages,
)
from .signal import esn0_db_to_noise_var, synthesize_received_signal


SCENARIO_KEYS: tuple[str, ...] = (
    "n", "d", "num_blocks", "num_codewords",
    "num_devices_active", "num_antennas", "esn0_db",
)


@dataclass
class Scenario:
    """One realisation of the V2 ODMA+URA setup, plus all derived data."""

    n: int
    d: int
    num_blocks: int
    num_codewords: int
    num_devices_active: int
    num_antennas: int
    esn0_db: float
    seed: int

    codebook: np.ndarray
    P_mats: dict[int, np.ndarray]
    block_dicts: dict[int, np.ndarray]
    msg_to_block: dict[int, int]
    block_to_msg_list: dict[int, list[int]]
    active_msgs: np.ndarray
    message_counts: np.ndarray
    block_coeffs: dict[int, np.ndarray]
    Y: np.ndarray
    Y_clean: np.ndarray
    noise_var: float

    @property
    def config_dict(self) -> dict:
        """Stable dict identifying the scenario (no derived arrays). Used for cache hashing."""
        return {k: getattr(self, k) for k in SCENARIO_KEYS}


def build_scenario(*, n: int, d: int, num_blocks: int, num_codewords: int,
                   num_devices_active: int, num_antennas: int, esn0_db: float,
                   seed: int) -> Scenario:
    """Construct a Scenario from scalar parameters and a seed.

    The V2 common-signature model assumes M_ant >= 2 (the orthogonal antenna
    subspace is needed for non-oracle noise estimation), so single-antenna
    setups are rejected at scenario build time.
    """
    if int(num_antennas) < 2:
        raise ValueError(
            f"num_antennas must be >= 2 for the V2 common-signature model "
            f"(got {num_antennas}); single-antenna runs are not supported.")
    rng = np.random.default_rng(seed)

    codebook = make_codebook(num_codewords, d, rng)
    blocks = make_odma_blocks(num_blocks, n, d, rng)
    P_mats = build_pattern_matrices(blocks, n)
    msg_to_block, block_to_msg_list = make_message_block_mapping(num_codewords, num_blocks)
    block_dicts = build_block_dictionaries(codebook, block_to_msg_list, num_blocks)

    active_msgs = sample_active_messages(num_devices_active, num_codewords, rng)
    message_counts = build_message_counts(active_msgs, num_codewords)
    block_coeffs = build_block_coefficients(
        active_msgs, msg_to_block, block_to_msg_list, num_blocks)

    noise_var = esn0_db_to_noise_var(esn0_db, d)
    Y, Y_clean = synthesize_received_signal(
        P_mats, block_dicts, block_coeffs, num_antennas, noise_var, rng)

    return Scenario(
        n=n, d=d, num_blocks=num_blocks, num_codewords=num_codewords,
        num_devices_active=num_devices_active, num_antennas=num_antennas,
        esn0_db=float(esn0_db), seed=int(seed),
        codebook=codebook, P_mats=P_mats, block_dicts=block_dicts,
        msg_to_block=msg_to_block, block_to_msg_list=block_to_msg_list,
        active_msgs=active_msgs, message_counts=message_counts,
        block_coeffs=block_coeffs,
        Y=Y, Y_clean=Y_clean, noise_var=noise_var,
    )
