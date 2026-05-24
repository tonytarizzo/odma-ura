"""Signal synthesis and SNR conversions for the V2 model (h = 1_M)."""

from __future__ import annotations

import numpy as np


def esn0_db_to_noise_var(esn0_db: float, d: int) -> float:
    """Es/N0 (dB) -> per-entry noise variance.  Es = 1/d (unit-norm codewords)."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    esn0_lin = 10.0 ** (esn0_db / 10.0)
    return 1.0 / (d * esn0_lin)


def esn0_db_to_ebn0_db(esn0_db: float, d: int, num_codewords: int) -> float:
    """Convert internal Es/N0 control to GMAC-style Eb/N0 (no real-AWGN 1/2 factor)."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if num_codewords < 2:
        raise ValueError(f"num_codewords must be at least 2, got {num_codewords}")
    bits_per_msg = np.log2(num_codewords)
    return esn0_db + 10.0 * np.log10(d / bits_per_msg)


def ebn0_db_to_esn0_db(ebn0_db: float, d: int, num_codewords: int) -> float:
    """Inverse of esn0_db_to_ebn0_db for the current unit-norm local-codeword convention."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if num_codewords < 2:
        raise ValueError(f"num_codewords must be at least 2, got {num_codewords}")
    bits_per_msg = np.log2(num_codewords)
    return ebn0_db - 10.0 * np.log10(d / bits_per_msg)


def synthesize_received_signal(P_mats: dict[int, np.ndarray],
                                block_dicts: dict[int, np.ndarray],
                                block_coeffs: dict[int, np.ndarray],
                                num_antennas: int,
                                noise_var: float,
                                rng: np.random.Generator,
                                complex_valued: bool = False,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise Y = outer(y_scalar, h) + Z with h = 1_M.

    Returns (Y_noisy, Y_clean), both shape (n, num_antennas).
    """
    n = next(iter(P_mats.values())).shape[0]
    dtype = np.complex128 if complex_valued else np.float64
    y_scalar = np.zeros(n, dtype=dtype)
    for b in P_mats:
        a_b = block_coeffs[b]
        if np.any(a_b):
            x_b = block_dicts[b].T @ a_b
            y_scalar += P_mats[b] @ x_b
    h = np.ones(num_antennas, dtype=dtype)
    Y_clean = np.outer(y_scalar, h)
    if complex_valued:
        noise = np.sqrt(noise_var / 2) * (
            rng.standard_normal((n, num_antennas))
            + 1j * rng.standard_normal((n, num_antennas)))
    else:
        noise = np.sqrt(noise_var) * rng.standard_normal((n, num_antennas))
    return Y_clean + noise, Y_clean
