"""Published Gaussian-MAC receiver chains, kept separate from the learned framework."""

import numpy as np


def make_baseline(name, payload_bits, n, seed=0, **params):
    if name == "odma_polar":
        from .odma_polar import ODMAPolarBaseline
        return ODMAPolarBaseline(payload_bits, n, seed, **params)
    if name == "dynamic_cs":
        from .dynamic_cs import DynamicCSBaseline
        return DynamicCSBaseline(payload_bits, n, seed, **params)
    if name in {"ccs_amp", "ccs_block"}:
        from .ccs_amp import CCSAMPBaseline
        return CCSAMPBaseline(payload_bits, n, seed, inner_mode="dense" if name == "ccs_amp" else "block_diagonal", **params)
    raise ValueError(f"Unknown published baseline {name!r}")


def message_bits(indices, width):
    if not 1 <= width <= 62:
        raise ValueError("Integer-index conversion is only for small alphabets; use bit arrays at large B")
    return ((np.asarray(indices, dtype=np.int64)[..., None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)


def message_indices(bits):
    bits = np.asarray(bits)
    if bits.shape[-1] > 62:
        raise ValueError("Use bit-array message identities above 62 bits")
    return bits.astype(np.int64) @ (1 << np.arange(bits.shape[-1] - 1, -1, -1))


def run(scenario, *, baseline):
    """Small-B ``run(scenario, **params) -> (counts, meta)`` adapter.

    Only Y, noise_var and known activity are read; truth/support fields are never read.
    The supplied scenario must contain this baseline's real, unfaded signal, not the
    legacy multi-antenna ODMA scenario. Large-B callers use decode() and bit lists.
    """
    y = np.asarray(scenario.Y)
    if y.shape == (baseline.n, 1):
        y = y[:, 0]
    if baseline.payload_bits > 20:
        raise ValueError("A global counts return is deliberately limited to B<=20; use the bit-list decode API")
    decoded, meta = baseline.decode(y, scenario.num_devices_active, scenario.noise_var)
    counts = np.zeros(1 << baseline.payload_bits, dtype=np.int64)
    counts[message_indices(decoded)] = 1
    return counts, {**meta, "count_semantics": "message presence, not estimated multiplicity", "baseline": baseline.metadata()}
