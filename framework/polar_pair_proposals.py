"""Bounded same-pattern polar pair lists; these are candidates, not posterior samples.

The joint search uses exact local GF(2)^2 kernels and ordinary finite-list pruning.
This NumPy prototype copies recursive path arrays; it does not implement lazy path
sharing. For alphabet size q=4, conservative costs are O(q^2 L d log d + q L d
log(q L)) arithmetic and O(q L d log d) temporary storage. Neither search builds
the payload alphabet. Physical scores cover the supplied pattern coordinates;
the caller must restore unoccupied-coordinate likelihoods when comparing patterns.
"""

from __future__ import annotations

from copy import copy

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode, append_crc


def _inputs(code, residual, amplitudes, noise_var, *sizes):
    if not isinstance(code, PolarCode):
        raise TypeError("code must be a PolarCode.")
    if np.iscomplexobj(residual) or np.iscomplexobj(amplitudes):
        raise ValueError("Same-pattern pair channels require real residuals and amplitudes.")
    residual, amplitudes = np.asarray(residual, dtype=float), np.asarray(amplitudes, dtype=float)
    if residual.shape != (code.n,) or amplitudes.shape != (code.n,):
        raise ValueError("Require one residual and amplitude per polar coded coordinate.")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
        raise ValueError("Require finite residuals and finite nonnegative amplitudes.")
    if not np.isscalar(noise_var) or np.iscomplexobj(noise_var) or not np.isfinite(noise_var) or noise_var <= 0:
        raise ValueError("noise_var must be finite and strictly positive.")
    if any(not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)) or size < 1 for size in sizes):
        raise ValueError("List sizes must be positive integers.")
    return residual, amplitudes, float(noise_var)


def _rank_pairs(code, pairs, residual, amplitudes, noise_var):
    if not len(pairs):
        return np.empty((0, 2, code.payload_bits), dtype=np.uint8), np.empty(0), np.empty(0, dtype=int)
    pairs, first = np.unique(pairs, axis=0, return_index=True)
    signs = 1.0 - 2.0 * code.encode(pairs)
    signals = amplitudes * signs.sum(axis=1)
    log_weights = -np.square(residual - signals).sum(axis=1) / (2.0 * noise_var)
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("Pair likelihood overflowed; residual/amplitude/noise scale is not representable.")
    order = np.argsort(-log_weights, kind="stable")
    return pairs[order], log_weights[order], first[order]


def _joint_scl(code, log_channels, list_size):
    symbols = np.arange(4, dtype=np.uint8)
    xor = symbols[:, None] ^ symbols[None, :]
    stats = {"expanded_symbols": 0, "peak_paths": 1}

    def visit(alpha, metrics, start):
        paths, width = alpha.shape[:2]
        if width == 1:
            costs = logsumexp(alpha[:, 0], axis=-1)[:, None] - alpha[:, 0]
            if code.frozen[start]:
                bits = np.zeros((paths, 1), dtype=np.uint8)
                stats["expanded_symbols"] += paths
                return bits, bits, metrics + costs[:, 0], np.arange(paths)
            extended = metrics[:, None] + costs
            selected = np.argsort(extended.ravel(), kind="stable")[:list_size]
            bits = (selected % 4).astype(np.uint8)[:, None]
            stats["expanded_symbols"] += 4 * paths
            stats["peak_paths"] = max(stats["peak_paths"], len(selected))
            return bits, bits, extended.ravel()[selected], selected // 4
        half = width // 2
        a, b = alpha[:, :half], alpha[:, half:]
        minus = logsumexp(a[..., xor] + b[..., None, :], axis=-1)
        minus -= logsumexp(minus, axis=-1, keepdims=True)
        left, left_u, left_metrics, origins = visit(minus, metrics, start)
        a, b = a[origins], b[origins]
        plus_indices = left[..., None] ^ symbols
        plus = np.take_along_axis(a, plus_indices, axis=-1) + b
        plus -= logsumexp(plus, axis=-1, keepdims=True)
        right, right_u, final_metrics, right_origins = visit(plus, left_metrics, start + half)
        beta = np.concatenate((left[right_origins] ^ right, right), axis=1)
        u = np.concatenate((left_u[right_origins], right_u), axis=1)
        return beta, u, final_metrics, origins[right_origins]

    _, u, metrics, _ = visit(log_channels[None], np.zeros(1), 0)
    return u, metrics, stats


def joint_pair_list(code: PolarCode, residual, amplitudes, noise_var, list_size):
    """Return CRC-valid ordered payload pairs from a four-symbol polar SCL search.

    At a common frozen position only symbol (0, 0) is allowed; information
    positions branch over all four symbols. Both terminal CRCs must pass.
    There are at most list_size results, and CRC failure returns an empty list.
    """
    residual, amplitudes, noise_var = _inputs(code, residual, amplitudes, noise_var, list_size)
    sums = np.array([2.0, 0.0, 0.0, -2.0])
    log_channels = ((amplitudes * residual)[:, None] * sums - .5 * amplitudes[:, None] ** 2 * sums ** 2) / noise_var
    if not np.all(np.isfinite(log_channels)):
        raise ValueError("Pair channel overflowed; residual/amplitude/noise scale is not representable.")
    log_channels -= logsumexp(log_channels, axis=1, keepdims=True)
    u, metrics, stats = _joint_scl(code, log_channels, int(list_size))
    data = np.stack((u >> 1, u & 1), axis=1)[:, :, code.information]
    payloads = data[:, :, :code.payload_bits]
    valid = np.all(append_crc(payloads, code.crc_bits) == data, axis=(1, 2))
    pairs, weights, selected = _rank_pairs(code, payloads[valid], residual, amplitudes, noise_var)
    return pairs, {"proposal": "joint_four_symbol_scl", "list_size": int(list_size), "terminal_paths": len(u),
                   "crc_valid_paths": int(valid.sum()), "candidate_count": len(pairs), "log_weights": weights,
                   "path_metrics": metrics[valid][selected], "array_storage": "recursive_copy_no_lazy_sharing", **stats}


def _binary_list(code, llr, size):
    decoder = copy(code)
    decoder.list_size = int(size)
    return decoder.decode_list(llr)[0]


def xor_pair_list(code: PolarCode, residual, amplitudes, noise_var, difference_size, payload_size):
    """Return bounded XOR-first ordered pair candidates with exact physical scores.

    The difference guide omits the code-consistency partition factor. Physical
    rescoring corrects scores only for retained pairs, not this discovery loss.
    Include both orders explicitly, deduplicate, and keep repeats when delta=0.
    Output has at most 2*difference_size*payload_size pairs; empty CRC lists stay empty.
    """
    residual, amplitudes, noise_var = _inputs(code, residual, amplitudes, noise_var, difference_size, payload_size)
    theta = 2.0 * amplitudes * residual / noise_var
    gamma = np.logaddexp(theta, -theta) - np.log(2.0) - 2.0 * amplitudes ** 2 / noise_var
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(theta)):
        raise ValueError("XOR channel overflowed; residual/amplitude/noise scale is not representable.")
    differences = _binary_list(code, gamma, difference_size)
    candidates, conditional_paths = [], 0
    coded_differences = code.encode(differences) if len(differences) else np.empty((0, code.n), dtype=np.uint8)
    for difference, coded in zip(differences, coded_differences):
        payloads = _binary_list(code, 2.0 * theta * (1 - coded), payload_size)
        conditional_paths += len(payloads)
        other = payloads ^ difference
        candidates.extend((np.stack((payloads, other), axis=1), np.stack((other, payloads), axis=1)))
    raw = np.concatenate(candidates, axis=0) if candidates else np.empty((0, 2, code.payload_bits), dtype=np.uint8)
    pairs, weights, _ = _rank_pairs(code, raw, residual, amplitudes, noise_var)
    return pairs, {"proposal": "xor_first_scl", "difference_size": int(difference_size), "payload_size": int(payload_size),
                   "difference_count": len(differences), "conditional_candidate_count": conditional_paths,
                   "candidate_count": len(pairs), "log_weights": weights, "guide_omits_code_consistency": True,
                   "array_storage": "recursive_copy_no_lazy_sharing"}
