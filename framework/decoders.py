"""Global decoders over the abstract Encoder interface.

The default reference is `oracle_k_omp`: a non-negative OMP that knows the true
number of active users K_a and projects the final amplitudes onto integer
counts summing to K_a. It is intentionally global - it consumes only Phi-linear
operations and the explicit matrix, so the same routine works for dense,
ODMA-product, and arbitrary factorised codebooks.

Decoders are registered in `DECODERS` so the pipeline can pick one by name.
The signature is `fn(encoder, Y, H, *, num_active, ...) -> DecoderOutput`.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import torch
from scipy.optimize import nnls

from .channel import matched_filter_collapse
from .core import DecoderOutput
from .encoder import Encoder


def project_nonneg_integer_total(x: torch.Tensor, total: int) -> torch.Tensor:
    """Project x >= 0 onto integer counts with exact sum = total."""
    if total < 0:
        raise ValueError(f"total must be nonnegative, got {total}")
    x = torch.clamp(x.detach().to(torch.float64), min=0.0)
    if x.numel() == 0:
        if total == 0:
            return x
        raise ValueError("cannot assign a positive total count to an empty support")
    if total == 0:
        return torch.zeros_like(x)
    u, _ = torch.sort(x, descending=True, stable=True)
    cssv = torch.cumsum(u, dim=0) - float(total)
    idx = torch.arange(1, x.numel() + 1, dtype=x.dtype, device=x.device)
    active = u - cssv / idx > 0
    if bool(active.any()):
        last = int(torch.nonzero(active, as_tuple=False)[-1, 0].item())
        theta = cssv[last] / float(int(active.sum().item()))
    else:
        theta = x.new_tensor(0.0)
    z = torch.clamp(x - theta, min=0.0)
    counts = torch.floor(z)
    rem = total - int(counts.sum().item())
    if rem > 0:
        order = torch.argsort(z - counts, descending=True, stable=True)
        counts[order[:rem]] += 1.0
    elif rem < 0:
        order = torch.argsort(z - counts, descending=False, stable=True)
        for i in order.tolist():
            if rem == 0:
                break
            take = min(int(counts[i].item()), -rem)
            counts[i] -= take
            rem += take
    if int(counts.sum().item()) != total:
        raise RuntimeError("integer total projection failed")
    return counts


def active_count_vector(num_active: int | torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Normalise scalar or per-sample K_a into a length-B integer tensor."""
    K = torch.as_tensor(num_active, dtype=torch.long, device=device)
    if K.ndim == 0:
        K = K.repeat(batch_size)
    if K.shape != (batch_size,):
        raise ValueError(f"num_active must be scalar or shape ({batch_size},), got {tuple(K.shape)}")
    if bool(torch.any(K <= 0)):
        raise ValueError(f"num_active entries must be positive, got {K.tolist()}")
    return K


def solve_nonnegative_least_squares(Phi_s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Non-negative least squares via scipy. Real/complex inputs are stacked."""
    A = Phi_s.detach().cpu().numpy()
    b = y.detach().cpu().numpy()
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        A = np.vstack([A.real, A.imag])
        b = np.concatenate([b.real, b.imag])
    x, _ = nnls(A, b)
    return torch.as_tensor(x, dtype=torch.float64, device=Phi_s.device)


def oracle_k_omp_single(Phi: torch.Tensor, y: torch.Tensor, K: int) -> tuple[torch.Tensor, list[int]]:
    n, M = Phi.shape
    residual = y
    support: list[int] = []
    used = torch.zeros(M, dtype=torch.bool, device=Phi.device)
    x_s = torch.zeros(0, dtype=Phi.dtype, device=Phi.device)
    max_steps = min(K, n, M)
    for _ in range(max_steps):
        corrs = (Phi.conj().transpose(-1, -2) @ residual).real
        corrs = corrs.masked_fill(used, float("-inf"))
        best = int(torch.argmax(corrs).item())
        if not bool(torch.isfinite(corrs[best])) or corrs[best].item() <= 0.0:
            break
        support.append(best); used[best] = True
        Phi_s = Phi[:, support]
        x_s = solve_nonnegative_least_squares(Phi_s, y)
        residual = y - Phi_s @ x_s.to(Phi.dtype)
    return x_s, support


def oracle_k_omp(encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                  num_active: int | torch.Tensor, noise_var: float | torch.Tensor | None = None) -> DecoderOutput:
    """Oracle-K non-negative OMP over the multi-antenna observation.

    Y has shape (B, n, M_ant); the routine matched-filters with the known H
    to collapse to a (B, n) signal and then runs scalar NNOMP.
    """
    y_mf = matched_filter_collapse(Y, H)
    Phi = encoder.explicit_matrix().detach()
    B, M = y_mf.shape[0], encoder.num_codewords
    K_vec = active_count_vector(num_active, B, Y.device)
    counts = torch.zeros(B, M, dtype=torch.float64, device=Y.device)
    supports: list[list[int]] = []
    for b in range(B):
        K = int(K_vec[b].item())
        x_s, support = oracle_k_omp_single(Phi, y_mf[b], K)
        if support:
            projected = project_nonneg_integer_total(x_s, K)
            counts[b, torch.tensor(support, dtype=torch.long, device=Y.device)] = projected
        supports.append(support)
    return DecoderOutput(counts=counts, meta={
        "decoder": "oracle_k_omp",
        "K_target": K_vec.detach().cpu().tolist(),
        "supports": supports,
    })


def exact_count_ml(encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                   num_active: int | torch.Tensor, noise_var: float | torch.Tensor | None = None,
                   max_hypotheses: int = 250_000, chunk_size: int = 4096) -> DecoderOutput:
    """Exact oracle-K count ML for tiny certification problems, including collisions.

    The hypothesis count is C(M+K-1,K), so this deliberately refuses scalable
    settings. Candidate signals are encoded through the implicit operator.
    """
    y = matched_filter_collapse(Y, H)
    K_vec = active_count_vector(num_active, y.shape[0], y.device)
    counts = torch.zeros(y.shape[0], encoder.num_codewords, dtype=torch.float64, device=y.device)
    tested: dict[int, int] = {}
    for K in torch.unique(K_vec, sorted=True).tolist():
        hypotheses = math.comb(encoder.num_codewords + int(K) - 1, int(K))
        if hypotheses > int(max_hypotheses):
            raise ValueError(f"exact_count_ml requires {hypotheses} hypotheses for M={encoder.num_codewords}, K={K}; "
                             f"limit is {max_hypotheses}")
        rows = torch.nonzero(K_vec == int(K), as_tuple=False).flatten()
        best_error = torch.full((rows.numel(),), float("inf"), dtype=y.real.dtype, device=y.device)
        best_counts = torch.zeros(rows.numel(), encoder.num_codewords, dtype=torch.float64, device=y.device)
        iterator = itertools.combinations_with_replacement(range(encoder.num_codewords), int(K))
        while True:
            combinations = list(itertools.islice(iterator, int(chunk_size)))
            if not combinations:
                break
            candidate = torch.zeros(len(combinations), encoder.num_codewords, dtype=encoder.dtype, device=y.device)
            indices = torch.tensor(combinations, dtype=torch.long, device=y.device)
            candidate.scatter_add_(1, indices, torch.ones_like(indices, dtype=encoder.dtype))
            signals = encoder.matvec(candidate)
            error = torch.sum(torch.abs(y.index_select(0, rows).unsqueeze(1) - signals.unsqueeze(0)) ** 2, dim=2).real
            chunk_error, chunk_index = torch.min(error, dim=1)
            improved = chunk_error < best_error
            best_error = torch.where(improved, chunk_error, best_error)
            best_counts[improved] = candidate.index_select(0, chunk_index[improved]).real.to(torch.float64)
        counts.index_copy_(0, rows, best_counts)
        tested[int(K)] = hypotheses
    return DecoderOutput(counts=counts, meta={"decoder": "exact_count_ml", "hypotheses_by_K": tested})


DecoderFn = Callable[..., DecoderOutput]


DECODERS: dict[str, DecoderFn] = {
    "oracle_k_omp": oracle_k_omp,
    "exact_count_ml": exact_count_ml,
}


def get_decoder(name: str) -> DecoderFn:
    if name not in DECODERS:
        raise KeyError(f"unknown decoder '{name}'. available: {sorted(DECODERS)}")
    return DECODERS[name]


def register_decoder(name: str, fn: DecoderFn) -> None:
    """Register a new decoder under `name`. Caller wins on conflicts."""
    DECODERS[name] = fn
