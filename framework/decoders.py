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
                  num_active: int) -> DecoderOutput:
    """Oracle-K non-negative OMP over the multi-antenna observation.

    Y has shape (B, n, M_ant); the routine matched-filters with the known H
    to collapse to a (B, n) signal and then runs scalar NNOMP.
    """
    if num_active <= 0:
        raise ValueError(f"num_active must be positive, got {num_active}")
    y_mf = matched_filter_collapse(Y, H)
    Phi = encoder.explicit_matrix().detach()
    B, M = y_mf.shape[0], encoder.num_codewords
    counts = torch.zeros(B, M, dtype=torch.float64, device=Y.device)
    supports: list[list[int]] = []
    for b in range(B):
        x_s, support = oracle_k_omp_single(Phi, y_mf[b], int(num_active))
        if support:
            projected = project_nonneg_integer_total(x_s, int(num_active))
            counts[b, torch.tensor(support, dtype=torch.long, device=Y.device)] = projected
        supports.append(support)
    return DecoderOutput(counts=counts, meta={
        "decoder": "oracle_k_omp",
        "K_target": int(num_active),
        "supports": supports,
    })


DecoderFn = Callable[..., DecoderOutput]


DECODERS: dict[str, DecoderFn] = {
    "oracle_k_omp": oracle_k_omp,
}


def get_decoder(name: str) -> DecoderFn:
    if name not in DECODERS:
        raise KeyError(f"unknown decoder '{name}'. available: {sorted(DECODERS)}")
    return DECODERS[name]


def register_decoder(name: str, fn: DecoderFn) -> None:
    """Register a new decoder under `name`. Caller wins on conflicts."""
    DECODERS[name] = fn
