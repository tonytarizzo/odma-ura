"""Operator-bank constructors for explicit framework codebooks."""

from __future__ import annotations

import torch


def placement_bank_from_supports(supports: torch.Tensor, n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Build ODMA-style placement matrices from support indices.

    ``supports`` has shape ``(Q, d)`` and the result has shape ``(Q, n, d)``.
    """

    supports = torch.as_tensor(supports, dtype=torch.long)
    if supports.ndim != 2:
        raise ValueError(f"supports must have shape (Q, d), got {tuple(supports.shape)}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if torch.any(supports < 0) or torch.any(supports >= n):
        raise ValueError("supports contains an out-of-range resource index")
    Q, d = supports.shape
    R = torch.zeros(Q, n, d, dtype=dtype, device=supports.device)
    q_idx = torch.arange(Q, device=supports.device)[:, None].expand(Q, d)
    d_idx = torch.arange(d, device=supports.device)[None, :].expand(Q, d)
    R[q_idx, supports, d_idx] = 1.0
    return R


def dense_operator(n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return a one-operator dense identity bank of shape ``(1, n, n)``."""

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return torch.eye(n, dtype=dtype).unsqueeze(0)
