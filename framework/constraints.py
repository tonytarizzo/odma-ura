"""Constraint projections for learnable factors.

These are applied with `torch.no_grad()` *after* each optimiser step, in the
classical "projected gradient" style. The function set is intentionally small:
add new projections only when an experiment actually needs them.
"""

from __future__ import annotations

from typing import Iterable

import torch


def project_unit_norm_columns(C: torch.Tensor, eps: float = 1e-12) -> None:
    """In-place projection of each column of C onto the unit sphere."""
    if C.ndim != 2:
        raise ValueError(f"C must be 2-D, got shape {tuple(C.shape)}")
    norms = C.norm(dim=0, keepdim=True).clamp_min(eps)
    C.div_(norms)


def project_unit_norm_codewords(Phi: torch.Tensor, eps: float = 1e-12) -> None:
    """In-place projection of each column of Phi onto the unit sphere."""
    project_unit_norm_columns(Phi, eps=eps)


def project_unit_frob(M: torch.Tensor, eps: float = 1e-12) -> None:
    """In-place rescaling of `M` to unit Frobenius norm."""
    f = M.norm().clamp_min(eps)
    M.div_(f)


def apply_constraints(named_tensors: Iterable[tuple[str, torch.Tensor, str]]) -> None:
    """Dispatch projections by name."""
    with torch.no_grad():
        for name, tensor, kind in named_tensors:
            if kind == "none":
                continue
            if kind == "unit_norm_columns":
                project_unit_norm_columns(tensor)
            elif kind == "unit_frob":
                project_unit_frob(tensor)
            else:
                raise ValueError(f"unknown constraint '{kind}' on '{name}'")
