"""Differentiable losses and regularisers for the URA framework.

These act on either the (B, M) count tensor (decoding losses) or directly on
the explicit codebook Phi (codebook-design regularisers). They are intentionally
small - the framework should add more as concrete experiments need them.
"""

from __future__ import annotations

import torch

from .encoder import Encoder


def count_mse_loss(counts_pred: torch.Tensor, counts_true: torch.Tensor) -> torch.Tensor:
    """Mean squared error against the true count vector."""
    if counts_pred.shape != counts_true.shape:
        raise ValueError(f"count shapes disagree: {tuple(counts_pred.shape)} vs {tuple(counts_true.shape)}")
    return ((counts_pred - counts_true) ** 2).mean()


def support_bce_loss(scores: torch.Tensor, counts_true: torch.Tensor) -> torch.Tensor:
    """Sigmoid binary cross-entropy on the support indicator.

    `scores` are real-valued logits of shape (B, M). The target indicator is
    (counts_true > 0).
    """
    target = (counts_true > 0).to(scores.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, target)


def power_penalty(encoder: Encoder, target_per_codeword: float = 1.0) -> torch.Tensor:
    """Sum_m (||phi_m||^2 - E_target)^2 over codewords."""
    Phi = encoder.explicit_matrix()
    energies = (Phi.conj() * Phi).sum(0).real if Phi.is_complex() else (Phi ** 2).sum(0)
    return ((energies - float(target_per_codeword)) ** 2).mean()


def coherence_penalty(encoder: Encoder) -> torch.Tensor:
    """Average squared coherence between normalised codeword pairs."""
    Phi = encoder.explicit_matrix()
    norms = Phi.norm(dim=0, keepdim=True).clamp_min(1e-12)
    P = Phi / norms
    G = P.conj().transpose(-1, -2) @ P if P.is_complex() else P.transpose(-1, -2) @ P
    G = G.abs() ** 2
    M = G.shape[0]
    off = G.sum() - torch.diagonal(G).sum()
    return off / max(M * (M - 1), 1)


def row_load_penalty(encoder: Encoder) -> torch.Tensor:
    """Variance of resource-row energy across the codebook."""
    Phi = encoder.explicit_matrix()
    row_energy = (Phi.conj() * Phi).sum(1).real if Phi.is_complex() else (Phi ** 2).sum(1)
    return row_energy.var(unbiased=False)
