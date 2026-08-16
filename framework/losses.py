"""Differentiable losses and regularisers for the URA framework.

These act on either the (B, M) count tensor (decoding losses) or directly on
the explicit codebook Phi (codebook-design regularisers). They are intentionally
small - the framework should add more as concrete experiments need them.
"""

from __future__ import annotations

import torch

from .encoder import Encoder
from .sectioned import SectionedEncoder


def count_mse_loss(counts_pred: torch.Tensor, counts_true: torch.Tensor) -> torch.Tensor:
    """Mean squared error against the true count vector."""
    if counts_pred.shape != counts_true.shape:
        raise ValueError(f"count shapes disagree: {tuple(counts_pred.shape)} vs {tuple(counts_true.shape)}")
    return ((counts_pred.real - counts_true.real) ** 2).mean()


def support_bce_loss(scores: torch.Tensor, counts_true: torch.Tensor) -> torch.Tensor:
    """Sigmoid binary cross-entropy on the support indicator.

    `scores` are real-valued logits of shape (B, M). The target indicator is
    (counts_true > 0).
    """
    target = (counts_true.real > 0).to(scores.real.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(scores.real, target)


def support_count_loss(output, counts_true: torch.Tensor, lambda_count: float = 0.1,
                       lambda_symmetry: float = 0.01, deep_supervision: bool = True
                       ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced support BCE plus a collision-aware count loss and optional D1 symmetry.

    BCE targets message presence, so two users selecting one message still produce a
    single positive target. The auxiliary count term preserves that multiplicity
    information without changing the low-collision support-recovery objective.
    """
    logits = output.meta.get("layer_logits", [output.meta["support_logits"]])
    if not deep_supervision:
        logits = logits[-1:]
    target_counts = counts_true.real
    target = (target_counts > 0).to(logits[-1].dtype)
    active = target.sum(dim=1).clamp_min(1.0)
    inactive = (target.shape[1] - target.sum(dim=1)).clamp_min(1.0)
    layer_losses = []
    for layer_logits in logits:
        per_entry = torch.nn.functional.binary_cross_entropy_with_logits(layer_logits, target, reduction="none")
        positive = (per_entry * target).sum(dim=1) / active
        negative = (per_entry * (1.0 - target)).sum(dim=1) / inactive
        layer_losses.append(0.5 * (positive + negative).mean())
    weights = torch.arange(1, len(layer_losses) + 1, dtype=target.dtype, device=target.device)
    support = torch.sum(weights * torch.stack(layer_losses)) / weights.sum()
    soft = output.meta["soft_counts"]
    K = target_counts.sum(dim=1).clamp_min(1.0)
    count = torch.nn.functional.smooth_l1_loss(soft, target_counts.to(soft.dtype), reduction="none").sum(dim=1)
    count = (count / K.to(count.dtype)).mean()
    symmetry = output.meta.get("symmetry_loss", support.new_zeros(()))
    total = support + float(lambda_count) * count + float(lambda_symmetry) * symmetry
    return total, {"support": support, "count": count, "symmetry": symmetry, "total": total}


def section_support_count_loss(output, section_counts_true: tuple[torch.Tensor, ...],
                               lambda_count: float = 0.1, deep_supervision: bool = True
                               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced support/count supervision over local sections without an M-axis target."""
    logits_by_layer = output.meta["section_layer_logits"]
    if not deep_supervision:
        logits_by_layer = logits_by_layer[-1:]
    if not logits_by_layer or len(logits_by_layer[-1]) != len(section_counts_true):
        raise ValueError("section logits and targets disagree")
    section_support: list[torch.Tensor] = []
    section_count: list[torch.Tensor] = []
    for ell, target_counts in enumerate(section_counts_true):
        target_counts = target_counts.real
        target = (target_counts > 0).to(logits_by_layer[-1][ell].dtype)
        active = target.sum(dim=1).clamp_min(1.0)
        inactive = (target.shape[1] - target.sum(dim=1)).clamp_min(1.0)
        layer_losses = []
        for layer in logits_by_layer:
            per_entry = torch.nn.functional.binary_cross_entropy_with_logits(layer[ell], target, reduction="none")
            positive = (per_entry * target).sum(dim=1) / active
            negative = (per_entry * (1.0 - target)).sum(dim=1) / inactive
            layer_losses.append(0.5 * (positive + negative).mean())
        weights = torch.arange(1, len(layer_losses) + 1, dtype=target.dtype, device=target.device)
        section_support.append(torch.sum(weights * torch.stack(layer_losses)) / weights.sum())
        soft = output.meta["soft_section_counts"][ell]
        K = target_counts.sum(dim=1).clamp_min(1.0)
        count = torch.nn.functional.smooth_l1_loss(soft, target_counts.to(soft.dtype), reduction="none").sum(dim=1)
        section_count.append((count / K.to(count.dtype)).mean())
    support = torch.stack(section_support).mean()
    count = torch.stack(section_count).mean()
    total = support + float(lambda_count) * count
    return total, {"support": support, "count": count, "total": total}


def power_penalty(encoder: Encoder, target_per_codeword: float = 1.0) -> torch.Tensor:
    """Sum_m (||phi_m||^2 - E_target)^2 over codewords."""
    Phi = encoder.explicit_matrix()
    energies = (Phi.conj() * Phi).sum(0).real if Phi.is_complex() else (Phi ** 2).sum(0)
    return ((energies - float(target_per_codeword)) ** 2).mean()


def sectioned_power_penalty(encoder: SectionedEncoder, valid_paths: torch.Tensor,
                            target_per_codeword: float = 1.0) -> torch.Tensor:
    """Scalable power penalty evaluated only on supplied procedurally valid paths."""
    energies = encoder.path_energies(valid_paths)
    return torch.mean((energies - float(target_per_codeword)) ** 2)


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
