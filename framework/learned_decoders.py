"""Differentiable learned decoders for fixed-encoder experiments.

The first decoder is deliberately conservative: an unrolled non-negative ISTA
solver over the global dictionary. It keeps the model-based gradient step and
learns only scalar per-layer calibration parameters, avoiding dense learned
message-index transforms whose locality assumptions are unclear for URA labels.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .channel import matched_filter_collapse
from .core import DecoderOutput
from .decoders import project_nonneg_integer_total
from .encoder import Encoder


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(float(x)))


def _sigmoid_logit(x: float) -> float:
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


def hard_project_batch(scores: torch.Tensor, total: int) -> torch.Tensor:
    """Top-score integer projection onto nonnegative counts summing to total."""
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B, M), got {tuple(scores.shape)}")
    out = torch.zeros_like(scores, dtype=torch.float64)
    for b in range(scores.shape[0]):
        projected = project_nonneg_integer_total(torch.clamp(scores[b], min=0.0), int(total))
        out[b] = projected.to(out.dtype)
    return out


class UnrolledNonnegativeISTA(nn.Module):
    """Unrolled global nonnegative ISTA with learnable scalar layer parameters.

    Forward pass:
        a_0 = 0
        u_t = a_t + eta_t Phi^H(y - Phi a_t)
        a_{t+1} = smooth_relu(u_t - tau_t)

    The step size is eta_t = softplus(raw_step_t) / ||Phi||_2^2, so the learned
    scalar calibrates a stable dictionary-normalised step.
    """

    def __init__(self, num_layers: int = 8, init_step_scale: float = 0.9,
                 init_threshold: float = 0.05, init_beta: float = 0.02,
                 init_damping: float = 0.05, normalize_sum: bool = True) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.num_layers = int(num_layers)
        self.normalize_sum = bool(normalize_sum)
        self.raw_step = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_step_scale)))
        self.raw_threshold = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_threshold)))
        self.raw_beta = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_beta)))
        self.raw_damping = nn.Parameter(torch.full((num_layers,), _sigmoid_logit(init_damping)))

    @staticmethod
    def _lipschitz(Phi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.linalg.matrix_norm(Phi.detach(), ord=2).square().clamp_min(1e-12)

    def forward(self, encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                num_active: int) -> DecoderOutput:
        y = matched_filter_collapse(Y, H)
        Phi = encoder.explicit_matrix()
        L = self._lipschitz(Phi).to(dtype=Phi.real.dtype if Phi.is_complex() else Phi.dtype, device=Phi.device)
        a = torch.zeros(y.shape[0], encoder.num_codewords, dtype=y.real.dtype if y.is_complex() else y.dtype, device=y.device)
        logits = a
        for t in range(self.num_layers):
            residual = y - encoder.matvec(a.to(encoder.dtype))
            grad = encoder.rmatvec(residual).real
            eta = torch.nn.functional.softplus(self.raw_step[t]) / L
            tau = torch.nn.functional.softplus(self.raw_threshold[t])
            beta = torch.nn.functional.softplus(self.raw_beta[t]).clamp_min(1e-4)
            damping = torch.sigmoid(self.raw_damping[t])
            u = a + eta * grad.to(a.dtype)
            logits = (u - tau) / beta
            proposal = torch.nn.functional.softplus(logits) * beta
            a = damping * a + (1.0 - damping) * proposal
        if self.normalize_sum:
            scale = float(num_active) / a.sum(dim=1, keepdim=True).clamp_min(1e-12)
            a = a * scale
        hard = hard_project_batch(a.detach(), int(num_active)).to(device=a.device)
        return DecoderOutput(counts=hard, meta={
            "soft_counts": a, "support_logits": logits,
            "decoder": "unrolled_nonnegative_ista",
        })


def matched_filter_decoder(encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                           num_active: int) -> DecoderOutput:
    y = matched_filter_collapse(Y, H)
    scores = torch.clamp(encoder.rmatvec(y).real, min=0.0)
    counts = hard_project_batch(scores, int(num_active)).to(device=Y.device)
    return DecoderOutput(counts=counts, meta={"soft_counts": scores, "support_logits": scores, "decoder": "matched_filter"})
