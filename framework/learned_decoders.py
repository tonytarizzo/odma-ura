"""Differentiable model-based decoders over the implicit framework encoder.

All learned decoders keep the exact data-consistency operations ``Phi a`` and
``Phi^H r``. They learn only calibration or the proximal/denoising module; no
dense message-index transform is materialised.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .channel import matched_filter_collapse
from .core import DecoderOutput
from .decoders import active_count_vector, project_nonneg_integer_total
from .encoder import Encoder


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(float(x)))


def _sigmoid_logit(x: float) -> float:
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


def _mass_normalize(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return x * K.to(x.dtype).unsqueeze(1) / x.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _effective_noise(noise_var: float | torch.Tensor | None, H: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    h_energy = torch.sum(torch.abs(H) ** 2, dim=1).real.clamp_min(1e-12)
    if noise_var is None:
        return torch.zeros_like(h_energy, dtype=dtype)
    sigma2 = torch.as_tensor(noise_var, dtype=dtype, device=H.device)
    if sigma2.ndim == 0:
        sigma2 = sigma2.expand(H.shape[0])
    if sigma2.shape != (H.shape[0],):
        raise ValueError(f"noise_var must be scalar or shape ({H.shape[0]},), got {tuple(sigma2.shape)}")
    return sigma2 / h_energy.to(dtype)


def hard_project_batch(scores: torch.Tensor, total: int | torch.Tensor) -> torch.Tensor:
    """Project every row onto nonnegative integer counts with its requested total."""
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B, M), got {tuple(scores.shape)}")
    K = active_count_vector(total, scores.shape[0], scores.device)
    out = torch.zeros_like(scores, dtype=torch.float64)
    for b in range(scores.shape[0]):
        out[b] = project_nonneg_integer_total(torch.clamp(scores[b], min=0.0), int(K[b].item())).to(out.dtype)
    return out


class UnrolledNonnegativeISTA(nn.Module):
    """Conservative smooth nonnegative ISTA baseline with learned layer scalars."""

    def __init__(self, num_layers: int = 8, init_step_scale: float = 0.9,
                 init_threshold: float = 0.05, init_beta: float = 0.02,
                 init_damping: float = 0.05, normalize_sum: bool = True,
                 power_iters: int = 12) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.num_layers = int(num_layers)
        self.normalize_sum = bool(normalize_sum)
        self.power_iters = int(power_iters)
        self.raw_step = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_step_scale)))
        self.raw_threshold = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_threshold)))
        self.raw_beta = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_beta)))
        self.raw_damping = nn.Parameter(torch.full((num_layers,), _sigmoid_logit(init_damping)))

    def forward(self, encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                num_active: int | torch.Tensor,
                noise_var: float | torch.Tensor | None = None) -> DecoderOutput:
        y = matched_filter_collapse(Y, H)
        K = active_count_vector(num_active, y.shape[0], y.device)
        lipschitz = encoder.spectral_norm_squared(self.power_iters).to(y.real.dtype)
        a = torch.zeros(y.shape[0], encoder.num_codewords, dtype=y.real.dtype, device=y.device)
        logits = a
        layer_logits = []
        for t in range(self.num_layers):
            residual = y - encoder.matvec(a.to(encoder.dtype))
            grad = encoder.rmatvec(residual).real
            eta = torch.nn.functional.softplus(self.raw_step[t]) / lipschitz
            tau = torch.nn.functional.softplus(self.raw_threshold[t])
            beta = torch.nn.functional.softplus(self.raw_beta[t]).clamp_min(1e-4)
            damping = torch.sigmoid(self.raw_damping[t])
            u = a + eta * grad.to(a.dtype)
            logits = (u - tau) / beta
            proposal = torch.nn.functional.softplus(logits) * beta
            if self.normalize_sum:
                proposal = _mass_normalize(proposal, K)
            a = damping * a + (1.0 - damping) * proposal
            if self.normalize_sum:
                a = _mass_normalize(a.clamp_min(1e-12), K)
            layer_logits.append(logits)
        hard = hard_project_batch(a.detach(), K).to(device=a.device)
        return DecoderOutput(counts=hard, meta={"soft_counts": a, "support_logits": logits,
                             "layer_logits": layer_logits, "decoder": "unrolled_nonnegative_ista"})


class UnrolledBernoulliPGD(nn.Module):
    """D0: implicit projected-gradient decoder with a calibrated Bernoulli denoiser."""

    def __init__(self, num_layers: int = 10, init_step_scale: float = 0.9,
                 init_damping: float = 0.05, power_iters: int = 12) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.num_layers = int(num_layers)
        self.power_iters = int(power_iters)
        self.raw_step = nn.Parameter(torch.full((num_layers,), _inv_softplus(init_step_scale)))
        self.raw_tau_scale = nn.Parameter(torch.full((num_layers,), _inv_softplus(1.0)))
        self.raw_noise_mix = nn.Parameter(torch.full((num_layers,), _sigmoid_logit(0.25)))
        self.raw_evidence_gain = nn.Parameter(torch.full((num_layers,), _inv_softplus(1.0)))
        self.raw_prior_gain = nn.Parameter(torch.full((num_layers,), _inv_softplus(1.0)))
        self.bias = nn.Parameter(torch.zeros(num_layers))
        self.raw_damping = nn.Parameter(torch.full((num_layers,), _sigmoid_logit(init_damping)))

    def _base_layer(self, encoder: Encoder, y: torch.Tensor, a: torch.Tensor, K: torch.Tensor,
                    noise_eff: torch.Tensor, lipschitz: torch.Tensor, t: int
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = y - encoder.matvec(a.to(encoder.dtype))
        grad = encoder.rmatvec(residual).real.to(a.dtype)
        eta = torch.nn.functional.softplus(self.raw_step[t]) / lipschitz
        u = a + eta * grad
        residual_scale = torch.mean(torch.abs(residual) ** 2, dim=1).real.to(a.dtype)
        noise_mix = torch.sigmoid(self.raw_noise_mix[t])
        variance_proxy = (1.0 - noise_mix) * residual_scale + noise_mix * noise_eff
        tau = torch.nn.functional.softplus(self.raw_tau_scale[t]) * variance_proxy.clamp_min(1e-6)
        rho = (K.to(a.dtype) / float(encoder.num_codewords)).clamp(1e-7, 1.0 - 1e-7)
        prior_logit = torch.log(rho) - torch.log1p(-rho)
        evidence_gain = torch.nn.functional.softplus(self.raw_evidence_gain[t])
        prior_gain = torch.nn.functional.softplus(self.raw_prior_gain[t])
        logits = evidence_gain * (u - 0.5) / tau.unsqueeze(1) + prior_gain * prior_logit.unsqueeze(1) + self.bias[t]
        return logits.clamp(-30.0, 30.0), u, grad, residual_scale, prior_logit

    def _proposal(self, logits: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return _mass_normalize(torch.sigmoid(logits).clamp_min(1e-12), K)

    def forward(self, encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                num_active: int | torch.Tensor,
                noise_var: float | torch.Tensor | None = None) -> DecoderOutput:
        y = matched_filter_collapse(Y, H)
        dtype = y.real.dtype
        K = active_count_vector(num_active, y.shape[0], y.device)
        noise_eff = _effective_noise(noise_var, H, dtype)
        lipschitz = encoder.spectral_norm_squared(self.power_iters).to(dtype)
        a = torch.zeros(y.shape[0], encoder.num_codewords, dtype=dtype, device=y.device)
        layer_logits = []
        for t in range(self.num_layers):
            logits, _, _, _, _ = self._base_layer(encoder, y, a, K, noise_eff, lipschitz, t)
            proposal = self._proposal(logits, K)
            damping = torch.sigmoid(self.raw_damping[t])
            a = _mass_normalize((damping * a + (1.0 - damping) * proposal).clamp_min(1e-12), K)
            layer_logits.append(logits)
        hard = hard_project_batch(a.detach(), K).to(device=a.device)
        return DecoderOutput(counts=hard, meta={"soft_counts": a, "support_logits": layer_logits[-1],
                             "layer_logits": layer_logits, "decoder": "unrolled_bernoulli_pgd",
                             "noise_effective": noise_eff.detach()})


class FactorAttentionProx(nn.Module):
    """Nonlocal, permutation-compatible analysis/threshold/synthesis proximal correction."""

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 32,
                 pattern_slots: int = 2, value_slots: int = 2, global_slots: int = 4) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.analysis = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pattern_queries = nn.Parameter(torch.randn(pattern_slots, hidden_dim) / math.sqrt(hidden_dim))
        self.value_queries = nn.Parameter(torch.randn(value_slots, hidden_dim) / math.sqrt(hidden_dim))
        self.global_queries = nn.Parameter(torch.randn(global_slots, hidden_dim) / math.sqrt(hidden_dim))
        self.mix = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.inverse = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.correction = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.raw_threshold = nn.Parameter(torch.tensor(_inv_softplus(0.05)))

    @staticmethod
    def _group_context(x: torch.Tensor, queries: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Pool latent slots within arbitrary groups, then broadcast them to group members."""
        batch, messages, hidden = x.shape
        groups = int(group_ids.max().item()) + 1
        scores = torch.einsum("bmh,sh->bms", x, queries) / math.sqrt(hidden)
        slot_count = scores.shape[2]
        index = group_ids.view(1, messages, 1).expand(batch, messages, slot_count)
        maxima = scores.detach().new_full((batch, groups, slot_count), float("-inf"))
        maxima.scatter_reduce_(1, index, scores.detach(), reduce="amax", include_self=True)
        unnormalised = torch.exp(scores - maxima.gather(1, index))
        denominators = scores.new_zeros(batch, groups, slot_count)
        denominators.scatter_add_(1, index, unnormalised)
        weights = unnormalised / denominators.gather(1, index).clamp_min(1e-12)
        slots = x.new_zeros(batch, groups, slot_count, hidden)
        slot_index = group_ids.view(1, messages, 1, 1).expand(batch, messages, slot_count, hidden)
        slots.scatter_add_(1, slot_index, weights.unsqueeze(-1) * x.unsqueeze(2))
        selected = slots.index_select(1, group_ids)
        broadcast_weights = torch.softmax(torch.einsum("bmh,bmsh->bms", x, selected) / math.sqrt(hidden), dim=2)
        return torch.einsum("bms,bmsh->bmh", broadcast_weights, selected)

    def forward(self, features: torch.Tensor, u: torch.Tensor, q_ids: torch.Tensor, v_ids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.analysis(features)
        pattern_context = self._group_context(latent, self.pattern_queries, q_ids)
        value_context = self._group_context(latent, self.value_queries, v_ids)
        global_context = self._group_context(latent, self.global_queries, torch.zeros_like(q_ids))
        mixed = latent + self.mix(torch.cat([latent, pattern_context, value_context, global_context], dim=-1))
        symmetry = torch.mean((self.inverse(mixed).squeeze(-1) - u) ** 2)
        threshold = torch.nn.functional.softplus(self.raw_threshold)
        sparse = torch.sign(mixed) * torch.relu(torch.abs(mixed) - threshold)
        return self.correction(sparse).squeeze(-1), symmetry


class FactorAttentionISTANet(UnrolledBernoulliPGD):
    """D1: D0 plus a nonlocal factor-aware ISTA-Net-style proximal correction."""

    def __init__(self, num_layers: int = 10, hidden_dim: int = 32,
                 pattern_slots: int = 2, value_slots: int = 2, global_slots: int = 4,
                 init_step_scale: float = 0.9, init_damping: float = 0.05,
                 power_iters: int = 12) -> None:
        super().__init__(num_layers, init_step_scale, init_damping, power_iters)
        self.prox_layers = nn.ModuleList([
            FactorAttentionProx(8, hidden_dim, pattern_slots, value_slots, global_slots) for _ in range(num_layers)
        ])

    @staticmethod
    def _factor_ids(encoder: Encoder) -> tuple[torch.Tensor, torch.Tensor]:
        if len(encoder.components) != 1:
            q = torch.zeros(encoder.num_codewords, dtype=torch.long, device=encoder.device)
            return q, torch.arange(encoder.num_codewords, device=encoder.device)
        comp = encoder.components[0]
        atoms = comp.msg_to_atom
        return comp.atom_q[atoms], comp.atom_v[atoms]

    def forward(self, encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                num_active: int | torch.Tensor,
                noise_var: float | torch.Tensor | None = None) -> DecoderOutput:
        y = matched_filter_collapse(Y, H)
        dtype = y.real.dtype
        K = active_count_vector(num_active, y.shape[0], y.device)
        noise_eff = _effective_noise(noise_var, H, dtype)
        lipschitz = encoder.spectral_norm_squared(self.power_iters).to(dtype)
        q_ids, v_ids = self._factor_ids(encoder)
        a = torch.zeros(y.shape[0], encoder.num_codewords, dtype=dtype, device=y.device)
        layer_logits, symmetry_losses = [], []
        for t in range(self.num_layers):
            base, u, grad, residual_scale, prior_logit = self._base_layer(
                encoder, y, a, K, noise_eff, lipschitz, t)
            pattern_mass = torch.zeros(a.shape[0], int(q_ids.max().item()) + 1, dtype=a.dtype, device=a.device)
            pattern_mass.scatter_add_(1, q_ids.unsqueeze(0).expand(a.shape[0], -1), a)
            atom_pattern_mass = pattern_mass.index_select(1, q_ids) / K.to(a.dtype).unsqueeze(1)
            features = torch.stack([
                u, a, grad, torch.tanh(base / 8.0), atom_pattern_mass,
                residual_scale.log1p().unsqueeze(1).expand_as(a),
                noise_eff.log1p().unsqueeze(1).expand_as(a),
                torch.tanh(prior_logit / 8.0).unsqueeze(1).expand_as(a),
            ], dim=-1)
            correction, symmetry = self.prox_layers[t](features, u, q_ids, v_ids)
            logits = (base + correction).clamp(-30.0, 30.0)
            proposal = self._proposal(logits, K)
            damping = torch.sigmoid(self.raw_damping[t])
            a = _mass_normalize((damping * a + (1.0 - damping) * proposal).clamp_min(1e-12), K)
            layer_logits.append(logits); symmetry_losses.append(symmetry)
        hard = hard_project_batch(a.detach(), K).to(device=a.device)
        return DecoderOutput(counts=hard, meta={"soft_counts": a, "support_logits": layer_logits[-1],
                             "layer_logits": layer_logits, "symmetry_loss": torch.stack(symmetry_losses).mean(),
                             "decoder": "factor_attention_istanet", "noise_effective": noise_eff.detach()})


def matched_filter_decoder(encoder: Encoder, Y: torch.Tensor, H: torch.Tensor,
                           num_active: int | torch.Tensor,
                           noise_var: float | torch.Tensor | None = None) -> DecoderOutput:
    y = matched_filter_collapse(Y, H)
    scores = torch.clamp(encoder.rmatvec(y).real, min=0.0)
    counts = hard_project_batch(scores, num_active).to(device=Y.device)
    return DecoderOutput(counts=counts, meta={"soft_counts": scores, "support_logits": scores,
                                              "decoder": "matched_filter"})
