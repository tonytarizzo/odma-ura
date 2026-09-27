"""Bounded-candidate decoding without a global ``M=2^B`` state.

D3 is a candidate-conditioned refinement stage, not a global candidate search.
It generates only the supplied codewords, applies the effective-channel algebra
of D2 on that bounded list, and reports proposal recall separately from
conditional decoding performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .candidate_geometry import CandidateGeometryCorrection, candidate_structure_features
from .channel import matched_filter_collapse
from .decoders import active_count_vector
from .learned_decoders import (_bernoulli_cardinality_projection, _effective_noise, _inv_softplus,
                               _sigmoid_logit, hard_project_batch)


@dataclass(frozen=True)
class CandidateSet:
    """A rectangular bounded candidate list.

    Exactly one identifier representation is required. ``message_bits`` has
    shape ``(batch,C,B)`` (or common-list ``(C,B)``) and is the scalable path
    for a procedural codebook. ``message_indices`` has shape ``(batch,C)`` (or
    ``(C,)``) and addresses a small-B global ``Encoder`` without materialising
    its full matrix. Scores are optional proposer diagnostics, not a prior.
    """

    message_bits: torch.Tensor | None = None
    message_indices: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    source: str = "supplied"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.message_bits is None) == (self.message_indices is None):
            raise ValueError("CandidateSet requires exactly one of message_bits or message_indices")
        if not self.source:
            raise ValueError("candidate source must be nonempty")


@dataclass
class CandidateDecoderOutput:
    """D3/D4 output whose count axis is the supplied candidate list."""

    candidates: CandidateSet
    counts: torch.Tensor
    meta: dict = field(default_factory=dict)


def _binary_tensor(value, device: torch.device, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must use a binary integer or Boolean dtype")
    if bool(torch.any((tensor != 0) & (tensor != 1))):
        raise ValueError(f"{name} must be binary")
    return tensor.to(torch.uint8)


def _expand_candidates(candidates: CandidateSet, batch_size: int, device: torch.device
                       ) -> tuple[CandidateSet, int]:
    bits, indices = candidates.message_bits, candidates.message_indices
    if bits is not None:
        bits = _binary_tensor(bits, device, "candidate message bits")
        if bits.ndim == 2:
            bits = bits.unsqueeze(0).expand(batch_size, -1, -1)
        if bits.ndim != 3 or bits.shape[0] != batch_size:
            raise ValueError(f"candidate bits must have shape (C,B) or ({batch_size},C,B), got {tuple(bits.shape)}")
        C = int(bits.shape[1])
    else:
        indices = torch.as_tensor(indices, device=device)
        if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise TypeError("candidate message indices must use an integer dtype")
        indices = indices.to(torch.long)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        if indices.ndim != 2 or indices.shape[0] != batch_size:
            raise ValueError(f"candidate indices must have shape (C,) or ({batch_size},C), got {tuple(indices.shape)}")
        C = int(indices.shape[1])
    if C <= 1:
        raise ValueError("candidate lists must contain at least two entries")
    identifiers = bits if bits is not None else indices
    if any(torch.unique(identifiers[b], dim=0).shape[0] != C for b in range(batch_size)):
        raise ValueError("candidate lists must not contain duplicate messages")
    scores = candidates.scores
    if scores is not None:
        scores = torch.as_tensor(scores, device=device)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0).expand(batch_size, -1)
        if scores.shape != (batch_size, C):
            raise ValueError(f"candidate scores must have shape (C,) or ({batch_size},{C}), got {tuple(scores.shape)}")
    return CandidateSet(bits, indices, scores, candidates.source, dict(candidates.metadata)), C


def _procedural_codebook(encoder):
    codebook = getattr(encoder, "codebook", encoder)
    return codebook if hasattr(codebook, "codewords") and hasattr(codebook, "payload_bits") else None


def _candidate_columns(encoder, candidates: CandidateSet, batch_size: int) -> torch.Tensor:
    """Generate ``(batch,n,C)`` columns, never an ``n x M`` matrix."""
    if candidates.message_bits is not None:
        codebook = _procedural_codebook(encoder)
        if codebook is None:
            raise TypeError("message-bit candidates require a procedural codebook")
        bits = candidates.message_bits
        if bits.shape[2] != int(codebook.payload_bits):
            raise ValueError(f"candidate messages have B={bits.shape[2]}, codebook expects B={codebook.payload_bits}")
        flat = bits.reshape(-1, bits.shape[-1])
        columns = codebook.codewords(flat).transpose(0, 1).reshape(batch_size, bits.shape[1], codebook.n)
        return columns.transpose(1, 2)

    if not hasattr(encoder, "message_columns"):
        raise TypeError("indexed candidates require an Encoder or object with message_columns")
    column_fn = encoder.message_columns
    num_codewords = getattr(encoder, "num_codewords", None)
    indices = candidates.message_indices
    if num_codewords is not None and (bool(torch.any(indices < 0)) or bool(torch.any(indices >= int(num_codewords)))):
        raise ValueError("candidate index lies outside the global encoder alphabet")
    return torch.stack([column_fn(indices[b]) for b in range(batch_size)], dim=0)


def candidate_recall(candidate_bits: torch.Tensor, true_message_bits: torch.Tensor,
                     true_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample device-weighted recall; repeated true messages count repeatedly."""
    candidate_bits = _binary_tensor(candidate_bits, torch.as_tensor(candidate_bits).device, "candidate message bits")
    truth = _binary_tensor(true_message_bits, candidate_bits.device, "true message bits")
    if candidate_bits.ndim != 3 or truth.ndim != 3 or candidate_bits.shape[0] != truth.shape[0] or candidate_bits.shape[2] != truth.shape[2]:
        raise ValueError("candidate and true bits must have shapes (batch,C,B) and (batch,K_max,B)")
    found = (truth.unsqueeze(2) == candidate_bits.unsqueeze(1)).all(dim=-1).any(dim=-1)
    if true_mask is None:
        mask = torch.ones_like(found, dtype=torch.bool)
    else:
        mask = torch.as_tensor(true_mask, dtype=torch.bool, device=found.device)
        if mask.shape != found.shape:
            raise ValueError(f"true_mask must have shape {tuple(found.shape)}, got {tuple(mask.shape)}")
    return (found & mask).sum(dim=1).to(torch.float32) / mask.sum(dim=1).clamp_min(1)


def candidate_count_targets(candidates: CandidateSet, *, global_counts: torch.Tensor | None = None,
                            true_message_bits: torch.Tensor | None = None,
                            true_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Construct candidate-axis supervision without silently treating misses as successes."""
    if candidates.message_indices is not None:
        if global_counts is None or true_message_bits is not None:
            raise ValueError("indexed candidates require global_counts and no true_message_bits")
        counts = torch.as_tensor(global_counts)
        if counts.ndim != 2:
            raise ValueError("global_counts must have shape (batch,M)")
        candidates, _ = _expand_candidates(candidates, counts.shape[0], counts.device)
        indices = candidates.message_indices
        return torch.gather(counts.real, 1, indices)
    if true_message_bits is None or global_counts is not None:
        raise ValueError("bit candidates require true_message_bits and no global_counts")
    truth_value = torch.as_tensor(true_message_bits)
    candidate_device = torch.as_tensor(candidates.message_bits).device
    truth = _binary_tensor(truth_value, candidate_device, "true message bits")
    if truth.ndim != 3:
        raise ValueError("true bits must have shape (batch,K_max,B)")
    candidates, _ = _expand_candidates(candidates, truth.shape[0], truth.device)
    bits = candidates.message_bits
    if truth.shape[2] != bits.shape[2]:
        raise ValueError("true bits must have shape (batch,K_max,B) matching candidate bits")
    matches = (truth.unsqueeze(2) == bits.unsqueeze(1)).all(dim=-1)
    if true_mask is not None:
        mask = torch.as_tensor(true_mask, dtype=torch.bool, device=bits.device)
        if mask.shape != matches.shape[:2]:
            raise ValueError(f"true_mask must have shape {tuple(matches.shape[:2])}, got {tuple(mask.shape)}")
        matches = matches & mask.unsqueeze(-1)
    return matches.sum(dim=1).to(torch.float32)


class BoundedMatchedFilterProposer:
    """Rank an explicitly supplied bounded pool by normalised matched filtering.

    This is a useful non-oracle local proposer, but it is not a search over all
    ``2^B`` messages. At large B the caller must still construct a bounded pool
    by some external inverse/search procedure.
    """

    def __init__(self, list_size: int) -> None:
        if list_size <= 1:
            raise ValueError("matched-filter candidate list size must exceed one")
        self.list_size = int(list_size)

    def __call__(self, encoder, Y: torch.Tensor, H: torch.Tensor, pool: CandidateSet) -> CandidateSet:
        pool, P = _expand_candidates(pool, Y.shape[0], Y.device)
        if self.list_size > P:
            raise ValueError(f"requested {self.list_size} candidates from a pool of {P}")
        Phi = _candidate_columns(encoder, pool, Y.shape[0])
        y = matched_filter_collapse(Y, H)
        correlations = torch.einsum("bnc,bn->bc", Phi.conj(), y).real
        energy = torch.sum(torch.abs(Phi) ** 2, dim=1).real.clamp_min(1e-12)
        scores = correlations / energy.sqrt()
        selected_scores, selected = torch.topk(scores, self.list_size, dim=1)
        if pool.message_bits is not None:
            gather = selected.unsqueeze(-1).expand(-1, -1, pool.message_bits.shape[-1])
            bits, indices = torch.gather(pool.message_bits, 1, gather), None
        else:
            bits, indices = None, torch.gather(pool.message_indices, 1, selected)
        metadata = dict(pool.metadata)
        pool_global_search = bool(metadata.get("global_search", False))
        metadata.update({"pool_size": P, "list_size": self.list_size, "score": "normalised_matched_filter",
                         "pool_global_search": pool_global_search, "global_search": pool_global_search,
                         "proposer_global_search": False, "proposal_scope": "bounded_supplied_pool"})
        return CandidateSet(bits, indices, selected_scores, "bounded_matched_filter", metadata)


class CandidateRestrictedEffectiveChannelPGD(nn.Module):
    """D3: D2 inference restricted to a supplied bounded candidate list.

    The exact real decision Gram ``Re(Phi_C^H Phi_C)`` is affordable on the
    candidate axis. D3 applies D2's mean-field row-energy approximation to that
    bounded Gram, so an exhaustive list reduces to the same effective-channel
    algebra while normal inference never has a global message state. The
    uniform conditional prior and known-K projection assume negligible message
    collisions and that the candidate list contains the transmitted messages.
    """

    def __init__(self, num_layers: int = 10, init_damping: float = 0.05) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.num_layers = int(num_layers)
        self.raw_tau_scale = nn.Parameter(torch.full((num_layers,), _inv_softplus(1.0)))
        self.raw_damping = nn.Parameter(torch.full((num_layers,), _sigmoid_logit(init_damping)))

    def _logit_correction(self, t: int, *, a: torch.Tensor, statistic: torch.Tensor,
                          evidence: torch.Tensor, tau: torch.Tensor, gram: torch.Tensor,
                          proposal_scores: torch.Tensor | None,
                          candidate_features: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """D4 extension point; D3 is the exact zero-correction member."""
        return torch.zeros_like(evidence)

    @staticmethod
    def _effective_variance(variance: torch.Tensor, decision_gram: torch.Tensor,
                            noise_eff: torch.Tensor,
                            complex_observation: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diagonal = torch.diagonal(decision_gram, dim1=-2, dim2=-1)
        C = variance.shape[1]
        other_mean = (variance.sum(dim=1, keepdim=True) - variance) / float(C - 1)
        row_energy = decision_gram.square().sum(dim=-1)
        interference = other_mean * (row_energy - diagonal.square()).clamp_min(0.0)
        real_noise_factor = 0.5 if complex_observation else 1.0
        physical = real_noise_factor * noise_eff.unsqueeze(1) * diagonal
        return physical + interference, physical, interference

    def forward(self, encoder, Y: torch.Tensor, H: torch.Tensor, num_active: int | torch.Tensor,
                noise_var: float | torch.Tensor | None = None, *, candidates: CandidateSet,
                true_message_bits: torch.Tensor | None = None,
                true_mask: torch.Tensor | None = None) -> CandidateDecoderOutput:
        y = matched_filter_collapse(Y, H)
        dtype, batch = y.real.dtype, y.shape[0]
        candidates, C = _expand_candidates(candidates, batch, y.device)
        K = active_count_vector(num_active, batch, y.device)
        if bool(torch.any(K >= C)):
            raise ValueError(f"D3's Bernoulli list model requires K < candidate count C={C}")
        Phi = _candidate_columns(encoder, candidates, batch).to(y.device)
        if Phi.shape[:2] != (batch, y.shape[1]) or Phi.shape[2] != C:
            raise RuntimeError("candidate column generator returned an inconsistent shape")
        gram = torch.einsum("bnc,bnd->bcd", Phi.conj(), Phi)
        decision_gram = gram.real.to(dtype)
        diagonal = torch.diagonal(decision_gram, dim1=-2, dim2=-1).clamp_min(torch.finfo(dtype).eps)
        noise_eff = _effective_noise(noise_var, H, dtype)
        rho = (K.to(dtype) / float(C)).unsqueeze(1)
        a = rho.expand(batch, C)
        variance = a * (1.0 - a)
        prior_logit = torch.log(rho) - torch.log1p(-rho)
        support_overlap, same_amplitude_label = candidate_structure_features(encoder, candidates.message_bits) \
            if candidates.message_bits is not None else (None, None)
        features = {"support_overlap": support_overlap, "same_amplitude_label": same_amplitude_label}
        layer_logits, layer_evidence, layer_raw_evidence, layer_variances = [], [], [], []
        layer_statistics, layer_gradients, layer_corrections = [], [], []
        last_physical = last_interference = None
        for t in range(self.num_layers):
            residual = y - torch.einsum("bnc,bc->bn", Phi, a.to(Phi.dtype))
            gradient = torch.einsum("bnc,bn->bc", Phi.conj(), residual).real.to(dtype)
            analytic_tau, physical, interference = self._effective_variance(
                variance, decision_gram, noise_eff, y.is_complex())
            tau = torch.nn.functional.softplus(self.raw_tau_scale[t]) * analytic_tau
            tau = tau.clamp_min(tau.new_tensor(1e-12))
            statistic = gradient + diagonal * a
            evidence = diagonal * (statistic - 0.5 * diagonal) / tau
            correction = self._logit_correction(t, a=a, statistic=statistic, evidence=evidence, tau=tau,
                                                gram=gram, proposal_scores=candidates.scores,
                                                candidate_features=features)
            if correction.shape != evidence.shape:
                raise ValueError(f"logit correction must have shape {tuple(evidence.shape)}, got {tuple(correction.shape)}")
            corrected_evidence = evidence + correction
            logits, proposal = _bernoulli_cardinality_projection(prior_logit + corrected_evidence, K)
            damping = torch.sigmoid(self.raw_damping[t])
            a = damping * a + (1.0 - damping) * proposal
            variance = a * (1.0 - a)
            layer_logits.append(logits); layer_evidence.append(corrected_evidence)
            layer_raw_evidence.append(evidence); layer_variances.append(tau)
            layer_statistics.append(statistic); layer_gradients.append(gradient); layer_corrections.append(correction)
            last_physical, last_interference = physical, interference
        hard = hard_project_batch(a.detach(), K).to(device=a.device)
        axis_flag = getattr(encoder, "global_message_axis_present", None)
        encoder_global_axis = True if axis_flag is None else bool(axis_flag)
        meta = {
            "soft_counts": a, "support_logits": layer_logits[-1], "layer_logits": layer_logits,
            "layer_evidence_logits": layer_evidence, "layer_effective_variances": layer_variances,
            "layer_raw_evidence_logits": layer_raw_evidence,
            "layer_decision_statistics": layer_statistics, "layer_gradients": layer_gradients,
            "layer_logit_corrections": layer_corrections,
            "effective_variance": layer_variances[-1], "physical_variance": last_physical,
            "interference_variance": last_interference, "candidate_codewords": Phi, "candidate_gram": gram,
            "candidate_decision_gram": decision_gram, "candidate_list_size": C,
            "candidate_support_overlap": support_overlap,
            "candidate_same_amplitude_label": same_amplitude_label,
            "candidate_list_size_per_sample": torch.full((batch,), C, dtype=torch.long, device=y.device),
            "candidate_source": candidates.source, "proposal_metadata": candidates.metadata,
            "proposal_scores": candidates.scores, "cardinality_residual": a.sum(dim=1) - K.to(dtype),
            "decoder": "candidate_restricted_effective_channel_pgd", "decoder_label": "D3",
            "prior": "uniform_bernoulli_conditioned_on_candidate_list_and_known_K",
            "variance_model": "candidate_gram_mean_field_off_diagonal",
            "decoder_global_message_axis_materialized": False,
            "encoder_global_message_axis_present": encoder_global_axis,
            "encoder_global_axis_capability_declared": axis_flag is not None,
            "end_to_end_global_message_axis_free": axis_flag is False,
            "candidate_search_solved": False,
            "decision_statistic": "interference_cancelled_real_adjoint", "noise_effective": noise_eff.detach(),
        }
        if true_message_bits is not None:
            if candidates.message_bits is None:
                raise ValueError("candidate recall from message bits requires bit-valued candidates")
            recall = candidate_recall(candidates.message_bits, true_message_bits, true_mask)
            meta.update({"candidate_recall": recall, "candidate_recall_mean": recall.mean()})
        return CandidateDecoderOutput(candidates, hard, meta)


class LearnedCandidateGeometryPGD(CandidateRestrictedEffectiveChannelPGD):
    """D4: D3 plus a zero-initialised permutation-equivariant graph correction.

    The bounded candidate set, exact candidate Gram, known-K prior, and
    cardinality projection are unchanged from D3. Learning can add only
    candidate-local evidence through Gram, hash-overlap, amplitude-label,
    proposal-score, state, and uncertainty features. Thus D4 equals D3 exactly
    before training and never constructs a global ``M``-axis attention tensor.
    """

    def __init__(self, num_layers: int = 10, hidden_dim: int = 32,
                 init_damping: float = 0.05) -> None:
        super().__init__(num_layers=num_layers, init_damping=init_damping)
        self.geometry_layers = nn.ModuleList(CandidateGeometryCorrection(hidden_dim) for _ in range(num_layers))

    def _logit_correction(self, t: int, *, a, statistic, evidence, tau, gram,
                          proposal_scores, candidate_features):
        return self.geometry_layers[t](a, statistic, evidence, tau, gram, proposal_scores,
                                       candidate_features.get("support_overlap"),
                                       candidate_features.get("same_amplitude_label"))

    def forward(self, *args, **kwargs) -> CandidateDecoderOutput:
        output = super().forward(*args, **kwargs)
        output.meta.update({"decoder": "learned_candidate_geometry_pgd", "decoder_label": "D4",
                            "geometry_correction": "permutation_equivariant_candidate_gram_hash_label_graph",
                            "geometry_edge_features": ("normalised_gram_real", "normalised_gram_imag",
                                                       "normalised_gram_magnitude", "support_overlap_fraction",
                                                       "equal_amplitude_label"),
                            "base_decoder": "D3", "decoder_global_message_axis_materialized": False})
        return output


__all__ = ["BoundedMatchedFilterProposer", "CandidateDecoderOutput", "CandidateRestrictedEffectiveChannelPGD",
           "CandidateSet", "LearnedCandidateGeometryPGD", "candidate_count_targets", "candidate_recall"]
