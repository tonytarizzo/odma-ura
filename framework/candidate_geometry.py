"""Candidate-local learned corrections for geometry-aware decoding.

The modules here never allocate a global ``M=2^B`` state.  They operate on a
bounded candidate list and are permutation equivariant in the candidate axis.
"""

from __future__ import annotations

import torch
from torch import nn


def candidate_structure_features(codebook, candidate_bits: torch.Tensor
                                 ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return pairwise hash-overlap fractions and equal-amplitude-label flags.

    Structural features are available for ``ProceduralHashCodebook`` and its
    small-B ``PrototypeHashEncoder`` adapter.  Other supplied codebooks simply
    return ``(None, None)``; their exact candidate Gram matrix still informs the
    learned correction.
    """
    procedural = getattr(codebook, "codebook", codebook)
    if candidate_bits.ndim != 3:
        raise ValueError(f"candidate_bits must have shape (batch,C,B), got {tuple(candidate_bits.shape)}")
    if not hasattr(procedural, "support_rows"):
        return None, None
    batch, candidates, payload_bits = candidate_bits.shape
    if hasattr(procedural, "payload_bits") and int(procedural.payload_bits) != payload_bits:
        raise ValueError(f"candidate bits have B={payload_bits}, but codebook expects B={procedural.payload_bits}")
    flat = candidate_bits.reshape(batch * candidates, payload_bits)
    rows = procedural.support_rows(flat).reshape(procedural.tables, batch, candidates).permute(1, 0, 2)
    overlap = (rows.unsqueeze(-1) == rows.unsqueeze(-2)).to(torch.float32).mean(dim=1)
    bank = getattr(procedural, "amplitude_bank", None)
    if bank is None or not hasattr(bank, "labels"):
        return overlap, None
    labels = bank.labels(flat).reshape(batch, candidates)
    same_label = (labels.unsqueeze(-1) == labels.unsqueeze(-2)).to(overlap.dtype)
    return overlap, same_label


class CandidateGeometryCorrection(nn.Module):
    """A bounded-list permutation-equivariant graph correction.

    Nodes are candidates and edges carry their normalised Gram coefficient,
    optional hash overlap, and optional amplitude-label equality.  The final
    head is zero-initialised, so adding this module to a decoder preserves the
    base decoder exactly before learning. Its output is centred over candidates
    because the known-cardinality projection cannot identify a shared logit
    shift.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)
        self.node_encoder = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_gate = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mix = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _check_nodes(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")

    def forward(self, a: torch.Tensor, statistic: torch.Tensor, evidence: torch.Tensor,
                tau: torch.Tensor, gram: torch.Tensor, proposal_scores: torch.Tensor | None = None,
                support_overlap: torch.Tensor | None = None,
                same_amplitude_label: torch.Tensor | None = None) -> torch.Tensor:
        if a.ndim != 2:
            raise ValueError(f"candidate state must have shape (batch,C), got {tuple(a.shape)}")
        batch, candidates = a.shape
        if candidates <= 1:
            raise ValueError("geometry correction requires at least two candidates")
        for name, value in (("statistic", statistic), ("evidence", evidence), ("tau", tau)):
            self._check_nodes(name, value, (batch, candidates))
        if gram.shape != (batch, candidates, candidates):
            raise ValueError(f"gram must have shape ({batch},{candidates},{candidates}), got {tuple(gram.shape)}")

        real_dtype = a.dtype
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).real.to(real_dtype).clamp_min(torch.finfo(real_dtype).eps)
        normalizer = torch.sqrt(diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2))
        normalized_gram = gram / normalizer.to(gram.dtype)
        gram_real = normalized_gram.real.to(real_dtype)
        gram_imag = normalized_gram.imag.to(real_dtype) if normalized_gram.is_complex() else torch.zeros_like(gram_real)
        gram_abs = torch.abs(normalized_gram).to(real_dtype)

        if proposal_scores is None:
            scores = torch.zeros_like(a)
        else:
            self._check_nodes("proposal_scores", proposal_scores, (batch, candidates))
            centered = proposal_scores.to(real_dtype) - proposal_scores.to(real_dtype).mean(dim=1, keepdim=True)
            scores = centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        rho = a.sum(dim=1, keepdim=True) / float(candidates)
        node_features = torch.stack((a, torch.tanh(statistic), torch.tanh(evidence / 8.0),
                                     torch.log1p(tau), diagonal, torch.tanh(scores), rho.expand_as(a)), dim=-1)
        node = self.node_encoder(node_features)

        overlap_available = support_overlap is not None
        label_available = same_amplitude_label is not None
        overlap = torch.zeros_like(gram_real) if support_overlap is None else support_overlap.to(real_dtype)
        same_label = torch.zeros_like(gram_real) if same_amplitude_label is None else same_amplitude_label.to(real_dtype)
        for name, value in (("support_overlap", overlap), ("same_amplitude_label", same_label)):
            if value.shape != (batch, candidates, candidates):
                raise ValueError(f"{name} must have shape ({batch},{candidates},{candidates}), got {tuple(value.shape)}")
        has_overlap = torch.full_like(gram_real, float(overlap_available))
        has_label = torch.full_like(gram_real, float(label_available))
        edge_features = torch.stack((gram_real, gram_imag, gram_abs, overlap, same_label, has_overlap, has_label), dim=-1)
        gates = torch.tanh(self.edge_gate(edge_features).squeeze(-1))
        off_diagonal = ~torch.eye(candidates, dtype=torch.bool, device=a.device).unsqueeze(0)
        gates = gates.masked_fill(~off_diagonal, 0.0)
        degree = gates.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        context = torch.bmm(gates, self.value(node)) / degree
        global_context = node.mean(dim=1, keepdim=True).expand_as(node)
        mixed = node + self.mix(torch.cat((node, context, global_context), dim=-1))
        correction = self.output(mixed).squeeze(-1)
        return correction - correction.mean(dim=1, keepdim=True)


__all__ = ["CandidateGeometryCorrection", "candidate_structure_features"]
