"""Focused algebra tests for the bounded candidate geometry correction."""

from __future__ import annotations

import torch

from framework.candidate_decoders import (CandidateRestrictedEffectiveChannelPGD, CandidateSet,
                                          LearnedCandidateGeometryPGD, candidate_count_targets)
from framework.candidate_geometry import CandidateGeometryCorrection, candidate_structure_features
from framework.hash_skeleton import all_message_bits
from framework.losses import effective_channel_loss
from framework.prototype_amplitudes import build_prototype_hash_encoder, build_random_procedural_hash_codebook


def synthetic_inputs(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(817)
    batch, n, candidates = 3, 9, 7
    columns = torch.randn(batch, n, candidates, dtype=dtype, generator=generator)
    columns = columns / columns.norm(dim=1, keepdim=True)
    gram = torch.bmm(columns.transpose(1, 2), columns)
    nodes = [torch.randn(batch, candidates, dtype=dtype, generator=generator) for _ in range(4)]
    nodes[0] = torch.sigmoid(nodes[0])
    nodes[3] = torch.nn.functional.softplus(nodes[3])
    scores = torch.randn(batch, candidates, dtype=dtype, generator=generator)
    overlap = torch.rand(batch, candidates, candidates, dtype=dtype, generator=generator)
    overlap = 0.5 * (overlap + overlap.transpose(1, 2))
    labels = torch.randint(4, (batch, candidates), generator=generator)
    same_label = (labels.unsqueeze(-1) == labels.unsqueeze(-2)).to(dtype)
    return nodes, gram, scores, overlap, same_label


def d4_integration_check() -> None:
    spec_args = dict(n=16, num_codewords=16, num_active=1, payload_bits=4)
    from framework.core import URASpec
    encoder = build_prototype_hash_encoder(URASpec(**spec_args), support_size=4, seed=821, label_bits=2,
                                           learn_amplitudes=False, search_candidates=8)
    all_bits = all_message_bits(4)
    truth = all_bits[torch.tensor([3, 11])]
    Y = encoder.codebook.codewords(truth).transpose(0, 1).unsqueeze(-1)
    H = torch.ones(2, 1)
    scores = torch.linspace(-1.0, 1.0, 16).unsqueeze(0).expand(2, -1)
    candidates = CandidateSet(message_bits=all_bits, scores=scores, source="exhaustive_small_B")
    d3 = CandidateRestrictedEffectiveChannelPGD(num_layers=2)
    d4 = LearnedCandidateGeometryPGD(num_layers=2, hidden_dim=10)
    out3 = d3(encoder, Y, H, 1, 1e-5, candidates=candidates)
    out4 = d4(encoder, Y, H, 1, 1e-5, candidates=candidates)
    assert out4.meta["candidate_support_overlap"].shape == (2, 16, 16)
    assert out4.meta["candidate_same_amplitude_label"].shape == (2, 16, 16)
    for key in ("soft_counts", "support_logits", "effective_variance"):
        assert torch.equal(out4.meta[key], out3.meta[key]), f"zero-initialised D4 changed D3's {key}"
    assert torch.equal(out4.counts, out3.counts)
    for corrected, raw in zip(out4.meta["layer_evidence_logits"], out4.meta["layer_raw_evidence_logits"]):
        assert torch.equal(corrected, raw), "zero-initialised D4 changed D3 evidence"

    targets = candidate_count_targets(out4.candidates, true_message_bits=truth.unsqueeze(1))
    loss, _ = effective_channel_loss(out4, targets); loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in d4.parameters())

    with torch.no_grad():
        generator = torch.Generator().manual_seed(822)
        for layer in d4.geometry_layers:
            layer.output.weight.normal_(generator=generator)
    reference = d4(encoder, Y, H, 1, 1e-5, candidates=candidates)
    assert any(not torch.equal(corrected, raw) for corrected, raw in zip(
        reference.meta["layer_evidence_logits"], reference.meta["layer_raw_evidence_logits"])), \
        "trained-capable D4 path did not alter loss-facing evidence"
    for correction in reference.meta["layer_logit_corrections"]:
        assert torch.allclose(correction.mean(dim=1), torch.zeros(correction.shape[0]), atol=2e-6), \
            "D4 retained a candidate-common correction that known-K projection cannot identify"
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(823))
    permuted_candidates = CandidateSet(message_bits=all_bits.index_select(0, permutation),
                                       scores=scores.index_select(1, permutation), source="exhaustive_small_B")
    permuted = d4(encoder, Y, H, 1, 1e-5, candidates=permuted_candidates)
    assert torch.allclose(permuted.meta["soft_counts"], reference.meta["soft_counts"].index_select(1, permutation),
                          atol=2e-6, rtol=2e-6)
    for actual, expected in zip(permuted.meta["layer_evidence_logits"], reference.meta["layer_evidence_logits"]):
        assert torch.allclose(actual, expected.index_select(1, permutation), atol=2e-5, rtol=2e-6)


def main() -> None:
    model = CandidateGeometryCorrection(hidden_dim=12).to(dtype=torch.float64)
    nodes, gram, scores, overlap, same_label = synthetic_inputs()
    correction = model(*nodes, gram, scores, overlap, same_label)
    assert torch.equal(correction, torch.zeros_like(correction)), "zero-initialised D4 correction must preserve D3"

    loss = (correction - torch.randn_like(correction)).square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.output.weight.grad.abs().sum() > 0, "zero-head D4 must learn its output direction on step one"

    with torch.no_grad():
        model.output.weight.normal_(generator=torch.Generator().manual_seed(818))
    model.zero_grad(set_to_none=True)
    reference = model(*nodes, gram, scores, overlap, same_label)
    assert torch.allclose(reference.mean(dim=1), torch.zeros(reference.shape[0], dtype=reference.dtype), atol=1e-12)
    reference.square().mean().backward()
    upstream = [parameter for name, parameter in model.named_parameters() if not name.startswith("output.")]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
               for parameter in upstream), "nonzero D4 head did not pass gradients into its graph features"
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    permuted_nodes = [node.index_select(1, permutation) for node in nodes]
    permuted_gram = gram.index_select(1, permutation).index_select(2, permutation)
    permuted_overlap = overlap.index_select(1, permutation).index_select(2, permutation)
    permuted_labels = same_label.index_select(1, permutation).index_select(2, permutation)
    permuted = model(*permuted_nodes, permuted_gram, scores.index_select(1, permutation),
                     permuted_overlap, permuted_labels)
    assert torch.allclose(permuted, reference.index_select(1, permutation), atol=1e-11, rtol=1e-11)

    complex_model = CandidateGeometryCorrection(hidden_dim=10).to(dtype=torch.float64)
    with torch.no_grad():
        complex_model.output.weight.normal_(generator=torch.Generator().manual_seed(819))
    complex_columns = torch.randn(3, 9, 7, dtype=torch.complex128, generator=torch.Generator().manual_seed(820))
    complex_columns = complex_columns / complex_columns.norm(dim=1, keepdim=True)
    complex_gram = torch.bmm(complex_columns.conj().transpose(1, 2), complex_columns)
    complex_reference = complex_model(*nodes, complex_gram, scores, overlap, same_label)
    complex_reference.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in complex_model.parameters()), "complex-Gram D4 path has invalid gradients"
    complex_permuted = complex_model(*permuted_nodes,
                                     complex_gram.index_select(1, permutation).index_select(2, permutation),
                                     scores.index_select(1, permutation), permuted_overlap, permuted_labels)
    assert torch.allclose(complex_permuted, complex_reference.index_select(1, permutation), atol=1e-11, rtol=1e-11)

    codebook = build_random_procedural_hash_codebook(8, 32, 8, 3, seed=821)
    bits = torch.randint(2, (2, 6, 8), dtype=torch.uint8, generator=torch.Generator().manual_seed(822))
    hash_overlap, equal_label = candidate_structure_features(codebook, bits)
    assert hash_overlap.shape == equal_label.shape == (2, 6, 6)
    assert torch.equal(torch.diagonal(hash_overlap, dim1=-2, dim2=-1), torch.ones(2, 6))
    assert torch.equal(torch.diagonal(equal_label, dim1=-2, dim2=-1), torch.ones(2, 6))
    structure_permutation = torch.tensor([4, 0, 2, 1, 5, 3])
    permuted_hash, permuted_equal = candidate_structure_features(codebook, bits.index_select(1, structure_permutation))
    assert torch.equal(permuted_hash, hash_overlap.index_select(1, structure_permutation).index_select(2, structure_permutation))
    assert torch.equal(permuted_equal, equal_label.index_select(1, structure_permutation).index_select(2, structure_permutation))

    large_codebook = build_random_procedural_hash_codebook(100, 256, 64, 8, seed=824)
    large_bits = torch.randint(2, (2, 12, 100), dtype=torch.uint8, generator=torch.Generator().manual_seed(825))
    active_columns = large_codebook.codewords(large_bits[:, :2].reshape(-1, 100)).transpose(0, 1)
    y = active_columns.reshape(2, 2, 256).sum(dim=1)
    large_output = LearnedCandidateGeometryPGD(num_layers=1, hidden_dim=8)(
        large_codebook, y.unsqueeze(-1), torch.ones(2, 1), 2, 1e-4,
        candidates=CandidateSet(message_bits=large_bits, source="bounded_external_pool"))
    assert large_output.counts.shape == (2, 12) and large_output.meta["end_to_end_global_message_axis_free"]
    d4_integration_check()
    print("candidate geometry: zero-init, finite gradients, permutation equivariance, and structural features passed")


if __name__ == "__main__":
    main()
