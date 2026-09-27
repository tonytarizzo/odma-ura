"""D3 bounded-list algebra, proposer boundary, gradients, and B=100 checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.candidate_decoders import (BoundedMatchedFilterProposer, CandidateRestrictedEffectiveChannelPGD,
                                          CandidateSet, candidate_count_targets, candidate_recall)
from framework.core import ComponentSpec, URASpec
from framework.encoder import build_encoder
from framework.hash_skeleton import all_message_bits
from framework.losses import effective_channel_loss
from framework.learned_decoders import UnrolledEffectiveChannelPGD
from framework.pipeline import product_all_pairs_component_specs
from framework.prototype_amplitudes import build_prototype_hash_encoder, build_random_procedural_hash_codebook


def procedural_oracle_and_proposer_checks() -> None:
    spec = URASpec(n=16, num_codewords=16, num_active=1, payload_bits=4)
    encoder = build_prototype_hash_encoder(spec, support_size=4, seed=7, label_bits=4,
                                           learn_amplitudes=True, search_candidates=8)
    all_bits = all_message_bits(4)
    truth = all_bits[torch.tensor([3, 11])]
    Y = encoder.codebook.codewords(truth).transpose(0, 1).unsqueeze(-1)
    H = torch.ones(2, 1)
    exhaustive = CandidateSet(message_bits=all_bits, source="exhaustive_small_B",
                              metadata={"global_search": True, "certification_only": True})
    proposed = BoundedMatchedFilterProposer(8)(encoder, Y, H, exhaustive)
    if not torch.all(candidate_recall(proposed.message_bits, truth.unsqueeze(1)) == 1.0):
        raise AssertionError("bounded matched-filter proposer missed a noiseless singleton from its exhaustive test pool")
    if not proposed.metadata["global_search"] or proposed.metadata["proposer_global_search"] or proposed.metadata["proposal_scope"] != "bounded_supplied_pool":
        raise AssertionError("bounded proposer overstated its search boundary")

    decoder = CandidateRestrictedEffectiveChannelPGD(num_layers=5)
    output = decoder(encoder, Y, H, 1, noise_var=1e-6, candidates=proposed,
                     true_message_bits=truth.unsqueeze(1))
    if not torch.all(output.counts[:, 0] == 1.0) or not torch.all(output.counts.sum(dim=1) == 1.0):
        raise AssertionError("D3 failed its noiseless singleton oracle-list check")
    if output.meta["decoder_global_message_axis_materialized"] or output.meta["candidate_search_solved"]:
        raise AssertionError("D3 misstated its candidate-conditioned scalability boundary")
    if not output.meta["encoder_global_message_axis_present"] or output.meta["end_to_end_global_message_axis_free"]:
        raise AssertionError("small-B adapter metadata hid its pre-existing global encoder axis")
    targets = candidate_count_targets(proposed, true_message_bits=truth.unsqueeze(1))
    loss, _ = effective_channel_loss(output, targets)
    loss.backward()
    for name, parameter in list(decoder.named_parameters()) + list(encoder.named_parameters()):
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"D3 training loss did not provide a finite gradient for {name}")


def exhaustive_d2_equivalence_check() -> None:
    """An exhaustive list makes D3 exactly D2 on nonorthogonal real and complex codebooks."""
    spec = URASpec(n=8, num_codewords=8, num_active=2, payload_bits=3)
    component = ComponentSpec(Q=1, d=8, V=8, N=8, R_init="identity", C_init="random_gaussian",
                              U_init="all_pairs", T_init="identity")
    for seed, dtype in ((11, torch.float64), (12, torch.complex128)):
        generator = torch.Generator().manual_seed(seed)
        encoder = build_encoder(spec, [component], dtype=dtype, generator=generator)
        counts = torch.zeros(2, 8, dtype=dtype); counts[0, [1, 5]] = 1.0; counts[1, [2, 7]] = 1.0
        noise = torch.randn(2, 8, 1, dtype=dtype, generator=generator) * 0.01
        Y = encoder.encode(counts).unsqueeze(-1) + noise
        H, noise_var = torch.ones(2, 1, dtype=dtype), 1e-4
        d2 = UnrolledEffectiveChannelPGD(num_layers=3, geometry_chunk_size=3).to(torch.float64)
        d3 = CandidateRestrictedEffectiveChannelPGD(num_layers=3).to(torch.float64)
        with torch.no_grad():
            d3.raw_tau_scale.copy_(d2.raw_tau_scale); d3.raw_damping.copy_(d2.raw_damping)
        out2 = d2(encoder, Y, H, 2, noise_var)
        out3 = d3(encoder, Y, H, 2, noise_var,
                  candidates=CandidateSet(message_indices=torch.arange(8), source="exhaustive_small_B"))
        keys = ("soft_counts", "support_logits", "effective_variance", "physical_variance", "interference_variance")
        for key in keys:
            if not torch.allclose(out3.meta[key], out2.meta[key], atol=1e-10, rtol=1e-10):
                error = float((out3.meta[key] - out2.meta[key]).abs().max())
                raise AssertionError(f"exhaustive D3 does not reduce to D2 for {dtype}/{key}: max error {error:.3e}")
        for actual, expected in zip(out3.meta["layer_evidence_logits"], out2.meta["layer_evidence_logits"]):
            if not torch.allclose(actual, expected, atol=1e-10, rtol=1e-10):
                raise AssertionError(f"exhaustive D3 and D2 evidence disagrees for {dtype}")
        if not torch.equal(out3.counts, out2.counts):
            raise AssertionError(f"exhaustive D3 and D2 hard decisions disagree for {dtype}")


def indexed_global_encoder_check() -> None:
    generator = torch.Generator().manual_seed(13)
    spec = URASpec(n=12, num_codewords=16, num_active=1, payload_bits=4)
    encoder = build_encoder(spec, product_all_pairs_component_specs(spec, 4, False), generator=generator)
    common = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
    candidates = CandidateSet(message_indices=common, source="supplied_small_B_indices")
    counts = torch.zeros(2, spec.num_codewords); counts[0, 2] = 1.0; counts[1, 12] = 1.0
    Y = encoder.encode(counts).unsqueeze(-1); H = torch.ones(2, 1)
    output = CandidateRestrictedEffectiveChannelPGD(num_layers=3)(encoder, Y, H, 1, 1e-6, candidates=candidates)
    expected = encoder.explicit_matrix()[:, common]
    if not torch.allclose(output.meta["candidate_codewords"], expected.unsqueeze(0).expand(2, -1, -1), atol=1e-6):
        raise AssertionError("indexed D3 columns disagree with the global encoder")
    target = candidate_count_targets(candidates, global_counts=counts)
    if not torch.equal(target.nonzero()[:, 1], torch.tensor([1, 6])):
        raise AssertionError("common indexed candidate targets lost the global-to-local mapping")
    bit_candidates = CandidateSet(message_bits=all_message_bits(4)[:8])
    bit_truth = all_message_bits(4)[torch.tensor([[2], [7]])]
    bit_target = candidate_count_targets(bit_candidates, true_message_bits=bit_truth)
    if not torch.equal(bit_target.nonzero()[:, 1], torch.tensor([2, 7])):
        raise AssertionError("common bit candidate targets lost the message mapping")
    for invalid in (CandidateSet(message_indices=common.to(torch.float32) + 0.5),
                    CandidateSet(message_bits=torch.tensor([[0, 1, 0, 256]]))):
        try:
            candidate_count_targets(invalid, global_counts=counts) if invalid.message_indices is not None \
                else candidate_count_targets(invalid, true_message_bits=bit_truth)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("candidate identifiers silently accepted a non-integral or non-binary value")


def miss_and_large_payload_checks() -> None:
    generator = torch.Generator().manual_seed(19)
    codebook = build_random_procedural_hash_codebook(100, 256, 64, 8, seed=19)
    pool = torch.randint(2, (2, 32, 100), dtype=torch.uint8, generator=generator)
    truth = pool[:, :3].clone()
    # Replace one active message in sample zero: diagnostics must expose the miss.
    missing = torch.randint(2, (100,), dtype=torch.uint8, generator=generator)
    while bool((missing == pool[0]).all(dim=1).any()):
        missing = torch.randint(2, (100,), dtype=torch.uint8, generator=generator)
    truth[0, 2] = missing
    true_columns = codebook.codewords(truth.reshape(-1, 100)).transpose(0, 1).reshape(2, 3, 256).transpose(1, 2)
    y = true_columns.sum(dim=2)
    candidates = CandidateSet(message_bits=pool, source="bounded_external_pool",
                              metadata={"global_search": False})
    output = CandidateRestrictedEffectiveChannelPGD(num_layers=2)(
        codebook, y.unsqueeze(-1), torch.ones(2, 1), 3, 1e-4, candidates=candidates,
        true_message_bits=truth)
    expected_recall = torch.tensor([2.0 / 3.0, 1.0])
    if not torch.allclose(output.meta["candidate_recall"], expected_recall):
        raise AssertionError("D3 candidate recall did not expose an omitted true message")
    targets = candidate_count_targets(output.candidates, true_message_bits=truth)
    if not torch.equal(targets.sum(dim=1), torch.tensor([2.0, 3.0])):
        raise AssertionError("candidate targets hid proposal misses")
    if output.counts.shape != (2, 32) or output.meta["candidate_list_size"] != 32:
        raise AssertionError("B=100 D3 did not remain on the bounded candidate axis")
    if not output.meta["end_to_end_global_message_axis_free"]:
        raise AssertionError("direct B=100 procedural decoding incorrectly reported a global encoder axis")


def main() -> None:
    exhaustive_d2_equivalence_check()
    procedural_oracle_and_proposer_checks()
    indexed_global_encoder_check()
    miss_and_large_payload_checks()
    print("D3 bounded-list generation, oracle/proposer boundary, gradients, mapping, recall, and B=100 checks passed")


if __name__ == "__main__":
    main()
