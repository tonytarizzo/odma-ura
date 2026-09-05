"""Deterministic algebra and materialisation checks for hash-skeleton codebooks."""

from __future__ import annotations

import torch

from framework.core import URASpec
from framework.encoder import ComponentConstraints, build_encoder
from framework.hash_skeleton import (HASH_SKELETON_FAMILIES, all_message_bits, gf2_rank,
                                     hash_skeleton_component_specs, linear_hash_bins, random_linear_hash_bank)
from tests.framework_sparsity_diagnostics import analyse_encoder_sparsity


def collision_score(metadata: dict) -> tuple:
    geometry = metadata["selected_collision_geometry"]
    return (geometry["collision_tables_max"], geometry["num_differences_at_max"],
            geometry["collision_tables_q999"], geometry["collision_tables_q99"], geometry["collision_sum_squares"])


def build_family(family: str, candidates: int = 8, learn_amplitudes: bool = False):
    spec = URASpec(n=16, num_codewords=64, num_active=4, payload_bits=6)
    components, metadata = hash_skeleton_component_specs(spec, family, support_size=4, seed=123,
                                                         search_candidates=candidates, learn_amplitudes=learn_amplitudes)
    constraints = [ComponentConstraints(C="unit_norm_columns" if learn_amplitudes else "none")]
    encoder = build_encoder(spec, components, constraints=constraints, generator=torch.Generator().manual_seed(123))
    return encoder, metadata


def main() -> None:
    built = {family: build_family(family) for family in HASH_SKELETON_FAMILIES}
    reference_magnitudes = None
    for family, (encoder, metadata) in built.items():
        Phi = encoder.explicit_matrix()
        support = Phi != 0
        assert Phi.shape == (16, 64)
        assert torch.all(support.sum(dim=0) == 4)
        assert float(torch.max(torch.abs(Phi.square().sum(dim=0) - 1.0))) < 1e-6
        magnitudes = torch.sort(torch.abs(Phi), dim=0).values
        if reference_magnitudes is None:
            reference_magnitudes = magnitudes
        else:
            assert torch.allclose(magnitudes, reference_magnitudes), "families did not receive paired amplitudes"
        if metadata["one_position_per_predetermined_table"]:
            for table in range(4):
                assert torch.all(support[4 * table:4 * (table + 1)].sum(dim=0) == 1)

        counts = torch.randn(3, 64, generator=torch.Generator().manual_seed(91))
        assert torch.allclose(encoder.matvec(counts), counts @ Phi.transpose(0, 1), atol=1e-6)
        diagnostics = analyse_encoder_sparsity(encoder, num_pairs=100, active_samples=8, active_k=4,
                                                active_gram_samples=4, sum_pair_samples=4, seed=92)
        assert diagnostics["active_gram_samples"] == 4 and diagnostics["sum_separation_pairs"] == 4

    random_metadata = built["hash_linear_random_fixed"][1]
    selected_metadata = built["hash_linear_selected_fixed"][1]
    assert selected_metadata["candidate_zero_collision_geometry"] == random_metadata["selected_collision_geometry"]
    assert collision_score(selected_metadata) <= collision_score(random_metadata)
    assert selected_metadata == build_family("hash_linear_selected_fixed")[1], "offline hash search is not deterministic"

    encoder, metadata = built["hash_linear_selected_fixed"]
    A = torch.tensor(metadata["A"], dtype=torch.uint8); b = torch.tensor(metadata["b"], dtype=torch.uint8)
    assert A.shape == (4, 2, 6) and b.shape == (4, 2)
    assert all(gf2_rank(block) == 2 for block in A)
    assert gf2_rank(A.reshape(8, 6)) == 6
    selected_messages = torch.tensor([0, 1, 7, 31, 63])
    generated_rows = torch.arange(4).unsqueeze(1) * 4 + linear_hash_bins(A, b, all_message_bits(6)[selected_messages])
    observed_rows = torch.nonzero(encoder.explicit_matrix()[:, selected_messages].transpose(0, 1), as_tuple=False)
    observed_rows = observed_rows[:, 1].reshape(selected_messages.numel(), 4).transpose(0, 1)
    assert torch.equal(generated_rows, observed_rows)
    assert metadata["support_tuple_injective"] and metadata["stacked_rank"] == 6

    large_A, large_b = random_linear_hash_bank(100, 64, 2, torch.Generator().manual_seed(93))
    large_messages = torch.randint(2, (5, 100), generator=torch.Generator().manual_seed(94), dtype=torch.uint8)
    large_bins = linear_hash_bins(large_A, large_b, large_messages)
    assert large_A.shape == (64, 2, 100) and large_bins.shape == (64, 5)
    assert gf2_rank(large_A.reshape(128, 100)) == 100 and int(large_bins.max()) < 4

    learned, _ = build_family("hash_linear_selected_fixed", learn_amplitudes=True)
    component = learned.components[0]; mask = component.fixed_C_support.clone(); initial = component.C.detach().clone()
    counts = torch.randn(3, 64, generator=torch.Generator().manual_seed(95))
    target = torch.randn(3, 16, generator=torch.Generator().manual_seed(96))
    loss = (learned.matvec(counts) - target).square().mean(); loss.backward()
    assert not bool((component.C.grad.masked_select(~mask) != 0).any())
    optimiser = torch.optim.Adam(learned.parameters(), lr=1e-2)
    optimiser.step(); learned.apply_constraints()
    assert not bool((component.C.detach().masked_select(~mask) != 0).any())
    assert float(torch.max(torch.abs(component.C.detach().square().sum(dim=0) - 1.0))) < 1e-6
    assert not torch.allclose(component.C.detach().masked_select(mask), initial.masked_select(mask))
    print("hash skeleton: support algebra, exact energy, pairing, selection, materialisation, and fixed-support learning passed")


if __name__ == "__main__":
    main()
