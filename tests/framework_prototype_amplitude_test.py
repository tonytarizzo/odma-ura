"""Algebra, gradient, endpoint, energy, and B=100 checks for prototype amplitudes."""

from __future__ import annotations

import torch

from framework.core import URASpec
from framework.encoder import build_encoder
from framework.hash_skeleton import all_message_bits, gf2_rank, hash_skeleton_component_specs
from framework.prototype_amplitudes import (PrototypeAmplitudeBank, ProceduralHashCodebook, amplitude_labels,
                                            build_prototype_hash_encoder, build_random_procedural_hash_codebook)


def build(J: int, learn: bool = False, init: str = "gaussian"):
    spec = URASpec(n=16, num_codewords=64, num_active=4, payload_bits=6)
    return build_prototype_hash_encoder(spec, support_size=4, seed=123, label_bits=J,
                                        learn_amplitudes=learn, amplitude_init=init, search_candidates=8)


def main() -> None:
    encoders = {J: build(J) for J in (0, 2, 4, 6)}
    projections = {J: encoders[J].codebook.amplitude_bank.projection for J in encoders}
    for small, large in ((0, 2), (2, 4), (4, 6)):
        assert torch.equal(projections[small], projections[large][:small])
        assert gf2_rank(projections[large]) == large

    bits = all_message_bits(6)
    for J, encoder in encoders.items():
        labels = amplitude_labels(projections[J], bits)
        loads = torch.bincount(labels, minlength=1 << J)
        assert torch.all(loads == 1 << (6 - J))
        Phi = encoder.explicit_matrix()
        assert torch.all((Phi != 0).sum(dim=0) == 4)
        assert float((Phi.square().sum(dim=0) - 1.0).abs().max()) < 1e-6

        counts = torch.randn(3, 64, generator=torch.Generator().manual_seed(91))
        residual = torch.randn(3, 16, generator=torch.Generator().manual_seed(92))
        assert torch.allclose(encoder.matvec(counts), counts @ Phi.T, atol=1e-6)
        assert torch.allclose(encoder.rmatvec(residual), residual @ Phi, atol=1e-6)

    old_specs, _ = hash_skeleton_component_specs(URASpec(16, 64, 4, payload_bits=6),
                                                  "hash_linear_selected_fixed", 4, 123, 8)
    old_encoder = build_encoder(URASpec(16, 64, 4, payload_bits=6), old_specs,
                                generator=torch.Generator().manual_seed(123))
    assert torch.allclose(encoders[6].explicit_matrix(), old_encoder.explicit_matrix(), atol=2e-7), "J=B must recover the old amplitude endpoint"

    fixed, learned = build(2), build(2, learn=True)
    assert torch.equal(fixed.explicit_matrix(), learned.explicit_matrix())
    assert not list(fixed.parameters())
    assert [name for name, _ in learned.named_parameters()] == ["codebook.amplitude_bank.prototypes"]
    initial = learned.codebook.amplitude_bank.prototypes.detach().clone()
    counts = torch.randn(3, 64, generator=torch.Generator().manual_seed(93))
    target = torch.randn(3, 16, generator=torch.Generator().manual_seed(94))
    loss = (learned.matvec(counts) - target).square().mean(); loss.backward()
    assert learned.codebook.amplitude_bank.prototypes.grad is not None
    opt = torch.optim.Adam(learned.parameters(), lr=1e-2); opt.step(); learned.apply_constraints()
    assert not torch.equal(initial, learned.codebook.amplitude_bank.prototypes.detach())
    assert float((learned.explicit_matrix().square().sum(dim=0) - 1.0).abs().max().detach()) < 1e-6

    parent = encoders[2].codebook.amplitude_bank
    V2 = parent.normalized_prototypes().detach()
    V3 = torch.cat((V2, V2), dim=1)
    nested = ProceduralHashCodebook(encoders[2].codebook.A, encoders[2].codebook.b,
                                    PrototypeAmplitudeBank(projections[4][:3], V3, False), n=16)
    assert torch.allclose(encoders[2].codebook.codewords(bits), nested.codewords(bits), atol=1e-7)

    equal = build(0, init="equal").explicit_matrix()
    rademacher = build(6, init="rademacher").explicit_matrix()
    assert torch.unique(equal[equal != 0]).numel() == 1
    assert torch.allclose(rademacher[rademacher != 0].abs(), torch.full_like(rademacher[rademacher != 0], 0.5))

    scalable = build_random_procedural_hash_codebook(100, 256, 64, 8, seed=125)
    selected_bits = torch.randint(2, (5, 100), generator=torch.Generator().manual_seed(126), dtype=torch.uint8)
    codewords = scalable.codewords(selected_bits)
    assert codewords.shape == (256, 5) and torch.all((codewords != 0).sum(dim=0) == 64)
    assert float((codewords.square().sum(dim=0) - 1.0).abs().max()) < 1e-6
    state = scalable.state_dict()
    assert tuple(state["A"].shape) == (64, 2, 100) and tuple(state["b"].shape) == (64, 2)
    assert tuple(state["amplitude_bank.projection"].shape) == (8, 100)
    assert tuple(state["amplitude_bank.prototypes"].shape) == (64, 256)
    assert max(value.numel() for value in state.values()) <= 64 * 256
    print("prototype amplitudes: nesting, endpoint, gradients, exact energy, controls, and M-free B=100 generation passed")


if __name__ == "__main__":
    main()
