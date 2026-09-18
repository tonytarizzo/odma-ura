"""Exact operator, nesting, gradient, initialization and M-free generator checks."""

import torch

from framework.coordinate_amplitudes import (CoordinateAmplitudeBank, balance_rows, build_coordinate_hash_encoder,
                                              build_random_coordinate_codebook)
from framework.core import URASpec
from framework.hash_skeleton import all_message_bits, gf2_rank
from framework.prototype_amplitudes import build_prototype_hash_encoder


def main():
    torch.set_num_threads(2)
    spec = URASpec(n=32, num_codewords=256, num_active=4, payload_bits=8)
    build = lambda J, learn=False, balanced=False: build_coordinate_hash_encoder(spec, 8, 31, J, learn, balanced,
                                                                                 search_candidates=2)
    small, large, parent = build(2), build(4), build(8)
    P2, P4 = (e.codebook.amplitude_bank.projection for e in (small, large))
    assert torch.equal(P2, P4[:, :2])
    assert gf2_rank(P2.reshape(-1, 8)) == 8
    assert all(gf2_rank(torch.cat((a, p))) == 4 for a, p in zip(small.codebook.A, P2))
    bits = all_message_bits(8)
    bank = small.codebook.amplitude_bank
    nested = CoordinateAmplitudeBank(P4, bank.prototypes.repeat(1, 4), False)
    assert torch.allclose(bank(bits), nested(bits), atol=1e-7)
    legacy = build_prototype_hash_encoder(spec, 8, 31, 8, False, search_candidates=2)
    assert torch.allclose(parent.explicit_matrix(), legacy.explicit_matrix(), atol=2e-7)
    shared = build_coordinate_hash_encoder(spec, 8, 31, 2, False, False, True, 2)
    legacy_shared = build_prototype_hash_encoder(spec, 8, 31, 2, False, search_candidates=2)
    assert torch.allclose(shared.explicit_matrix(), legacy_shared.explicit_matrix(), atol=2e-7)
    assert torch.all(torch.bincount(bank.labels(bits)[0]) == 64)
    balanced = build(2, balanced=True).codebook.amplitude_bank.prototypes
    assert balanced.mean(dim=1).abs().max() < 1e-6
    assert (balanced.square().mean(dim=1) - 1).abs().max() < 1e-6
    assert torch.allclose(balanced, balance_rows(bank.prototypes))

    learned = build(2, learn=True, balanced=True)
    assert not list(small.parameters())
    assert [name for name, _ in learned.named_parameters()] == ["codebook.amplitude_bank.prototypes"]
    Phi = learned.explicit_matrix()
    assert torch.all((Phi != 0).sum(dim=0) == 8)
    counts, residual = torch.randn(3, 256), torch.randn(3, 32)
    assert torch.allclose(learned.matvec(counts), counts @ Phi.T, atol=2e-6)
    assert torch.allclose(learned.rmatvec(residual), residual @ Phi, atol=2e-6)
    indices = torch.tensor([9, 0, 9, 21])
    assert torch.equal(learned.message_columns(indices), Phi[:, indices])
    values = learned.codebook.amplitude_bank.prototypes
    loss = learned.matvec(counts).square().mean()
    gradient = torch.autograd.grad(loss, values, retain_graph=True)[0]
    reference = torch.autograd.grad((counts @ Phi.T).square().mean(), values)[0]
    assert torch.allclose(gradient, reference, atol=2e-6)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    initial = values.detach().clone()
    (learned.matvec(counts) - residual).square().mean().backward()
    torch.optim.Adam(learned.parameters(), lr=.01).step()
    updated = values.detach().clone(); learned.apply_constraints()
    assert torch.equal(values, updated) and not torch.equal(initial, updated), "no spurious bank-column projection"
    assert learned.prototype_diagnostics()["max_energy_deviation"] < 1e-6
    restored = build(2, learn=True, balanced=True)
    restored.load_state_dict(learned.state_dict())
    assert torch.equal(restored.explicit_matrix(), learned.explicit_matrix())
    first_scale = restored.spectral_norm_squared(4, use_cache=False)
    torch.randn(100)
    assert torch.equal(first_scale, restored.spectral_norm_squared(4, use_cache=False))
    assert not any("_message" in key for key in learned.state_dict())
    zero = CoordinateAmplitudeBank(P2, torch.zeros(8, 4), False, target_energy=3)
    assert torch.allclose(zero(bits).square().sum(dim=0), torch.full((256,), 3.0))

    scalable = build_random_coordinate_codebook(100, 256, 64, 4, 32, learn_amplitudes=True)
    sampled = torch.randint(2, (7, 100), dtype=torch.uint8)
    columns = scalable.codewords(sampled)
    assert columns.shape == (256, 7)
    assert torch.all((columns != 0).sum(dim=0) == 64)
    assert (columns.square().sum(dim=0) - 1).abs().max() < 1e-6
    columns.sum().backward()
    assert scalable.amplitude_bank.prototypes.grad.norm() > 0
    assert scalable.amplitude_bank.projection.shape == (64, 4, 100)
    assert scalable.amplitude_bank.prototypes.shape == (64, 16)
    assert max(t.numel() for t in scalable.state_dict().values()) <= 64 * 4 * 100
    print("coordinate amplitudes: endpoints, nesting, ranks, operators, gradients, initialization, energy, B=100 passed")


if __name__ == "__main__":
    main()
