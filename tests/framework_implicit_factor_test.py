"""Small exact checks for implicit factor operators and learned decoders."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.channel import (constant_fading, sample_batch, uniform_count_range_generator,
                               uniform_counts_generator)  # noqa: E402
from framework.core import URASpec  # noqa: E402
from framework.decoders import exact_count_ml  # noqa: E402
from framework.encoder import build_encoder, matvec_with_matrix, rmatvec_with_matrix  # noqa: E402
from framework.learned_decoders import (FactorAttentionISTANet, UnrolledBernoulliPGD, UnrolledEffectiveChannelPGD,
                                        _bernoulli_cardinality_projection)  # noqa: E402
from framework.losses import effective_channel_loss, support_count_loss  # noqa: E402
from framework.pipeline import ccs_component_specs, odma_component_specs, product_all_pairs_component_specs  # noqa: E402


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-10) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=atol):
        error = float(torch.max(torch.abs(actual - expected)).item())
        raise AssertionError(f"{name} mismatch: max error {error:.3e}")


def operator_checks(dtype: torch.dtype, operator_init: str) -> None:
    gen = torch.Generator().manual_seed(19)
    spec = URASpec(n=8, num_codewords=12, num_active=3, num_antennas=1, payload_bits=4)
    components = product_all_pairs_component_specs(spec, 3, False, operator_init)
    encoder = build_encoder(spec, components, dtype=dtype, generator=gen)
    Phi = encoder.explicit_matrix()
    a = torch.randn(5, spec.num_codewords, dtype=dtype, generator=gen)
    r = torch.randn(5, spec.n, dtype=dtype, generator=gen)
    check_close("batched matvec", encoder.matvec(a), matvec_with_matrix(Phi, a))
    check_close("batched adjoint", encoder.rmatvec(r), rmatvec_with_matrix(Phi, r))
    check_close("vector matvec", encoder.matvec(a[0]), matvec_with_matrix(Phi, a[0]))
    lhs = torch.sum(encoder.matvec(a[0]).conj() * r[0])
    rhs = torch.sum(a[0].conj() * encoder.rmatvec(r[0]))
    check_close("adjoint identity", lhs, rhs)
    selected = torch.tensor([0, 5, 11])
    check_close("selected columns", encoder.components[0].message_columns(selected), Phi[:, selected])
    if tuple(encoder.components[0].R.shape) != (3, 8):
        raise AssertionError("diagonal operators must be stored compactly as (Q,n)")


def mapped_preset_checks() -> None:
    gen = torch.Generator().manual_seed(21)
    spec = URASpec(n=12, num_codewords=12, num_active=3, num_antennas=1, payload_bits=4)
    for name, components in [("odma", odma_component_specs(spec, 4, 3, False, False)),
                             ("ccs", ccs_component_specs(spec, 2, False))]:
        encoder = build_encoder(spec, components, dtype=torch.float64, generator=gen)
        Phi = encoder.explicit_matrix()
        a = torch.randn(3, spec.num_codewords, dtype=encoder.dtype, generator=gen)
        r = torch.randn(3, spec.n, dtype=encoder.dtype, generator=gen)
        check_close(f"{name} mapped matvec", encoder.matvec(a), matvec_with_matrix(Phi, a))
        check_close(f"{name} mapped adjoint", encoder.rmatvec(r), rmatvec_with_matrix(Phi, r))
        selected = torch.tensor([0, 5, 11])
        check_close(f"{name} selected global columns", encoder.message_columns(selected), Phi[:, selected])


def decoder_checks() -> None:
    gen = torch.Generator().manual_seed(23)
    spec = URASpec(n=12, num_codewords=16, num_active=4, num_antennas=1, payload_bits=4)
    encoder = build_encoder(spec, product_all_pairs_component_specs(spec, 4, False), dtype=torch.float32, generator=gen)
    sampler = uniform_count_range_generator(2, 4, spec.num_codewords, gen, encoder.device)
    fading = constant_fading(1, encoder.dtype, encoder.device)
    realised = set()
    for _ in range(20):
        counts, _ = sampler(3)
        K = counts.sum(dim=1)
        if not torch.all(K == K[0]):
            raise AssertionError("the training contract requires one sampled K per batch")
        realised.add(int(K[0].item()))
    if realised != {2, 3, 4}:
        raise AssertionError(f"range sampler did not cover its seeded test range: {realised}")

    batch = sample_batch(encoder, 3, sampler, fading, 2.0, gen)

    def distinct_sample(batch_size: int):
        active = torch.stack([torch.randperm(spec.num_codewords, generator=gen)[:3] for _ in range(batch_size)])
        counts = torch.zeros(batch_size, spec.num_codewords)
        counts.scatter_(1, active, 1.0)
        return counts, active

    d2_batch = sample_batch(encoder, 3, distinct_sample, fading, 2.0, gen)
    for model in [UnrolledBernoulliPGD(num_layers=2, power_iters=3),
                  FactorAttentionISTANet(num_layers=2, hidden_dim=8, pattern_slots=1, global_slots=1, power_iters=3),
                  UnrolledEffectiveChannelPGD(num_layers=2)]:
        model_batch = d2_batch if isinstance(model, UnrolledEffectiveChannelPGD) else batch
        out = model(encoder, model_batch.Y, model_batch.H, model_batch.num_active, noise_var=model_batch.noise_var)
        if not torch.all(out.counts.sum(dim=1) == model_batch.num_active):
            raise AssertionError("hard decoder output must preserve the supplied per-sample K")
        if isinstance(model, UnrolledEffectiveChannelPGD):
            target, loss_fn = model_batch.counts, effective_channel_loss
        else:
            target = model_batch.counts.clone(); target[0].zero_(); target[0, 0] = 2.0
            loss_fn = support_count_loss
        loss, parts = loss_fn(out, target, lambda_count=0.1)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in parts.values()):
            raise AssertionError("decoder loss must remain finite")
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError("decoder loss did not reach every decoder parameter with finite gradients")
        if isinstance(model, UnrolledEffectiveChannelPGD):
            check_close("D2 soft cardinality", out.meta["soft_counts"].sum(dim=1), model_batch.num_active.to(torch.float32), 1e-5)
            if out.meta["variance_model"] != "gram_row_energy_mean_field_off_diagonal":
                raise AssertionError("D2 did not report its analytic variance approximation")


def d2_algebra_checks() -> None:
    logits = torch.tensor([[120.0, -80.0, 3.0, -2.0], [-100.0, 90.0, 20.0, -30.0]],
                          dtype=torch.float64, requires_grad=True)
    K = torch.tensor([1, 3])
    shifted, probabilities = _bernoulli_cardinality_projection(logits, K)
    check_close("D2 cardinality projection", probabilities.sum(dim=1), K.to(torch.float64), 1e-9)
    probabilities.square().sum().backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all() or not torch.isfinite(shifted).all():
        raise AssertionError("D2 cardinality projection must have finite values and gradients")
    if not torch.allclose(logits.grad.sum(dim=1), torch.zeros(2, dtype=logits.dtype), atol=1e-9, rtol=1e-9):
        raise AssertionError("D2 implicit cardinality-shift gradient must be invariant to constant logit offsets")

    gen = torch.Generator().manual_seed(31)
    spec = URASpec(n=7, num_codewords=10, num_active=3, num_antennas=1, payload_bits=4)
    encoder = build_encoder(spec, product_all_pairs_component_specs(spec, 2, False), dtype=torch.float64, generator=gen)
    decoder = UnrolledEffectiveChannelPGD(num_layers=1, geometry_chunk_size=3).to(dtype=torch.float64)
    diagonal, row_energy = decoder._gram_geometry(encoder)
    Phi = encoder.explicit_matrix()
    G = Phi.transpose(-1, -2) @ Phi
    check_close("D2 Gram diagonal", diagonal, torch.diagonal(G))
    check_close("D2 squared-Gram row energy", row_energy, torch.sum(G.square(), dim=1))
    variance = torch.stack([torch.full((spec.num_codewords,), 0.12),
                            torch.full((spec.num_codewords,), 0.27)]).to(torch.float64)
    noise = torch.tensor([0.03, 0.09], dtype=torch.float64)
    total, physical, interference = decoder._effective_variance(
        variance, diagonal, row_energy, noise, complex_observation=False)
    expected_physical = noise.unsqueeze(1) * diagonal.unsqueeze(0)
    expected_interference = variance * (torch.sum(G.square(), dim=1) - torch.diagonal(G).square())
    check_close("D2 physical variance", physical, expected_physical)
    check_close("D2 uniform-variance Gram interference", interference, expected_interference)
    check_close("D2 total effective variance", total, expected_physical + expected_interference)

    old_row_energy = row_energy.clone()
    with torch.no_grad():
        encoder.components[0].C.mul_(1.1)
    decoder.clear_geometry_cache()
    refreshed_diagonal, refreshed_row_energy = decoder._gram_geometry(encoder)
    refreshed_phi = encoder.explicit_matrix()
    refreshed_gram = refreshed_phi.transpose(-1, -2) @ refreshed_phi
    check_close("cleared D2 cache Gram diagonal", refreshed_diagonal, torch.diagonal(refreshed_gram))
    check_close("cleared D2 cache row energy", refreshed_row_energy, torch.sum(refreshed_gram.square(), dim=1))
    if torch.equal(refreshed_row_energy, old_row_energy):
        raise AssertionError("D2 geometry cache clear did not expose a changed frozen encoder")

    complex_encoder = build_encoder(
        spec, product_all_pairs_component_specs(spec, 2, False, "random_phase_diagonal"),
        dtype=torch.complex128, generator=gen)
    complex_decoder = UnrolledEffectiveChannelPGD(num_layers=1, geometry_chunk_size=3).to(dtype=torch.float64)
    diagonal, row_energy = complex_decoder._gram_geometry(complex_encoder)
    Phi = complex_encoder.explicit_matrix()
    G = Phi.conj().transpose(-1, -2) @ Phi
    check_close("complex D2 Gram diagonal", diagonal, torch.diagonal(G).real)
    check_close("complex D2 real-Gram row energy", row_energy, torch.sum(G.real.square(), dim=1))
    total, physical, interference = complex_decoder._effective_variance(
        variance, diagonal, row_energy, noise, complex_observation=True)
    expected_physical = 0.5 * noise.unsqueeze(1) * diagonal.unsqueeze(0)
    expected_interference = variance * (torch.sum(G.real.square(), dim=1) - torch.diagonal(G).real.square())
    check_close("complex D2 physical variance", physical, expected_physical)
    check_close("complex D2 uniform-variance Gram interference", interference, expected_interference)
    check_close("complex D2 total effective variance", total, expected_physical + expected_interference)

    noise_batch = sample_batch(
        complex_encoder, 4096,
        uniform_counts_generator(1, spec.num_codewords, torch.Generator().manual_seed(36)),
        constant_fading(1, torch.complex128), 3.0, torch.Generator().manual_seed(37))
    empirical_noise = torch.mean(torch.abs(noise_batch.Y - noise_batch.Y_clean) ** 2)
    if abs(float(empirical_noise / noise_batch.noise_var) - 1.0) > 0.04:
        raise AssertionError("complex AWGN samples do not match the declared total complex variance")

    learned_encoder = build_encoder(
        spec, product_all_pairs_component_specs(spec, 2, True), dtype=torch.float64,
        generator=torch.Generator().manual_seed(33))
    distinct_gen = torch.Generator().manual_seed(34)

    def distinct_sampler(batch_size: int):
        active = torch.stack([torch.randperm(spec.num_codewords, generator=distinct_gen)[:3] for _ in range(batch_size)])
        counts = torch.zeros(batch_size, spec.num_codewords, dtype=learned_encoder.dtype, device=learned_encoder.device)
        counts.scatter_(1, active.to(learned_encoder.device), 1.0)
        return counts, active

    batch = sample_batch(learned_encoder, 2, distinct_sampler, constant_fading(1, torch.float64), 3.0,
                         torch.Generator().manual_seed(35))
    learned_decoder = UnrolledEffectiveChannelPGD(num_layers=2, geometry_chunk_size=3).to(torch.float64)
    output = learned_decoder(learned_encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
    loss, _ = effective_channel_loss(output, batch.counts)
    loss.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in learned_encoder.parameters() if parameter.requires_grad):
        raise AssertionError("chunked D2 geometry did not pass finite gradients to a learned encoder")


def exact_ml_check() -> None:
    gen = torch.Generator().manual_seed(29)
    spec = URASpec(n=8, num_codewords=6, num_active=2, num_antennas=1, payload_bits=3)
    encoder = build_encoder(spec, product_all_pairs_component_specs(spec, 2, False), dtype=torch.float64, generator=gen)
    counts = torch.zeros(2, spec.num_codewords, dtype=encoder.dtype)
    counts[0, 1] = 2.0
    counts[1, 2] = 1.0; counts[1, 5] = 1.0
    H = torch.ones(2, 1, dtype=encoder.dtype)
    Y = encoder.matvec(counts).unsqueeze(-1)
    out = exact_count_ml(encoder, Y, H, torch.tensor([2, 2]))
    check_close("noiseless exact count ML", out.counts, counts)


def main() -> None:
    operator_checks(torch.float64, "random_sign_diagonal")
    operator_checks(torch.complex128, "random_phase_diagonal")
    mapped_preset_checks()
    decoder_checks()
    d2_algebra_checks()
    exact_ml_check()
    print("implicit factor, adjoint, variable-K, single-antenna, collision-loss, and exact-ML checks passed")


if __name__ == "__main__":
    main()
