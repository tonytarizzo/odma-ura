"""Small exact checks for implicit factor operators and learned decoders."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.channel import constant_fading, sample_batch, uniform_count_range_generator  # noqa: E402
from framework.core import URASpec  # noqa: E402
from framework.decoders import exact_count_ml  # noqa: E402
from framework.encoder import build_encoder, matvec_with_matrix, rmatvec_with_matrix  # noqa: E402
from framework.learned_decoders import FactorAttentionISTANet, UnrolledBernoulliPGD  # noqa: E402
from framework.losses import support_count_loss  # noqa: E402
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
    for model in [UnrolledBernoulliPGD(num_layers=2, power_iters=3),
                  FactorAttentionISTANet(num_layers=2, hidden_dim=8, pattern_slots=1, global_slots=1, power_iters=3)]:
        out = model(encoder, batch.Y, batch.H, batch.num_active, noise_var=batch.noise_var)
        if not torch.all(out.counts.sum(dim=1) == batch.num_active):
            raise AssertionError("hard decoder output must preserve the supplied per-sample K")
        collision_target = batch.counts.clone()
        collision_target[0].zero_(); collision_target[0, 0] = 2.0
        loss, parts = support_count_loss(out, collision_target, lambda_count=0.1, lambda_symmetry=0.01)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in parts.values()):
            raise AssertionError("collision-aware decoder loss must remain finite")
        if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError("decoder loss did not produce finite parameter gradients")


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
    exact_ml_check()
    print("implicit factor, adjoint, variable-K, single-antenna, collision-loss, and exact-ML checks passed")


if __name__ == "__main__":
    main()
