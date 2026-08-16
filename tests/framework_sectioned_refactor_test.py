"""Certify the no-M section backend against the explicit global backend.

The checks deliberately separate three claims:
  1. the LocalAtomBank refactor preserves legacy global matvec/rmatvec behavior;
  2. L>1 section counts reproduce the same transmitted signals and local/global evidence;
  3. for L=1, section-domain D0 is numerically identical to global D0, including its loss gradients.

A final B=100 construction check proves that the scalable path allocates only
``sum_l N_l`` state and never constructs ``M = 2^100``.
"""

from __future__ import annotations

import torch

from framework.channel import constant_fading
from framework.core import ComponentSpec, SectionedURASpec, URASpec
from framework.encoder import build_encoder
from framework.learned_decoders import UnrolledBernoulliPGD, UnrolledSectionedCountPGD
from framework.losses import section_support_count_loss, support_count_loss
from framework.pipeline import ccs_component_specs, product_all_pairs_component_specs
from framework.sectioned import (build_sectioned_encoder, sample_sectioned_batch,
                                 sectioned_from_explicit, uniform_section_paths_generator)


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-10) -> None:
    error = float(torch.max(torch.abs(actual - expected)).detach()) if actual.numel() else 0.0
    if error > atol:
        raise AssertionError(f"{name}: max error {error:.3e} exceeds {atol:.3e}")


def global_counts(active: torch.Tensor, M: int, dtype: torch.dtype) -> torch.Tensor:
    counts = torch.zeros(active.shape[0], M, dtype=dtype)
    counts.scatter_add_(1, active, torch.ones_like(active, dtype=dtype))
    return counts


def paths_from_explicit(encoder, active: torch.Tensor) -> torch.Tensor:
    return torch.stack([component.msg_to_atom[active] for component in encoder.components], dim=-1)


def test_multisection_equivalence() -> None:
    gen = torch.Generator().manual_seed(17)
    spec = URASpec(n=48, num_codewords=256, num_active=5, num_antennas=1, payload_bits=8)
    explicit = build_encoder(spec, ccs_component_specs(spec, num_sections=4, learn_C=False),
                             dtype=torch.float64, generator=gen)
    sectioned = sectioned_from_explicit(explicit)
    active = torch.randint(spec.num_codewords, (7, spec.num_active), generator=gen)
    counts = global_counts(active, spec.num_codewords, torch.float64)
    paths = paths_from_explicit(explicit, active)
    section_signal, section_counts = sectioned.encode_paths(paths)
    check_close("L>1 transmitted signal", section_signal, explicit.matvec(counts))

    for ell, (component, local) in enumerate(zip(explicit.components, section_counts)):
        expected = component._messages_to_atoms(counts)
        check_close(f"section {ell} counts", local, expected)

    residual = torch.randn(7, spec.n, dtype=torch.float64, generator=gen)
    local_evidence = sectioned.local_adjoint(residual)
    reconstructed_global = sum(component._atoms_to_messages(local)
                               for component, local in zip(explicit.components, local_evidence))
    check_close("L>1 adjoint reconstruction", reconstructed_global, explicit.rmatvec(residual))

    all_messages = torch.arange(spec.num_codewords)
    all_paths = paths_from_explicit(explicit, all_messages.unsqueeze(0)).squeeze(0)
    section_columns = sum(bank.atom_columns(all_paths[:, ell]) for ell, bank in enumerate(sectioned.banks))
    check_close("L>1 explicit columns", section_columns, explicit.explicit_matrix())


def test_l1_decoder_and_gradient_equivalence() -> None:
    gen = torch.Generator().manual_seed(23)
    # K_a=1 makes the Binomial local-count prior exactly Bernoulli, certifying
    # that the general count decoder contains the original D0 as a special case.
    spec = URASpec(n=24, num_codewords=16, num_active=1, num_antennas=1, payload_bits=4)
    explicit = build_encoder(spec, product_all_pairs_component_specs(spec, 4, False),
                             dtype=torch.float64, generator=gen)
    sectioned = sectioned_from_explicit(explicit)
    active = torch.randint(spec.num_codewords, (6, spec.num_active), generator=gen)
    counts = global_counts(active, spec.num_codewords, torch.float64)
    paths = paths_from_explicit(explicit, active)
    y = explicit.matvec(counts)
    Y = (y + 0.03 * torch.randn(y.shape, dtype=y.dtype, generator=gen)).unsqueeze(-1)
    H = torch.ones(Y.shape[0], 1, dtype=Y.dtype)

    global_decoder = UnrolledBernoulliPGD(num_layers=4, power_iters=8).to(dtype=torch.float64)
    section_decoder = UnrolledSectionedCountPGD(num_layers=4, power_iters=8).to(dtype=torch.float64)
    section_decoder.load_state_dict(global_decoder.state_dict())
    exact_lipschitz = torch.linalg.matrix_norm(explicit.explicit_matrix(), ord=2) ** 2
    explicit._spectral_cache[8] = exact_lipschitz
    sectioned._spectral_cache[8] = exact_lipschitz

    global_output = global_decoder(explicit, Y, H, spec.num_active, noise_var=0.03 ** 2)
    section_output = section_decoder(sectioned, Y, H, spec.num_active, noise_var=0.03 ** 2)
    check_close("L=1 soft decoder state", section_output.meta["soft_section_counts"][0],
                global_output.meta["soft_counts"])
    check_close("L=1 hard decoder state", section_output.section_counts[0], global_output.counts)
    for layer, (global_logits, local_logits) in enumerate(zip(global_output.meta["layer_logits"],
                                                               section_output.meta["section_layer_logits"])):
        check_close(f"L=1 layer {layer} logits", local_logits[0], global_logits)

    global_loss, global_parts = support_count_loss(global_output, counts, lambda_count=0.1, lambda_symmetry=0.0)
    section_loss, section_parts = section_support_count_loss(section_output, (counts,), lambda_count=0.1)
    check_close("L=1 total loss", section_loss, global_loss)
    check_close("L=1 support loss", section_parts["support"], global_parts["support"])
    check_close("L=1 count loss", section_parts["count"], global_parts["count"])
    global_loss.backward(); section_loss.backward()
    for name, parameter in global_decoder.named_parameters():
        local_parameter = dict(section_decoder.named_parameters())[name]
        check_close(f"L=1 gradient {name}", local_parameter.grad, parameter.grad, atol=2e-9)


def test_b100_has_no_global_axis() -> None:
    gen = torch.Generator().manual_seed(29)
    spec = SectionedURASpec(n=64, payload_bits=100, num_active=5, num_antennas=1)
    components = [ComponentSpec(Q=1, d=64, V=1024, N=1024, R_init="identity", C_init="random_gaussian",
                                U_init="all_pairs", learn_R=False, learn_C=False) for _ in range(10)]
    encoder = build_sectioned_encoder(spec, components, dtype=torch.float32, generator=gen)
    if encoder.section_sizes != (1024,) * 10 or encoder.state_size != 10_240:
        raise AssertionError(f"unexpected B=100 local state {encoder.section_sizes}")
    if any(hasattr(bank, "msg_to_atom") for bank in encoder.banks) or hasattr(encoder, "num_codewords"):
        raise AssertionError("the section backend leaked a global message axis")
    sampler = uniform_section_paths_generator(spec.num_active, encoder.section_sizes, gen, encoder.device)
    batch = sample_sectioned_batch(encoder, 3, sampler, constant_fading(1, encoder.dtype, encoder.device), 3.0, gen)
    if batch.active_paths.shape != (3, 5, 10):
        raise AssertionError(f"unexpected path shape {tuple(batch.active_paths.shape)}")
    if sum(x.shape[1] for x in batch.section_counts) != 10_240:
        raise AssertionError("B=100 batch state is not section-local")
    decoder = UnrolledSectionedCountPGD(num_layers=1, power_iters=2)
    output = decoder(encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
    if tuple(x.shape for x in output.section_counts) != ((3, 1024),) * 10:
        raise AssertionError("B=100 decoder created an unexpected state shape")
    if output.meta["section_denoiser"] != "binomial_count":
        raise AssertionError(f"unexpected section denoiser {output.meta['section_denoiser']}")


def test_multisection_learning_gradients() -> None:
    gen = torch.Generator().manual_seed(31)
    spec = SectionedURASpec(n=18, payload_bits=6, num_active=3, num_antennas=1)
    components = [ComponentSpec(Q=1, d=18, V=8, N=8, R_init="identity", C_init="random_gaussian",
                                U_init="all_pairs", learn_R=False, learn_C=True) for _ in range(3)]
    encoder = build_sectioned_encoder(spec, components, dtype=torch.float64, generator=gen)
    sampler = uniform_section_paths_generator(spec.num_active, encoder.section_sizes, gen, encoder.device)
    batch = sample_sectioned_batch(encoder, 4, sampler, constant_fading(1, encoder.dtype, encoder.device), 4.0, gen)
    decoder = UnrolledSectionedCountPGD(num_layers=2, power_iters=3).to(dtype=torch.float64)
    output = decoder(encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
    if output.meta["section_denoiser"] != "binomial_count":
        raise AssertionError(f"section decoder did not use count-aware priors: {output.meta['section_denoiser']}")
    loss, parts = section_support_count_loss(output, batch.section_counts)
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in parts.values()):
        raise AssertionError("section-domain training loss is not finite")
    if not all(bank.C.grad is not None and torch.isfinite(bank.C.grad).all() for bank in encoder.banks):
        raise AssertionError("section-domain loss did not reach every learnable local codebook")
    if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in decoder.parameters()):
        raise AssertionError("section-domain loss did not reach the decoder")


def test_binomial_count_posterior() -> None:
    u = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    proposal, logits = UnrolledSectionedCountPGD._binomial_count_proposal(
        u, torch.tensor([3]), 4, torch.tensor([1e-3], dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64))
    check_close("binomial posterior mean", proposal, u, atol=1e-8)
    if not torch.isfinite(logits).all() or not (logits[0, 0] < 0.0 < logits[0, 1]):
        raise AssertionError(f"unexpected binomial support logits {logits}")


def main() -> None:
    test_multisection_equivalence()
    test_l1_decoder_and_gradient_equivalence()
    test_multisection_learning_gradients()
    test_binomial_count_posterior()
    test_b100_has_no_global_axis()
    print("section-domain refactor: L>1 algebra/training, L=1 D0 equivalence, and B=100 no-M execution passed")


if __name__ == "__main__":
    main()
