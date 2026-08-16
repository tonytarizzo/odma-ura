"""Certify exact and sampled-power contracts for procedural sectioned codewords."""

from __future__ import annotations

import torch

from framework.core import ComponentSpec, SectionedURASpec
from framework.losses import sectioned_power_penalty
from framework.encoder import SubsampledHadamardAtomBank
from framework.outer_code import triadic_outer_code
from framework.sectioned import (build_default_scalable_setup, build_orthogonal_sectioned_encoder, build_sectioned_encoder,
                                 sampled_energy_report)


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-10) -> None:
    error = float(torch.max(torch.abs(actual - expected)).detach()) if actual.numel() else 0.0
    if error > atol:
        raise AssertionError(f"{name}: max error {error:.3e} exceeds {atol:.3e}")


def test_exact_orthogonal_energy() -> None:
    gen = torch.Generator().manual_seed(51)
    code = triadic_outer_code(payload_bits=4, section_bits=1)
    spec = SectionedURASpec(n=32, payload_bits=4, num_active=3)
    encoder = build_orthogonal_sectioned_encoder(spec, code, dtype=torch.float64, generator=gen)
    paths = code.enumerate_paths()
    check_close("all complete-path energies", encoder.path_energies(paths), torch.ones(16, dtype=torch.float64))
    certificate = encoder.certify_exact_energy(tolerance=1e-10)
    if not certificate["guaranteed"] or certificate["mode"] != "orthogonal_exact":
        raise AssertionError(f"exact-energy certificate failed: {certificate}")
    report = sampled_energy_report(encoder, code, num_samples=40, generator=gen)
    if report["max_abs_unit_deviation"] > 1e-10:
        raise AssertionError(f"sampled exact-energy report failed: {report}")

    counts = tuple(torch.randn(3, size, dtype=torch.float64, generator=gen) for size in encoder.section_sizes)
    residual = torch.randn(3, spec.n, dtype=torch.float64, generator=gen)
    lhs = torch.sum(encoder.synthesize(counts) * residual)
    rhs = sum(torch.sum(local * adjoint) for local, adjoint in zip(counts, encoder.local_adjoint(residual)))
    check_close("orthogonal forward/adjoint", lhs, rhs)

    with torch.no_grad():
        encoder.banks[0].C.mul_(1.7)
    if encoder.certify_exact_energy()["guaranteed"]:
        raise AssertionError("certificate ignored a violated local-column constraint")
    encoder.apply_constraints()
    if not encoder.certify_exact_energy(tolerance=1e-10)["guaranteed"]:
        raise AssertionError("post-update projection did not restore the exact-energy contract")
    check_close("new-message energies after projection", encoder.path_energies(paths), torch.ones(16, dtype=torch.float64))


def test_overlapping_sampled_penalty() -> None:
    gen = torch.Generator().manual_seed(53)
    code = triadic_outer_code(payload_bits=4, section_bits=1)
    spec = SectionedURASpec(n=12, payload_bits=4, num_active=2)
    components = [ComponentSpec(Q=1, d=12, V=2, N=2, R_init="identity", C_init="random_gaussian",
                                U_init="all_pairs", learn_C=True) for _ in range(code.num_sections)]
    encoder = build_sectioned_encoder(spec, components, dtype=torch.float64, generator=gen)
    paths = code.encode_bits(torch.randint(2, (24, 4), generator=gen))
    penalty = sectioned_power_penalty(encoder, paths)
    penalty.backward()
    if not torch.isfinite(penalty) or not all(bank.C.grad is not None and torch.isfinite(bank.C.grad).all()
                                              for bank in encoder.banks):
        raise AssertionError("sampled power penalty did not provide finite gradients to every overlapping bank")
    if encoder.certify_exact_energy()["guaranteed"]:
        raise AssertionError("overlapping sampled-power mode incorrectly claimed a structural guarantee")


def test_implicit_hadamard_bank_and_b128_energy() -> None:
    gen = torch.Generator().manual_seed(55)
    bank = SubsampledHadamardAtomBank(num_atoms=16, output_dimension=7, dtype=torch.float64, generator=gen)
    counts = torch.randn(3, 16, dtype=torch.float64, generator=gen)
    residual = torch.randn(3, 7, dtype=torch.float64, generator=gen)
    matrix = bank.explicit_local_matrix()
    check_close("implicit Hadamard forward", bank.local_matvec(counts), counts @ matrix.transpose(0, 1))
    check_close("implicit Hadamard adjoint", bank.local_rmatvec(residual), residual @ matrix)
    check_close("implicit Hadamard column energy", torch.sum(matrix.square(), dim=0), torch.ones(16, dtype=torch.float64))

    encoder, code = build_default_scalable_setup(num_active=2, n=256, mixing_stages=4, generator=gen)
    bits = torch.randint(2, (1, 2, 128), generator=gen)
    signal, _, paths = encoder.encode_bits(bits, code)
    if encoder.state_size != 16 * (1 << 16) or signal.shape != (1, 256):
        raise AssertionError("B=128 implicit physical encoder has unexpected dimensions")
    if any(hasattr(local, "C") for local in encoder.banks):
        raise AssertionError("B=128 implicit encoder stored dense local codebooks")
    if not encoder.certify_exact_energy()["guaranteed"]:
        raise AssertionError("B=128 implicit encoder failed its structural energy certificate")
    check_close("B=128 selected codeword energy", encoder.path_energies(paths), torch.ones(1, 2), atol=2e-6)


def main() -> None:
    test_exact_orthogonal_energy()
    test_overlapping_sampled_penalty()
    test_implicit_hadamard_bank_and_b128_energy()
    print("sectioned energy: exact orthogonal guarantee, projection, adjoint, and sampled penalty passed")


if __name__ == "__main__":
    main()
