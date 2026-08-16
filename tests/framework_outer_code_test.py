"""Certify procedural outer codes without constructing a global message table."""

from __future__ import annotations

import torch

from framework.core import ComponentSpec, URASpec
from framework.encoder import build_encoder
from framework.outer_code import (IdentityOuterCode, SparseLinearOuterCode, gf_multiply,
                                  random_sparse_outer_code, triadic_outer_code)
from framework.sectioned import sectioned_from_explicit


def check_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name}:\nactual={actual}\nexpected={expected}")


def all_payload_bits(payload_bits: int) -> torch.Tensor:
    messages = torch.arange(1 << payload_bits, dtype=torch.long)
    shifts = torch.arange(payload_bits - 1, -1, -1, dtype=torch.long)
    return ((messages.unsqueeze(1) >> shifts) & 1)


def global_counts(active: torch.Tensor, num_messages: int, dtype: torch.dtype) -> torch.Tensor:
    counts = torch.zeros(active.shape[0], num_messages, dtype=dtype)
    counts.scatter_add_(1, active, torch.ones_like(active, dtype=dtype))
    return counts


def test_identity_code() -> None:
    code = IdentityOuterCode(payload_bits=7, section_bits=3)
    if code.section_bits != (3, 3, 1) or code.section_sizes != (8, 8, 2) or code.factor_graph is not None:
        raise AssertionError("identity outer-code structure is incorrect")
    bits = all_payload_bits(7)
    paths = code.enumerate_paths()
    if torch.unique(paths, dim=0).shape[0] != 128 or not bool(code.is_valid(paths).all()):
        raise AssertionError("identity code is not an injective all-tuples mapping")
    check_equal("identity round trip", code.decode_bits(paths), bits)
    if bool(code.is_valid(torch.tensor([[0, 0, 2]])).item()):
        raise AssertionError("identity code accepted an out-of-range local symbol")


def test_finite_field_and_generic_sparse_code() -> None:
    values = torch.arange(16)
    for multiplier in range(1, 16):
        if torch.unique(gf_multiply(values, multiplier, 4)).numel() != 16:
            raise AssertionError(f"nonzero GF(16) multiplier {multiplier} is not invertible")
    a = torch.tensor([1, 3, 7, 12]); b = torch.tensor([2, 5, 9, 4]); c = torch.tensor([6, 11, 3, 8])
    check_equal("GF distributivity", gf_multiply(a, b ^ c, 4), gf_multiply(a, b, 4) ^ gf_multiply(a, c, 4))

    code = SparseLinearOuterCode(payload_bits=8, section_bits=2,
                                 parity_supports=[(0, 1), (1, 2, 3), (0, 3)],
                                 parity_coefficients=[(1, 2), (3, 1, 2), (2, 3)])
    bits = all_payload_bits(8)
    paths = code.enumerate_paths()
    if paths.shape != (256, 7) or torch.unique(paths, dim=0).shape[0] != 256:
        raise AssertionError("generic sparse code lost an information message")
    if not bool(code.is_valid(paths).all()) or not bool((code.factor_graph.syndrome(paths) == 0).all()):
        raise AssertionError("generic sparse encoder does not satisfy Hx=0")
    check_equal("generic sparse round trip", code.decode_bits(paths), bits)
    corrupted = paths[:4].clone(); corrupted[:, code.parity_positions[0]] ^= 1
    if bool(code.is_valid(corrupted).any()):
        raise AssertionError("generic sparse code accepted a corrupted parity symbol")
    if code.factor_graph.parity_check_matrix().shape != (3, 7):
        raise AssertionError("unexpected sparse parity-check matrix shape")


def test_triadic_structure() -> None:
    code = triadic_outer_code(payload_bits=8, section_bits=2)
    expected_checks = ((0, 1, 2), (2, 3, 4), (4, 5, 6), (0, 6, 7))
    actual_checks = tuple(check.variables for check in code.factor_graph.checks)
    if actual_checks != expected_checks or code.info_positions != (0, 2, 4, 6):
        raise AssertionError(f"unexpected triadic graph {actual_checks}")
    bits = torch.randint(2, (3, 5, 8), generator=torch.Generator().manual_seed(31))
    paths = code.encode_bits(bits)
    if paths.shape != (3, 5, 8) or not bool(code.is_valid(paths).all()):
        raise AssertionError("triadic code did not produce valid batched paths")
    check_equal("triadic round trip", code.decode_bits(paths), bits)


def test_explicit_framework_equivalence() -> None:
    gen = torch.Generator().manual_seed(37)
    code = triadic_outer_code(payload_bits=8, section_bits=2)
    all_bits = all_payload_bits(8)
    all_paths = code.encode_bits(all_bits)
    spec = URASpec(n=20, num_codewords=256, num_active=3, payload_bits=8)
    components = [ComponentSpec(Q=1, d=20, V=4, N=4, R_init="identity", C_init="random_gaussian",
                                U_init="all_pairs", T_init="explicit", learn_R=False, learn_C=True,
                                explicit_msg_to_atom=all_paths[:, ell]) for ell in range(code.num_sections)]
    explicit = build_encoder(spec, components, dtype=torch.float64, generator=gen)
    sectioned = sectioned_from_explicit(explicit)
    active = torch.tensor([[0, 17, 201], [44, 44, 255]])
    active_bits = all_bits[active]
    section_signal, section_counts, active_paths = sectioned.encode_bits(active_bits, code)
    check_equal("procedural active paths", active_paths, all_paths[active])
    expected_signal = explicit.matvec(global_counts(active, 256, torch.float64))
    if not torch.allclose(section_signal, expected_signal, atol=1e-12, rtol=0.0):
        raise AssertionError("procedural outer code and explicit global table produce different signals")
    if any(not torch.equal(local.sum(dim=1), torch.full((2,), 3.0, dtype=torch.float64)) for local in section_counts):
        raise AssertionError("valid paths did not contribute one count to every section")
    section_signal.square().mean().backward()
    if not all(bank.C.grad is not None and torch.isfinite(bank.C.grad).all() for bank in sectioned.banks):
        raise AssertionError("procedural encoding did not preserve gradients to physical atom banks")


def test_b100_procedural_scaling() -> None:
    gen = torch.Generator().manual_seed(41)
    code = random_sparse_outer_code(payload_bits=100, section_bits=10, num_parity_sections=4,
                                    check_degree=3, generator=gen)
    bits = torch.randint(2, (4, 7, 100), generator=gen)
    paths = code.encode_bits(bits)
    if paths.shape != (4, 7, 14) or code.section_sizes != (1024,) * 14:
        raise AssertionError("unexpected B=100 procedural-code dimensions")
    if not bool(code.is_valid(paths).all()):
        raise AssertionError("B=100 procedural code generated invalid paths")
    check_equal("B=100 round trip", code.decode_bits(paths), bits)
    try:
        code.enumerate_paths()
    except ValueError:
        pass
    else:
        raise AssertionError("B=100 code permitted accidental global message enumeration")


def main() -> None:
    test_identity_code()
    test_finite_field_and_generic_sparse_code()
    test_triadic_structure()
    test_explicit_framework_equivalence()
    test_b100_procedural_scaling()
    print("procedural outer codes: identity, generic sparse, triadic, explicit equivalence, and B=100 passed")


if __name__ == "__main__":
    main()
