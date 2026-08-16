"""Certify differentiable modular BP, path losses, beam extraction, and collision routes."""

from __future__ import annotations

import torch

from framework.core import ComponentSpec, OuterBPOutput, SectionedURASpec
from framework.outer_code import SparseLinearOuterCode, ccs_amp_paper_outer_code, triadic_outer_code
from framework.outer_decoder import (DifferentiableOuterBP, SectionedOuterDecoder, ValidPathListDecoder, outer_marginal_loss,
                                     outer_path_contrastive_loss, path_list_pupe, sectioned_outer_training_loss)
from framework.sectioned import build_sectioned_encoder


def section_counts(paths: torch.Tensor, section_size: int) -> tuple[torch.Tensor, ...]:
    output = []
    for ell in range(paths.shape[2]):
        local = torch.zeros(paths.shape[0], section_size, dtype=torch.float64)
        local.scatter_add_(1, paths[:, :, ell], torch.ones_like(paths[:, :, ell], dtype=torch.float64))
        output.append(local)
    return tuple(output)


def test_tree_bp_matches_exhaustive_marginals() -> None:
    gen = torch.Generator().manual_seed(61)
    code = SparseLinearOuterCode(payload_bits=6, section_bits=2, parity_supports=[(0, 1), (1, 2)])
    logits = tuple(torch.randn(2, 4, dtype=torch.float64, generator=gen) for _ in range(code.num_sections))
    bp = DifferentiableOuterBP(num_iterations=4, init_damping=0.0).to(dtype=torch.float64)
    with torch.no_grad():
        bp.raw_damping.fill_(-30.0)
    output = bp(code.factor_graph, logits)

    paths = code.enumerate_paths()
    stacked = torch.stack(logits, dim=1)
    exact = torch.zeros_like(output.log_beliefs)
    for b in range(stacked.shape[0]):
        scores = sum(stacked[b, ell, paths[:, ell]] for ell in range(code.num_sections))
        weights = torch.softmax(scores, dim=0)
        for ell in range(code.num_sections):
            exact[b, ell].scatter_add_(0, paths[:, ell], weights)
    error = float(torch.max(torch.abs(output.log_beliefs.exp() - exact)).detach())
    if error > 2e-10:
        raise AssertionError(f"tree BP/exhaustive marginal error {error:.3e}")


def test_differentiable_multiuser_losses() -> None:
    gen = torch.Generator().manual_seed(63)
    code = triadic_outer_code(payload_bits=8, section_bits=2)
    bits = torch.randint(2, (3, 4, 8), generator=gen)
    paths = code.encode_bits(bits)
    counts = section_counts(paths, 4)
    logits = tuple(torch.randn(3, 4, dtype=torch.float64, generator=gen, requires_grad=True)
                   for _ in range(code.num_sections))
    bp = DifferentiableOuterBP(num_iterations=2).to(dtype=torch.float64)
    output = bp(code.factor_graph, logits)
    marginal = outer_marginal_loss(output, counts)
    contrastive = outer_path_contrastive_loss(output, paths, code, num_negatives=8, generator=gen)
    loss = marginal + 0.2 * contrastive
    loss.backward()
    if not torch.isfinite(loss) or not all(value.grad is not None and torch.isfinite(value.grad).all() for value in logits):
        raise AssertionError("outer losses did not produce finite gradients to every section")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in bp.parameters()):
        raise AssertionError("outer losses did not reach BP calibration parameters")


def test_end_to_end_d0_bp_gradients() -> None:
    gen = torch.Generator().manual_seed(65)
    code = triadic_outer_code(payload_bits=4, section_bits=1)
    spec = SectionedURASpec(n=16, payload_bits=4, num_active=2)
    components = [ComponentSpec(Q=1, d=16, V=2, N=2, R_init="identity", C_init="random_gaussian",
                                U_init="all_pairs", learn_C=True) for _ in range(code.num_sections)]
    encoder = build_sectioned_encoder(spec, components, dtype=torch.float64, generator=gen)
    bits = torch.randint(2, (3, 2, 4), generator=gen)
    y, counts, paths = encoder.encode_bits(bits, code)
    H = torch.ones(3, 1, dtype=torch.float64)
    decoder = SectionedOuterDecoder(d0_layers=2, bp_iterations=2, power_iters=3).to(dtype=torch.float64)
    output = decoder(encoder, code, y.unsqueeze(-1), H, num_active=2, noise_var=1e-4)
    loss, parts = sectioned_outer_training_loss(output, counts, paths, code, encoder=encoder,
                                                lambda_path=0.1, lambda_power=0.05,
                                                num_path_negatives=4, generator=gen)
    loss.backward()
    if not all(bank.C.grad is not None and torch.isfinite(bank.C.grad).all() for bank in encoder.banks):
        raise AssertionError("end-to-end D0+BP loss did not reach every physical atom bank")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in decoder.parameters()):
        raise AssertionError("end-to-end D0+BP loss did not reach every decoder parameter")
    if not all(torch.isfinite(value) for value in parts.values()):
        raise AssertionError("combined sectioned outer training loss has a nonfinite component")


def peaked_output(paths: torch.Tensor, section_size: int) -> OuterBPOutput:
    counts = section_counts(paths, section_size)
    probabilities = torch.stack(tuple((local + 0.02) / (local + 0.02).sum(dim=1, keepdim=True) for local in counts), dim=1)
    return OuterBPOutput(log_beliefs=torch.log(probabilities))


def test_beam_and_collision_routes() -> None:
    small = triadic_outer_code(payload_bits=4, section_bits=1)
    small_bits = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]])
    small_paths = small.encode_bits(small_bits)
    small_counts = section_counts(small_paths, 2)
    decoder = ValidPathListDecoder(beam_width=16, list_extra=5, candidate_cap=None, multiplicity_iterations=200)
    recovered = decoder.decode(peaked_output(small_paths, 2), small, small_counts, num_active=3)
    if recovered.meta["collision_mode"] != "complete_path_multiplicity":
        raise AssertionError("small-B route did not retain complete-path multiplicity")
    if float(path_list_pupe(recovered, small_paths).max()) > 0.0:
        raise AssertionError(f"small-B multiplicity path recovery failed: {recovered.paths}, {recovered.counts}")

    large = triadic_outer_code(payload_bits=24, section_bits=4)
    large_bits = torch.tensor([[[0] * 24, [1] * 24]])
    large_paths = large.encode_bits(large_bits)
    large_counts = section_counts(large_paths, 16)
    recovered = ValidPathListDecoder(beam_width=64, list_extra=8, candidate_cap=None).decode(
        peaked_output(large_paths, 16), large, large_counts, num_active=2)
    if recovered.meta["collision_mode"] != "unique_complete_paths" or not torch.equal(recovered.counts.sum(dim=1), torch.tensor([2.0])):
        raise AssertionError("large-B route did not use unique complete paths")
    if float(path_list_pupe(recovered, large_paths).max()) > 0.0:
        raise AssertionError("large-B valid-path recovery failed")


def test_paper_scale_outer_dimensions() -> None:
    code = ccs_amp_paper_outer_code()
    if code.payload_bits != 128 or code.uniform_section_bits != 16 or code.num_sections != 16:
        raise AssertionError("paper-scale default is not B=128,J=16,L=16")
    bits = torch.randint(2, (2, 3, 128), generator=torch.Generator().manual_seed(67))
    paths = code.encode_bits(bits)
    if paths.shape != (2, 3, 16) or not bool(code.is_valid(paths).all()):
        raise AssertionError("paper-scale procedural paths are invalid")
    if not torch.equal(code.decode_bits(paths), bits):
        raise AssertionError("paper-scale procedural outer code did not round trip")
    logits = tuple(torch.zeros(1, 1 << 16) for _ in range(code.num_sections))
    with torch.no_grad():
        beliefs = DifferentiableOuterBP(num_iterations=1)(code.factor_graph, logits).log_beliefs
    if beliefs.shape != (1, 16, 1 << 16) or not bool(torch.isfinite(beliefs).all()):
        raise AssertionError("paper-scale full-alphabet BP did not execute")
    selected = paths[:1, :2]
    selected_counts = section_counts(selected, 1 << 16)
    recovered = ValidPathListDecoder(beam_width=64, list_extra=4, candidate_cap=32).decode(
        peaked_output(selected, 1 << 16), code, selected_counts, num_active=2)
    if float(path_list_pupe(recovered, selected).max()) > 0.0:
        raise AssertionError("paper-scale evaluation beam did not retain the supplied valid paths")


def main() -> None:
    test_tree_bp_matches_exhaustive_marginals()
    test_differentiable_multiuser_losses()
    test_end_to_end_d0_bp_gradients()
    test_beam_and_collision_routes()
    test_paper_scale_outer_dimensions()
    print("outer decoder: exhaustive BP, gradients, beam/multiplicity routes, and B=128,J=16 passed")


if __name__ == "__main__":
    main()
