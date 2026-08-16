"""Small-B causal bridge between the explicit and scalable section-domain paths.

``l1`` keeps the original physical codebook and proves that the section backend
with the Bernoulli-compatibility prior is the old global D0 computation at the
actual multi-user loads. It separately trains the preferred Binomial local
prior so that a prior change is not hidden inside the backend comparison.

``lgt`` constructs an L>1 procedural encoder, freezes it, materialises its
induced global codebook only because B is small, and trains global D0 and the
scalable local D0/outer decoder on matched data streams. This separates encoder
geometry from local inference, association, and BP.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.channel import constant_fading, sample_batch, uniform_count_range_generator, uniform_counts_generator  # noqa: E402
from framework.core import ComponentSpec, OuterBPOutput, PathListOutput, SectionedURASpec, URASpec  # noqa: E402
from framework.encoder import build_encoder  # noqa: E402
from framework.learned_decoders import UnrolledBernoulliPGD, UnrolledSectionedCountPGD  # noqa: E402
from framework.losses import section_support_count_loss, support_count_loss  # noqa: E402
from framework.metrics import aggregate_metrics, batch_evaluate  # noqa: E402
from framework.outer_code import IdentityOuterCode, SparseLinearOuterCode, triadic_outer_code  # noqa: E402
from framework.outer_decoder import (SectionedOuterDecoder, ValidPathListDecoder, path_list_pupe,
                                     sectioned_outer_training_loss)  # noqa: E402
from framework.pipeline import dense_component_specs, odma_component_specs, sparse_global_component_specs  # noqa: E402
from framework.sectioned import (build_orthogonal_sectioned_encoder, outer_code_path_generator,
                                 sample_sectioned_batch, sampled_energy_report, sectioned_from_explicit)  # noqa: E402


def float_grid(text: str) -> list[float]: return [float(value) for value in text.split(",") if value.strip()]


def int_grid(text: str) -> list[int]: return [int(value) for value in text.split(",") if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", choices=["l1", "lgt"], required=True)
    p.add_argument("--encoder", choices=["dense_fixed", "sparse_global_fixed", "odma_fixed"], default="dense_fixed")
    p.add_argument("--outer-code", choices=["triadic", "identity"], default="triadic")
    p.add_argument("-B", "--payload-bits", type=int, default=12)
    p.add_argument("-J", "--section-bits", type=int, default=4)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--learn-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--k-min", type=int, default=9)
    p.add_argument("--k-max", type=int, default=26)
    p.add_argument("--eval-k", type=int_grid, default=int_grid("9,17,26,30"))
    p.add_argument("--steps", type=int, default=8000, help="training steps for each final decoder")
    p.add_argument("--encoder-steps", type=int, default=4000, help="local D0 encoder pretraining; ignored when fixed")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--train-ebn0-min", type=float, default=-4.0)
    p.add_argument("--train-ebn0-max", type=float, default=12.0)
    p.add_argument("--eval-ebn0", type=float_grid, default=float_grid("-4,0,4,8,12"))
    p.add_argument("--d0-layers", type=int, default=8)
    p.add_argument("--bp-iterations", type=int, default=4)
    p.add_argument("--power-iters", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-count", type=float, default=0.1)
    p.add_argument("--lambda-marginal", type=float, default=0.1)
    p.add_argument("--lambda-path", type=float, default=0.02)
    p.add_argument("--num-path-negatives", type=int, default=32)
    p.add_argument("--beam-width", type=int, default=512)
    p.add_argument("--list-extra", type=int, default=32)
    p.add_argument("--candidate-cap", type=int, default=None)
    p.add_argument("--mixing-stages", type=int, default=8)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=2501)
    p.add_argument("--out-dir", default="results/framework_sectioned_bridge")
    args = p.parse_args(argv)
    if args.payload_bits <= 0 or args.section_bits <= 0 or args.payload_bits % args.section_bits:
        p.error("B and J must be positive, with J dividing B")
    if args.k_min <= 0 or args.k_max < args.k_min or any(value <= 0 for value in args.eval_k):
        p.error("invalid training or evaluation load")
    if args.steps <= 0 or args.encoder_steps < 0 or args.batch_size <= 0 or args.eval_batches <= 0:
        p.error("steps and batch sizes must be positive")
    if args.train_ebn0_max < args.train_ebn0_min:
        p.error("training Eb/N0 range is reversed")
    return args


def random_ebn0(args: argparse.Namespace, generator: torch.Generator) -> float:
    return args.train_ebn0_min + float(torch.rand((), generator=generator)) * (args.train_ebn0_max - args.train_ebn0_min)


def random_k(args: argparse.Namespace, generator: torch.Generator) -> int:
    return int(torch.randint(args.k_min, args.k_max + 1, (), generator=generator))


def make_global_encoder(args: argparse.Namespace, generator: torch.Generator):
    M = 1 << args.payload_bits
    spec = URASpec(args.n, M, args.k_max, 1, args.payload_bits)
    if args.encoder == "dense_fixed":
        components = dense_component_specs(spec, False)
    elif args.encoder == "sparse_global_fixed":
        components = sparse_global_component_specs(spec, args.n // 4, generator)
    else:
        components = odma_component_specs(spec, args.n // 4, 4, False, False)
    return build_encoder(spec, components, dtype=torch.float32, generator=generator)


def bits_to_indices(bits: torch.Tensor) -> torch.Tensor:
    powers = 1 << torch.arange(bits.shape[-1] - 1, -1, -1, dtype=torch.long, device=bits.device)
    return torch.sum(bits.to(torch.long) * powers, dim=-1)


def global_counts_from_paths(paths: torch.Tensor, code, M: int, dtype: torch.dtype) -> torch.Tensor:
    indices = bits_to_indices(code.decode_bits(paths))
    counts = torch.zeros(paths.shape[0], M, dtype=dtype, device=paths.device)
    counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=dtype))
    return counts


def materialise_global_encoder(sectioned, code, num_active: int):
    paths = code.enumerate_paths(device=sectioned.device)
    Phi = sectioned.codeword_columns(paths).transpose(0, 1).detach()
    spec = URASpec(sectioned.n, Phi.shape[1], num_active, sectioned.spec.num_antennas,
                   sectioned.spec.payload_bits, sectioned.spec.energy_per_codeword)
    component = ComponentSpec(Q=1, d=sectioned.n, V=Phi.shape[1], N=Phi.shape[1], R_init="identity",
                              C_init="explicit", U_init="all_pairs", T_init="identity", explicit_C=Phi)
    return build_encoder(spec, [component], dtype=sectioned.dtype), Phi


def optimise(parameters, loss_fn, steps: int, args: argparse.Namespace, label: str, post_step=None) -> list[dict]:
    parameters = list(parameters)
    optimiser = torch.optim.Adam(parameters, lr=args.lr)
    progress = []
    sums: dict[str, float] = {}
    for step in range(1, steps + 1):
        loss, parts = loss_fn(step)
        optimiser.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimiser.step()
        if post_step is not None: post_step()
        for name, value in parts.items(): sums[name] = sums.get(name, 0.0) + float(value.detach())
        sums["grad_norm"] = sums.get("grad_norm", 0.0) + float(grad_norm)
        if step % args.log_every == 0 or step == steps:
            count = args.log_every if step % args.log_every == 0 else steps % args.log_every
            row = {"step": step, **{name: value / count for name, value in sums.items()}}
            progress.append(row); sums = {}
            print(f"{label} step={step:5d} loss={row['total']:.5f} grad={row['grad_norm']:.4f}", flush=True)
    return progress


def train_global_from_global(args: argparse.Namespace, encoder, decoder, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    sampler = uniform_count_range_generator(args.k_min, args.k_max, encoder.num_codewords, generator, encoder.device)
    fading = constant_fading(1, encoder.dtype, encoder.device)

    def loss_fn(_: int):
        batch = sample_batch(encoder, args.batch_size, sampler, fading, random_ebn0(args, generator), generator,
                             encoder.spec.energy_per_codeword)
        output = decoder(encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
        return support_count_loss(output, batch.counts, args.lambda_count, 0.0)

    return optimise(decoder.parameters(), loss_fn, args.steps, args, "global-D0")


def train_l1_binomial(args: argparse.Namespace, global_encoder, sectioned, decoder, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    sampler = uniform_count_range_generator(args.k_min, args.k_max, global_encoder.num_codewords, generator,
                                            global_encoder.device)
    fading = constant_fading(1, global_encoder.dtype, global_encoder.device)

    def loss_fn(_: int):
        batch = sample_batch(global_encoder, args.batch_size, sampler, fading, random_ebn0(args, generator), generator,
                             global_encoder.spec.energy_per_codeword)
        output = decoder(sectioned, batch.Y, batch.H, batch.num_active, batch.noise_var)
        return section_support_count_loss(output, (batch.counts,), args.lambda_count)

    return optimise(decoder.parameters(), loss_fn, args.steps, args, "section-Binomial-D0")


def train_encoder(args: argparse.Namespace, encoder, code, seed: int) -> list[dict]:
    if not args.learn_encoder or args.encoder_steps == 0:
        return []
    decoder = UnrolledSectionedCountPGD(args.d0_layers, power_iters=args.power_iters)
    generator = torch.Generator().manual_seed(seed)
    fading = constant_fading(1, encoder.dtype, encoder.device)

    def loss_fn(_: int):
        K = random_k(args, generator)
        sampler = outer_code_path_generator(K, code, generator, encoder.device)
        batch = sample_sectioned_batch(encoder, args.batch_size, sampler, fading, random_ebn0(args, generator), generator)
        output = decoder(encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
        return section_support_count_loss(output, batch.section_counts, args.lambda_count)

    parameters = list(decoder.parameters()) + [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    progress = optimise(parameters, loss_fn, args.encoder_steps, args, "encoder-local-D0", encoder.apply_constraints)
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    return progress


def train_global_from_sectioned(args: argparse.Namespace, sectioned, code, global_encoder, decoder, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    fading = constant_fading(1, sectioned.dtype, sectioned.device)

    def loss_fn(_: int):
        K = random_k(args, generator)
        sampler = outer_code_path_generator(K, code, generator, sectioned.device)
        batch = sample_sectioned_batch(sectioned, args.batch_size, sampler, fading, random_ebn0(args, generator), generator)
        truth = global_counts_from_paths(batch.active_paths, code, global_encoder.num_codewords, batch.Y.real.dtype)
        output = decoder(global_encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
        return support_count_loss(output, truth, args.lambda_count, 0.0)

    return optimise(decoder.parameters(), loss_fn, args.steps, args, "materialised-global-D0")


def train_local(args: argparse.Namespace, sectioned, code, decoder, seed: int) -> list[dict]:
    data_generator = torch.Generator().manual_seed(seed)
    negative_generator = torch.Generator().manual_seed(seed + 10_000)
    fading = constant_fading(1, sectioned.dtype, sectioned.device)

    def loss_fn(_: int):
        K = random_k(args, data_generator)
        sampler = outer_code_path_generator(K, code, data_generator, sectioned.device)
        batch = sample_sectioned_batch(
            sectioned, args.batch_size, sampler, fading, random_ebn0(args, data_generator), data_generator)
        if isinstance(code, SparseLinearOuterCode):
            output = decoder(sectioned, code, batch.Y, batch.H, batch.num_active, batch.noise_var)
            return sectioned_outer_training_loss(output, batch.section_counts, batch.active_paths, code, sectioned,
                                                 args.lambda_count, args.lambda_marginal, args.lambda_path, 0.0,
                                                 args.num_path_negatives, negative_generator)
        output = decoder(sectioned, batch.Y, batch.H, batch.num_active, batch.noise_var)
        return section_support_count_loss(output, batch.section_counts, args.lambda_count)

    return optimise(decoder.parameters(), loss_fn, args.steps, args, "local-D0-outer" if isinstance(
        code, SparseLinearOuterCode) else "local-D0-no-outer")


def identity_list_decode(log_beliefs: torch.Tensor, code: IdentityOuterCode,
                         soft_counts: tuple[torch.Tensor, ...], K: torch.Tensor,
                         extractor: ValidPathListDecoder) -> PathListOutput:
    list_size = min(extractor.beam_width, int(K.max()) + extractor.list_extra)
    batch_paths, batch_scores, batch_counts = [], [], []
    for b in range(log_beliefs.shape[0]):
        paths = torch.empty(1, 0, dtype=torch.long, device=log_beliefs.device)
        scores = torch.zeros(1, dtype=log_beliefs.dtype, device=log_beliefs.device)
        for ell, size in enumerate(code.section_sizes):
            cap = size if extractor.candidate_cap is None else min(size, extractor.candidate_cap)
            candidates = torch.topk(log_beliefs[b, ell], cap).indices
            paths = torch.cat((paths.repeat_interleave(cap, 0), candidates.repeat(paths.shape[0]).unsqueeze(1)), dim=1)
            expanded_scores = (scores.unsqueeze(1) + log_beliefs[b, ell, candidates].unsqueeze(0)).reshape(-1)
            keep = min(extractor.beam_width, expanded_scores.numel())
            scores, order = torch.topk(expanded_scores, keep); paths = paths.index_select(0, order)
        paths, scores = paths[:list_size], scores[:list_size]
        counts = extractor._fit_multiplicities(paths, tuple(section[b] for section in soft_counts), int(K[b]))
        batch_paths.append(paths); batch_scores.append(scores); batch_counts.append(counts)
    paths = torch.stack(batch_paths); scores = torch.stack(batch_scores); counts = torch.stack(batch_counts)
    return PathListOutput(paths, counts, scores, code.decode_bits(paths), {"collision_mode": "complete_path_multiplicity",
                                                                          "association": "unconstrained_identity_beam"})


def evaluate_l1(args: argparse.Namespace, global_encoder, sectioned, global_decoder, compat_decoder,
                binomial_decoder) -> tuple[list[dict], dict[str, float]]:
    rows, differences = [], {"soft": 0.0, "hard": 0.0, "logits": 0.0}
    fading = constant_fading(1, global_encoder.dtype, global_encoder.device)
    global_decoder.eval(); compat_decoder.eval(); binomial_decoder.eval()
    with torch.no_grad():
        for K in args.eval_k:
            for snr_index, ebn0 in enumerate(args.eval_ebn0):
                generator = torch.Generator().manual_seed(args.seed + 100_000 + 1009 * K + snr_index)
                sampler = uniform_counts_generator(K, global_encoder.num_codewords, generator, global_encoder.device)
                metrics = {name: [] for name in ("global_bernoulli", "section_bernoulli_compat", "section_binomial")}
                collision_batches = []
                for _ in range(args.eval_batches):
                    batch = sample_batch(global_encoder, args.batch_size, sampler, fading, ebn0, generator,
                                         global_encoder.spec.energy_per_codeword)
                    collision_batches += (batch.counts > 1).any(dim=1).to(torch.float32).cpu().tolist()
                    outputs = {"global_bernoulli": global_decoder(global_encoder, batch.Y, batch.H, batch.num_active, batch.noise_var),
                               "section_bernoulli_compat": compat_decoder(sectioned, batch.Y, batch.H, batch.num_active, batch.noise_var),
                               "section_binomial": binomial_decoder(sectioned, batch.Y, batch.H, batch.num_active, batch.noise_var)}
                    compat = outputs["section_bernoulli_compat"]
                    differences["soft"] = max(differences["soft"], float(torch.max(torch.abs(
                        outputs["global_bernoulli"].meta["soft_counts"] - compat.meta["soft_section_counts"][0]))))
                    differences["hard"] = max(differences["hard"], float(torch.max(torch.abs(
                        outputs["global_bernoulli"].counts - compat.section_counts[0]))))
                    for old, new in zip(outputs["global_bernoulli"].meta["layer_logits"], compat.meta["section_layer_logits"]):
                        differences["logits"] = max(differences["logits"], float(torch.max(torch.abs(old - new[0]))))
                    for name, output in outputs.items():
                        counts = output.counts if name == "global_bernoulli" else output.section_counts[0]
                        per, _ = batch_evaluate(batch.counts, counts.to(batch.counts), max_list_size=K); metrics[name] += per
                for name, values in metrics.items():
                    rows.append({"decoder": name, "K": K, "ebn0_db": float(ebn0),
                                 "empirical_any_complete_collision": sum(collision_batches) / len(collision_batches),
                                 "theoretical_any_complete_collision": 1.0 - math.prod(
                                     1.0 - i / global_encoder.num_codewords for i in range(K)),
                                 **aggregate_metrics(values)})
    if max(differences.values()) > 2e-5:
        raise AssertionError(f"L=1 compatibility path differs from global D0: {differences}")
    return rows, differences


def evaluate_lgt(args: argparse.Namespace, sectioned, code, global_encoder, global_decoder, local_decoder) -> tuple[list[dict], float]:
    rows, max_signal_error = [], 0.0
    extractor = ValidPathListDecoder(args.beam_width, args.list_extra, args.candidate_cap)
    fading = constant_fading(1, sectioned.dtype, sectioned.device)
    global_decoder.eval(); local_decoder.eval(); sectioned.eval()
    with torch.no_grad():
        for K in args.eval_k:
            for snr_index, ebn0 in enumerate(args.eval_ebn0):
                generator = torch.Generator().manual_seed(args.seed + 100_000 + 1009 * K + snr_index)
                sampler = outer_code_path_generator(K, code, generator, sectioned.device)
                values = {"materialised_global_d0": [], "local_d0_association": []}
                if isinstance(code, SparseLinearOuterCode): values["local_d0_bp_association"] = []
                complete_collision_batches, local_collision_batches = [], []
                for _ in range(args.eval_batches):
                    batch = sample_sectioned_batch(sectioned, args.batch_size, sampler, fading, ebn0, generator)
                    truth = global_counts_from_paths(batch.active_paths, code, global_encoder.num_codewords, batch.Y.real.dtype)
                    complete_collision_batches += (truth > 1).any(dim=1).to(torch.float32).cpu().tolist()
                    local_collision_batches += torch.stack(
                        [(counts > 1).any(dim=1) for counts in batch.section_counts], dim=1).any(dim=1).to(
                            torch.float32).cpu().tolist()
                    max_signal_error = max(max_signal_error, float(torch.max(torch.abs(
                        global_encoder.matvec(truth) - batch.y_clean))))
                    global_output = global_decoder(global_encoder, batch.Y, batch.H, batch.num_active, batch.noise_var)
                    per, _ = batch_evaluate(truth, global_output.counts.to(truth), max_list_size=K)
                    values["materialised_global_d0"] += per
                    if isinstance(code, SparseLinearOuterCode):
                        output = local_decoder(sectioned, code, batch.Y, batch.H, batch.num_active, batch.noise_var)
                        soft = output.meta["d0_output"].meta["soft_section_counts"]
                        d0_logits = torch.stack(output.meta["d0_output"].meta["section_support_logits"], dim=1)
                        d0_output = OuterBPOutput(torch.log_softmax(d0_logits, dim=-1))
                        d0_list = extractor.decode(d0_output, code, soft, batch.num_active)
                        bp_list = extractor.decode(output, code, soft, batch.num_active)
                        values["local_d0_association"] += path_list_pupe(d0_list, batch.active_paths).cpu().tolist()
                        values["local_d0_bp_association"] += path_list_pupe(bp_list, batch.active_paths).cpu().tolist()
                    else:
                        output = local_decoder(sectioned, batch.Y, batch.H, batch.num_active, batch.noise_var)
                        logits = torch.stack(output.meta["section_support_logits"], dim=1)
                        listed = identity_list_decode(torch.log_softmax(logits, dim=-1), code,
                                                      output.meta["soft_section_counts"], batch.num_active, extractor)
                        values["local_d0_association"] += path_list_pupe(listed, batch.active_paths).cpu().tolist()
                for name, samples in values.items():
                    common = {"decoder": name, "K": K, "ebn0_db": float(ebn0),
                              "empirical_any_complete_collision": sum(complete_collision_batches) / len(
                                  complete_collision_batches),
                              "empirical_any_local_collision": sum(local_collision_batches) / len(local_collision_batches),
                              "theoretical_any_complete_collision": 1.0 - math.prod(
                                  1.0 - i / global_encoder.num_codewords for i in range(K))}
                    if name == "materialised_global_d0":
                        rows.append({**common, **aggregate_metrics(samples)})
                    else:
                        tensor = torch.as_tensor(samples, dtype=torch.float64)
                        rows.append({**common, "pupe": float(tensor.mean()),
                                     "pupe_standard_error": float(tensor.std(unbiased=False) / math.sqrt(tensor.numel()))})
    if max_signal_error > 2e-5:
        raise AssertionError(f"materialised induced codebook differs from section synthesis by {max_signal_error:.3e}")
    return rows, max_signal_error


def run_l1(args: argparse.Namespace) -> dict:
    encoder_generator = torch.Generator().manual_seed(args.seed + 1)
    global_encoder = make_global_encoder(args, encoder_generator)
    sectioned = sectioned_from_explicit(global_encoder)
    initial = UnrolledBernoulliPGD(args.d0_layers, power_iters=args.power_iters)
    global_decoder = copy.deepcopy(initial)
    binomial_decoder = UnrolledSectionedCountPGD(args.d0_layers, power_iters=args.power_iters)
    binomial_decoder.load_state_dict(initial.state_dict())
    lipschitz = global_encoder.spectral_norm_squared(args.power_iters)
    sectioned._spectral_cache[args.power_iters] = lipschitz
    progress_global = train_global_from_global(args, global_encoder, global_decoder, args.seed + 10_000)
    progress_binomial = train_l1_binomial(args, global_encoder, sectioned, binomial_decoder, args.seed + 10_000)
    compat_decoder = UnrolledSectionedCountPGD(
        args.d0_layers, power_iters=args.power_iters, prior_mode="bernoulli_compat")
    compat_decoder.load_state_dict(global_decoder.state_dict())
    rows, differences = evaluate_l1(args, global_encoder, sectioned, global_decoder, compat_decoder, binomial_decoder)
    return {"metadata": {"args": vars(args), "bridge_claim": "same physical codebook and exact old/new D0 equations",
                         "M": global_encoder.num_codewords, "L": sectioned.num_sections,
                         "section_sizes": list(sectioned.section_sizes), "compatibility_max_abs_difference": differences,
                         "mean_codeword_energy": global_encoder.mean_codeword_energy()},
            "progress": {"global_bernoulli": progress_global, "section_binomial": progress_binomial}, "rows": rows,
            "checkpoint": {"global_encoder": global_encoder.state_dict(), "global_decoder": global_decoder.state_dict(),
                           "section_binomial_decoder": binomial_decoder.state_dict()}}


def run_lgt(args: argparse.Namespace) -> dict:
    code = triadic_outer_code(args.payload_bits, args.section_bits) if args.outer_code == "triadic" else IdentityOuterCode(
        args.payload_bits, args.section_bits)
    spec = SectionedURASpec(args.n, args.payload_bits, args.k_max)
    sectioned = build_orthogonal_sectioned_encoder(spec, code, learn_C=args.learn_encoder,
                                                   mixing_stages=args.mixing_stages, dtype=torch.float32,
                                                   generator=torch.Generator().manual_seed(args.seed + 1))
    encoder_progress = train_encoder(args, sectioned, code, args.seed + 20_000)
    global_encoder, Phi = materialise_global_encoder(sectioned, code, args.k_max)
    initial = UnrolledBernoulliPGD(args.d0_layers, power_iters=args.power_iters)
    global_decoder = copy.deepcopy(initial)
    if isinstance(code, SparseLinearOuterCode):
        local_decoder = SectionedOuterDecoder(args.d0_layers, args.bp_iterations, args.power_iters)
        local_decoder.d0.load_state_dict(initial.state_dict())
    else:
        local_decoder = UnrolledSectionedCountPGD(args.d0_layers, power_iters=args.power_iters)
        local_decoder.load_state_dict(initial.state_dict())
    global_progress = train_global_from_sectioned(args, sectioned, code, global_encoder, global_decoder, args.seed + 30_000)
    local_progress = train_local(args, sectioned, code, local_decoder, args.seed + 30_000)
    rows, signal_error = evaluate_lgt(args, sectioned, code, global_encoder, global_decoder, local_decoder)
    energy = sectioned.certify_exact_energy()
    if not energy["guaranteed"]:
        raise AssertionError(f"bridge encoder lacks an exact unit-energy certificate: {energy}")
    metadata = {"args": vars(args), "bridge_claim": "one frozen L>1 encoder with global and scalable decoder routes",
                "M_materialised_for_certification": Phi.shape[1], "induced_phi_shape": list(Phi.shape),
                "L": sectioned.num_sections, "section_sizes": list(sectioned.section_sizes),
                "local_state_size": sectioned.state_size, "outer_constraints": isinstance(code, SparseLinearOuterCode),
                "materialisation_is_small_B_only": True, "max_signal_equivalence_error": signal_error,
                "energy_certificate": energy, "sampled_energy": sampled_energy_report(
                    sectioned, code, min(4096, 1 << args.payload_bits), torch.Generator().manual_seed(args.seed + 40_000))}
    return {"metadata": metadata, "progress": {"encoder": encoder_progress, "materialised_global_d0": global_progress,
                                                "local_decoder": local_progress}, "rows": rows,
            "checkpoint": {"sectioned_encoder": sectioned.state_dict(), "global_decoder": global_decoder.state_dict(),
                           "local_decoder": local_decoder.state_dict()}}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = run_l1(args) if args.bridge == "l1" else run_lgt(args)
    checkpoint = result.pop("checkpoint")
    result["metadata"]["wall_s"] = time.time() - started
    values = [row["pupe"] for row in result["rows"]]
    if not values or not all(math.isfinite(value) for value in values):
        raise AssertionError("bridge evaluation produced missing or nonfinite PUPE")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    torch.save({"metadata": result["metadata"], **checkpoint}, out_dir / "checkpoint.pt")
    print(f"bridge={args.bridge} rows={len(result['rows'])} mean_PUPE={sum(values) / len(values):.4f} "
          f"wall={result['metadata']['wall_s']:.1f}s", flush=True)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
