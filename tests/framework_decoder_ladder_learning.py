"""Deterministic local learning check for the D0--D4 decoder ladder.

This is a small-B verification runner, not a performance experiment.  D0--D2
see the complete materialised message alphabet.  D3--D4 receive an explicitly
oracle-aided bounded list containing every transmitted message plus random
distractors; their results therefore test conditional refinement only, not
candidate search.  Validation is deterministic and the epoch-zero model is an
early-stopping candidate, so restoring the best state can never silently make
the decoder worse than its initialisation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.candidate_decoders import (CandidateRestrictedEffectiveChannelPGD, CandidateSet,
                                          LearnedCandidateGeometryPGD, candidate_count_targets)
from framework.channel import constant_fading, sample_batch
from framework.core import URASpec
from framework.early_stopping import EarlyStopping
from framework.hash_skeleton import all_message_bits
from framework.learned_decoders import FactorAttentionISTANet, UnrolledBernoulliPGD, UnrolledEffectiveChannelPGD
from framework.losses import effective_channel_loss, support_count_loss
from framework.metrics import batch_evaluate
from framework.prototype_amplitudes import build_prototype_hash_encoder


PRESETS = {
    "smoke": {"payload_bits": 4, "n": 16, "support_size": 4, "label_bits": 2, "num_active": 1,
              "candidate_size": 8, "num_layers": 2, "max_epochs": 3, "batches_per_epoch": 2,
              "batch_size": 4, "validation_batches": 2, "hidden_dim": 8, "power_iters": 3},
    "laptop": {"payload_bits": 6, "n": 32, "support_size": 8, "label_bits": 3, "num_active": 3,
               "candidate_size": 24, "num_layers": 4, "max_epochs": 200, "batches_per_epoch": 20,
               "batch_size": 16, "validation_batches": 8, "hidden_dim": 24, "power_iters": 8},
}
DECODER_NAMES = ("d0", "d1", "d2", "d3", "d4")


def decoder_names(text: str) -> list[str]:
    values = [value.strip().lower() for value in text.split(",") if value.strip()]
    invalid = [value for value in values if value not in DECODER_NAMES]
    if not values or invalid or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"expected unique comma-separated values from {','.join(DECODER_NAMES)}")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="laptop")
    parser.add_argument("--decoders", type=decoder_names, default=list(DECODER_NAMES))
    parser.add_argument("-B", "--payload-bits", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--support-size", type=int)
    parser.add_argument("--label-bits", type=int)
    parser.add_argument("--num-active", type=int)
    parser.add_argument("--candidate-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--power-iters", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--train-ebn0-min", type=float, default=0.0)
    parser.add_argument("--train-ebn0-max", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lambda-count", type=float, default=0.1)
    parser.add_argument("--lambda-symmetry", type=float, default=0.01)
    parser.add_argument("--hash-search-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9331)
    parser.add_argument("--out-dir", default="results/framework_decoder_ladder_learning")
    args = parser.parse_args(argv)
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def validate_args(args: argparse.Namespace) -> None:
    M = 1 << int(args.payload_bits)
    positive = ("payload_bits", "n", "support_size", "num_active", "candidate_size", "num_layers",
                "hidden_dim", "power_iters", "max_epochs", "batches_per_epoch", "batch_size", "validation_batches")
    if any(int(getattr(args, name)) <= 0 for name in positive):
        raise SystemExit("all size, layer, epoch, and batch arguments must be positive")
    if int(args.payload_bits) > 20:
        raise SystemExit("this full-axis D0--D4 comparison is intentionally restricted to B <= 20")
    if not 0 <= int(args.label_bits) <= int(args.payload_bits):
        raise SystemExit("--label-bits must lie in [0,B]")
    if not int(args.num_active) < int(args.candidate_size) <= M:
        raise SystemExit("the oracle candidate size must satisfy K < C <= 2^B")
    if int(args.n) % int(args.support_size):
        raise SystemExit("--support-size must divide n")
    bins = int(args.n) // int(args.support_size)
    if bins & (bins - 1):
        raise SystemExit("n/support_size must be a power of two")
    if bins.bit_length() - 1 > int(args.payload_bits):
        raise SystemExit("the affine hash bin width log2(n/support_size) cannot exceed B")
    if args.train_ebn0_max < args.train_ebn0_min:
        raise SystemExit("--train-ebn0-max must be at least --train-ebn0-min")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0.0:
        raise SystemExit("early-stopping patience and minimum delta must be nonnegative")


def make_decoder(name: str, args: argparse.Namespace):
    if name == "d0":
        return UnrolledBernoulliPGD(args.num_layers, power_iters=args.power_iters)
    if name == "d1":
        return FactorAttentionISTANet(args.num_layers, hidden_dim=args.hidden_dim, pattern_slots=2,
                                      value_slots=2, global_slots=4, power_iters=args.power_iters)
    if name == "d2":
        return UnrolledEffectiveChannelPGD(args.num_layers)
    if name == "d3":
        return CandidateRestrictedEffectiveChannelPGD(args.num_layers)
    if name == "d4":
        return LearnedCandidateGeometryPGD(args.num_layers, hidden_dim=args.hidden_dim)
    raise ValueError(name)


def random_ebn0(args: argparse.Namespace, generator: torch.Generator) -> float:
    weight = float(torch.rand((), generator=generator))
    return args.train_ebn0_min + weight * (args.train_ebn0_max - args.train_ebn0_min)


def oracle_candidates(active: torch.Tensor, all_bits: torch.Tensor, size: int,
                      generator: torch.Generator) -> CandidateSet:
    """Truth-containing random lists; the ordering carries no truth information."""
    batch, _ = active.shape
    M = all_bits.shape[0]
    rows = []
    for sample in range(batch):
        truth = torch.unique(active[sample], sorted=False)
        permutation = torch.randperm(M, generator=generator, device=active.device)
        distractors = permutation[~torch.isin(permutation, truth)][:size - truth.numel()]
        chosen = torch.cat((truth, distractors))
        chosen = chosen[torch.randperm(size, generator=generator, device=active.device)]
        rows.append(chosen)
    indices = torch.stack(rows)
    metadata = {"oracle_aided": True, "candidate_search_solved": False, "global_search": False,
                "construction": "all_unique_transmitted_messages_plus_random_distractors"}
    return CandidateSet(message_bits=all_bits.to(active.device)[indices], source="oracle_truth_containing_random_list",
                        metadata=metadata)


def distinct_counts_sampler(num_active: int, num_codewords: int, generator: torch.Generator, device: torch.device):
    """Sample without message collisions, matching D2--D4's large-B Bernoulli regime."""
    def sample(batch_size: int):
        active = torch.stack([torch.randperm(num_codewords, generator=generator, device=device)[:num_active]
                              for _ in range(batch_size)])
        counts = torch.zeros(batch_size, num_codewords, dtype=torch.float32, device=device)
        counts.scatter_(1, active, 1.0)
        return counts, active
    return sample


def sample_problem(encoder, args: argparse.Namespace, generator: torch.Generator):
    counts_sampler = distinct_counts_sampler(args.num_active, encoder.num_codewords, generator, encoder.device)
    fading = constant_fading(encoder.spec.num_antennas, encoder.dtype, encoder.device)
    batch = sample_batch(encoder, args.batch_size, counts_sampler, fading, random_ebn0(args, generator), generator,
                         energy_per_codeword=encoder.spec.energy_per_codeword)
    all_bits = all_message_bits(args.payload_bits).to(encoder.device)
    candidates = oracle_candidates(batch.active_messages, all_bits, args.candidate_size, generator)
    truth_bits = all_bits[batch.active_messages]
    return batch, candidates, truth_bits


def forward_and_target(name: str, decoder, encoder, batch, candidates: CandidateSet, truth_bits: torch.Tensor):
    if name in {"d3", "d4"}:
        output = decoder(encoder, batch.Y, batch.H, batch.num_active, noise_var=batch.noise_var,
                         candidates=candidates, true_message_bits=truth_bits)
        target = candidate_count_targets(output.candidates, true_message_bits=truth_bits)
        recall = output.meta["candidate_recall"]
        if not torch.equal(recall, torch.ones_like(recall)):
            raise AssertionError("the explicitly oracle-aided candidate construction lost a transmitted message")
        return output, target
    return decoder(encoder, batch.Y, batch.H, batch.num_active, noise_var=batch.noise_var), batch.counts.real


def loss_for(name: str, output, target: torch.Tensor, args: argparse.Namespace):
    if name in {"d2", "d3", "d4"}:
        return effective_channel_loss(output, target, args.lambda_count)
    return support_count_loss(output, target, args.lambda_count, args.lambda_symmetry)


def fixed_validation(name: str, decoder, encoder, args: argparse.Namespace, seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    was_training = decoder.training
    decoder.eval()
    loss_sum = pupe_sum = recall_sum = 0.0
    with torch.no_grad():
        for _ in range(args.validation_batches):
            batch, candidates, truth_bits = sample_problem(encoder, args, generator)
            output, target = forward_and_target(name, decoder, encoder, batch, candidates, truth_bits)
            loss, _ = loss_for(name, output, target, args)
            _, metrics = batch_evaluate(target, output.counts.to(target), max_list_size=args.num_active)
            loss_sum += float(loss); pupe_sum += float(metrics["pupe"])
            if name in {"d3", "d4"}:
                recall_sum += float(output.meta["candidate_recall_mean"])
    decoder.train(was_training)
    denominator = float(args.validation_batches)
    result = {"loss": loss_sum / denominator, "pupe": pupe_sum / denominator}
    if name in {"d3", "d4"}:
        result["candidate_recall"] = recall_sum / denominator
    return result


def train_one(name: str, encoder, args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed + 1000 + DECODER_NAMES.index(name))
    decoder = make_decoder(name, args).to(encoder.device)
    parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    stopper = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    validation_seed = args.seed + 300_000
    initial = fixed_validation(name, decoder, encoder, args, validation_seed)
    stopper.update(initial["loss"], 0, {"decoder": decoder})
    train_generator = torch.Generator().manual_seed(args.seed + 100_000)
    progress: list[dict] = []
    started = time.time()
    for epoch in range(1, args.max_epochs + 1):
        decoder.train(); sums = {"support": 0.0, "count": 0.0, "symmetry": 0.0, "total": 0.0}
        for _ in range(args.batches_per_epoch):
            batch, candidates, truth_bits = sample_problem(encoder, args, train_generator)
            output, target = forward_and_target(name, decoder, encoder, batch, candidates, truth_bits)
            loss, parts = loss_for(name, output, target, args)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip); optimizer.step()
            for key in sums:
                sums[key] += float(parts[key].detach())
        validation = fixed_validation(name, decoder, encoder, args, validation_seed)
        record = {"epoch": epoch, **{f"training_{key}": value / args.batches_per_epoch for key, value in sums.items()},
                  "validation_loss": validation["loss"], "validation_pupe": validation["pupe"]}
        if "candidate_recall" in validation:
            record["candidate_recall"] = validation["candidate_recall"]
        progress.append(record)
        print(f"{name.upper()} epoch={epoch:3d} train={record['training_total']:.6f} "
              f"validation={validation['loss']:.6f} PUPE={validation['pupe']:.4f}", flush=True)
        if stopper.update(validation["loss"], epoch, {"decoder": decoder}):
            break
    last_epoch = fixed_validation(name, decoder, encoder, args, validation_seed)
    restored = stopper.restore({"decoder": decoder}) if stopper.enabled else False
    final = fixed_validation(name, decoder, encoder, args, validation_seed)
    values = [initial["loss"], initial["pupe"], last_epoch["loss"], final["loss"], final["pupe"]]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{name.upper()} produced a non-finite local-learning diagnostic")
    stopping = stopper.summary(len(progress), restored)
    stopping.update({"initial_validation_loss": initial["loss"], "last_epoch_validation_loss": last_epoch["loss"],
                     "final_validation_loss": final["loss"], "final_is_restored_best": restored,
                     "best_decreased_from_initial": stopper.best_value < initial["loss"] - 1e-12})
    if stopper.enabled and abs(final["loss"] - stopper.best_value) > 1e-7:
        raise AssertionError(f"{name.upper()} did not restore the deterministic best validation state")
    return {"decoder": name.upper(), "parameter_count": sum(parameter.numel() for parameter in parameters),
            "candidate_mode": "oracle_truth_containing" if name in {"d3", "d4"} else None,
            "candidate_search_tested": False if name in {"d3", "d4"} else None,
            "initial_validation": initial, "final_validation": final, "progress": progress,
            "early_stopping": stopping, "wall_s": time.time() - started}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); validate_args(args)
    torch.manual_seed(args.seed)
    spec = URASpec(n=args.n, num_codewords=1 << args.payload_bits, num_active=args.num_active,
                   num_antennas=1, payload_bits=args.payload_bits)
    encoder = build_prototype_hash_encoder(spec, args.support_size, args.seed, args.label_bits, False,
                                           "gaussian", args.hash_search_candidates)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise AssertionError("decoder-ladder check requires a frozen encoder")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = [train_one(name, encoder, args) for name in args.decoders]
    payload = {"purpose": "small_B_decoder_learning_verification", "not_a_performance_comparison": True,
               "encoder": "fixed_selected_affine_hash_with_fixed_prototype_amplitudes", "args": vars(args),
               "candidate_boundary": {"D3_D4": "oracle truth-containing lists; conditional refinement only",
                                      "candidate_search_tested": False}, "results": results}
    path = out_dir / "summary.json"; path.write_text(json.dumps(payload, indent=2, default=str))
    print("\nDecoder  initial val   best val      final val     epochs  candidate mode")
    for result in results:
        stopping = result["early_stopping"]
        print(f"{result['decoder']:<8} {stopping['initial_validation_loss']:<13.6f} "
              f"{stopping['best_validation_loss']:<13.6f} {stopping['final_validation_loss']:<13.6f} "
              f"{stopping['epochs_run']:<7d} {result['candidate_mode'] or 'full small-B alphabet'}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
