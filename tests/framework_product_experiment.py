"""Train and evaluate the reduced L=1 factorised URA encoder/decoder experiments.

Normal training and inference use only Encoder.matvec/rmatvec. Materialising Phi
is reserved for optional tiny-scale oracle diagnostics outside this runner.
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

from framework.channel import constant_fading, sample_batch, uniform_count_range_generator, uniform_counts_generator  # noqa: E402
from framework.core import URASpec  # noqa: E402
from framework.encoder import ComponentConstraints, build_encoder  # noqa: E402
from framework.learned_decoders import (FactorAttentionISTANet, UnrolledBernoulliPGD,
                                        UnrolledNonnegativeISTA, matched_filter_decoder)  # noqa: E402
from framework.losses import support_count_loss  # noqa: E402
from framework.metrics import aggregate_metrics, batch_evaluate  # noqa: E402
from framework.pipeline import (dense_component_specs, odma_component_specs,
                                product_all_pairs_component_specs, sparse_global_component_specs)  # noqa: E402


def parse_float_grid(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_grid(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", choices=["product_fixed", "product_learned", "dense_fixed", "dense_learned",
                                                   "odma_fixed", "sparse_global_fixed"], default="product_fixed")
    p.add_argument("--decoder", choices=["d0", "d1", "ista"], default="d0")
    p.add_argument("-B", "--payload-bits", type=int, default=12)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--Q", type=int, default=4)
    p.add_argument("--odma-d", type=int, default=None)
    p.add_argument("--sparse-support", type=int, default=None)
    p.add_argument("--num-antennas", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--pattern-slots", type=int, default=2)
    p.add_argument("--value-slots", type=int, default=2)
    p.add_argument("--global-slots", type=int, default=4)
    p.add_argument("--power-iters", type=int, default=12)
    p.add_argument("--k-min", type=int, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--eval-k", type=str, default=None, help="comma-separated; default uses low/mid/high training loads")
    p.add_argument("--extrapolate-k", action="store_true")
    p.add_argument("--train-ebn0-min", type=float, default=-4.0)
    p.add_argument("--train-ebn0-max", type=float, default=12.0)
    p.add_argument("--eval-ebn0", type=parse_float_grid, default=parse_float_grid("-4,0,4,8,12"))
    p.add_argument("--encoder-epochs", type=int, default=10)
    p.add_argument("--decoder-epochs", type=int, default=20)
    p.add_argument("--batches-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-count", type=float, default=0.1)
    p.add_argument("--lambda-symmetry", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/framework_product_experiment")
    return p.parse_args(argv)


def load_range(args: argparse.Namespace) -> tuple[int, int, list[int]]:
    k_min = int(args.k_min) if args.k_min is not None else max(1, round(0.4 * args.n / args.payload_bits))
    k_max = int(args.k_max) if args.k_max is not None else max(k_min, round(1.2 * args.n / args.payload_bits))
    if args.eval_k is None:
        eval_k = sorted({k_min, round(0.8 * args.n / args.payload_bits), k_max})
    else:
        eval_k = parse_int_grid(args.eval_k)
    if args.extrapolate_k:
        eval_k = sorted(set(eval_k + [round(1.4 * args.n / args.payload_bits)]))
    if k_min <= 0 or k_max < k_min:
        raise ValueError(f"invalid K range [{k_min}, {k_max}]")
    return k_min, k_max, eval_k


def build_experiment_encoder(args: argparse.Namespace, gen: torch.Generator):
    M = 1 << int(args.payload_bits)
    k_min, k_max, _ = load_range(args)
    spec = URASpec(n=int(args.n), num_codewords=M, num_active=k_max, num_antennas=int(args.num_antennas),
                   payload_bits=int(args.payload_bits))
    learn_C = args.encoder.endswith("learned")
    if args.encoder.startswith("product"):
        components = product_all_pairs_component_specs(spec, int(args.Q), learn_C, "random_sign_diagonal")
    elif args.encoder.startswith("dense"):
        components = dense_component_specs(spec, learn_C)
    elif args.encoder == "odma_fixed":
        d = int(args.odma_d) if args.odma_d is not None else max(spec.n // int(args.Q), 1)
        components = odma_component_specs(spec, d, int(args.Q), False, False)
    elif args.encoder == "sparse_global_fixed":
        support = int(args.sparse_support) if args.sparse_support is not None else max(spec.n // int(args.Q), 1)
        components = sparse_global_component_specs(spec, support, gen)
    else:
        raise ValueError(args.encoder)
    constraints = [ComponentConstraints(C="unit_norm_columns" if c.learn_C else "none") for c in components]
    encoder = build_encoder(spec, components, constraints=constraints, dtype=torch.float32, generator=gen)
    return encoder, k_min, k_max


def make_decoder(args: argparse.Namespace):
    common = {"num_layers": int(args.num_layers), "power_iters": int(args.power_iters)}
    if args.decoder == "d0":
        return UnrolledBernoulliPGD(**common)
    if args.decoder == "d1":
        return FactorAttentionISTANet(hidden_dim=int(args.hidden_dim), pattern_slots=int(args.pattern_slots),
                                      value_slots=int(args.value_slots), global_slots=int(args.global_slots), **common)
    return UnrolledNonnegativeISTA(**common)


def random_ebn0(args: argparse.Namespace, gen: torch.Generator) -> float:
    u = float(torch.rand((), generator=gen).item())
    return float(args.train_ebn0_min + u * (args.train_ebn0_max - args.train_ebn0_min))


def train_phase(name: str, encoder, decoder, parameters, counts_sampler, fading_sampler,
                args: argparse.Namespace, gen: torch.Generator, epochs: int) -> list[dict]:
    if epochs <= 0:
        return []
    opt = torch.optim.Adam(parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    progress = []
    for epoch in range(1, epochs + 1):
        sums = {"support": 0.0, "count": 0.0, "symmetry": 0.0, "total": 0.0}
        decoder.train()
        for _ in range(int(args.batches_per_epoch)):
            ebn0_db = random_ebn0(args, gen)
            batch = sample_batch(encoder, int(args.batch_size), counts_sampler, fading_sampler, ebn0_db, gen,
                                 energy_per_codeword=encoder.spec.energy_per_codeword)
            out = decoder(encoder, batch.Y, batch.H, batch.num_active, noise_var=batch.noise_var)
            loss, parts = support_count_loss(out, batch.counts, args.lambda_count, args.lambda_symmetry)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(args.grad_clip))
            opt.step()
            if any(p.requires_grad for p in encoder.parameters()):
                encoder.apply_constraints()
            for key in sums:
                sums[key] += float(parts[key].detach())
        record = {"phase": name, "epoch": epoch,
                  **{key: value / int(args.batches_per_epoch) for key, value in sums.items()}}
        progress.append(record)
        print(f"{name} epoch={epoch:3d} loss={record['total']:.5f} support={record['support']:.5f} "
              f"count={record['count']:.5f} symmetry={record['symmetry']:.5f}", flush=True)
    return progress


def evaluate_one(encoder, decoder, K: int, ebn0_db: float, args: argparse.Namespace,
                 gen: torch.Generator, fading_sampler) -> tuple[dict, dict]:
    sampler = uniform_counts_generator(K, encoder.num_codewords, gen, encoder.device)
    rows_learned, rows_matched = [], []
    collision_batches = []
    decoder.eval()
    with torch.no_grad():
        for _ in range(int(args.eval_batches)):
            batch = sample_batch(encoder, int(args.batch_size), sampler, fading_sampler, ebn0_db, gen,
                                 energy_per_codeword=encoder.spec.energy_per_codeword)
            learned = decoder(encoder, batch.Y, batch.H, batch.num_active, noise_var=batch.noise_var)
            matched = matched_filter_decoder(encoder, batch.Y, batch.H, batch.num_active, noise_var=batch.noise_var)
            learned_rows, _ = batch_evaluate(batch.counts, learned.counts.to(batch.counts), max_list_size=K)
            matched_rows, _ = batch_evaluate(batch.counts, matched.counts.to(batch.counts), max_list_size=K)
            rows_learned.extend(learned_rows); rows_matched.extend(matched_rows)
            collision_batches.extend((batch.counts > 1).any(dim=1).to(torch.float32).cpu().tolist())
    actual_Q = encoder.components[0].Q if len(encoder.components) == 1 else 1
    common = {"K": K, "ebn0_db": ebn0_db, "expected_users_per_pattern": K / actual_Q,
              "empirical_any_collision": sum(collision_batches) / max(len(collision_batches), 1),
              "theoretical_any_collision": 1.0 - math.prod(1.0 - i / encoder.num_codewords for i in range(K))}
    return {**common, **aggregate_metrics(rows_learned)}, {**common, **aggregate_metrics(rows_matched)}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_antennas <= 0:
        raise SystemExit("--num-antennas must be positive")
    if args.train_ebn0_max < args.train_ebn0_min:
        raise SystemExit("--train-ebn0-max must be >= --train-ebn0-min")
    torch.manual_seed(int(args.seed))
    gen = torch.Generator().manual_seed(int(args.seed))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    encoder, k_min, k_max = build_experiment_encoder(args, gen)
    _, _, eval_k = load_range(args)
    train_sampler = uniform_count_range_generator(k_min, k_max, encoder.num_codewords, gen, encoder.device)
    fading_sampler = constant_fading(encoder.spec.num_antennas, encoder.dtype, encoder.device)
    progress, t0 = [], time.time()

    learned_encoder = any(p.requires_grad for p in encoder.parameters())
    decoder = None
    if learned_encoder:
        surrogate = UnrolledBernoulliPGD(num_layers=int(args.num_layers), power_iters=int(args.power_iters))
        params = list(surrogate.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
        progress += train_phase("encoder_d0", encoder, surrogate, params, train_sampler, fading_sampler,
                                args, gen, int(args.encoder_epochs))
        for p in encoder.parameters():
            p.requires_grad_(False)
        if args.decoder == "d0":
            decoder = surrogate
    if decoder is None:
        decoder = make_decoder(args)
    progress += train_phase("decoder", encoder, decoder, list(decoder.parameters()), train_sampler, fading_sampler,
                            args, gen, int(args.decoder_epochs))

    learned_results, matched_results = [], []
    for K in eval_k:
        for ebn0_db in args.eval_ebn0:
            learned, matched = evaluate_one(encoder, decoder, int(K), float(ebn0_db), args, gen, fading_sampler)
            learned_results.append(learned); matched_results.append(matched)
            print(f"eval K={K:3d} Eb/N0={ebn0_db:5.1f} learned PUPE={learned['pupe']:.4f} "
                  f"matched PUPE={matched['pupe']:.4f} collision={learned['empirical_any_collision']:.3f}", flush=True)

    component = encoder.components[0]
    metadata = {"args": vars(args), "K_train": [k_min, k_max], "K_eval": eval_k,
                "M": encoder.num_codewords, "V": component.V, "d": component.d,
                "operator_storage_shape": list(component.R.shape), "implicit_forward": True,
                "decoder_knows_K": True, "decoder_knows_noise_variance": True,
                "receiver_knows_fading": True, "single_antenna_default": True,
                "wall_s": time.time() - t0}
    checkpoint = {"metadata": metadata, "encoder": encoder.state_dict(), "decoder": decoder.state_dict()}
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    (out_dir / "summary.json").write_text(json.dumps({"metadata": metadata, "progress": progress,
                                                       "learned": learned_results, "matched_filter": matched_results},
                                                      indent=2, default=str))
    print(f"Wrote {out_dir / 'summary.json'} and {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
