"""Command-line entry point for framework training / inference.

Example:
    python -m framework.pipeline --mode infer --preset dense \
        --num-codewords 32 --n 64 --num-active 4 --ebn0-db 5 --num-antennas 2

    python -m framework.pipeline --mode train --preset odma \
        --num-codewords 32 --n 64 --num-active 4 --ebn0-db 5 \
        --epochs 10 --batches-per-epoch 50 --batch-size 32 \
        --learn-C --lambda-power 1e-3

A *preset* fixes a sensible default ComponentSpec list (dense / odma / ccs).
All flags can be overridden from the command line, and per-matrix init choices
are exposed so future experiments can swap them without code changes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .analysis import analyze_encoder
from .channel import constant_fading, iid_gaussian_fading, sample_batch
from .core import ComponentSpec, URASpec
from .datasets import DatasetConfig, generate_uniform_count_dataset, load_count_dataset, make_dataset_sampler
from .decoders import get_decoder
from .encoder import ComponentConstraints, build_encoder
from .metrics import batch_evaluate
from .plotting import plot_count_estimate, plot_training_curves
from .training import TrainConfig, evaluate, train


# --- presets --------------------------------------------------------------


def dense_component_specs(spec: URASpec, learn_C: bool) -> list[ComponentSpec]:
    return [ComponentSpec(Q=1, d=spec.n, V=spec.num_codewords, N=spec.num_codewords,
                            R_init="identity", C_init="random_gaussian",
                            U_init="all_pairs", T_init="identity",
                            learn_R=False, learn_C=learn_C)]


def odma_component_specs(spec: URASpec, d: int, num_blocks: int, learn_C: bool,
                         learn_R: bool) -> list[ComponentSpec]:
    """Legacy ODMA+URA convention from src/: message m uses block m mod Q."""
    if d <= 0 or d > spec.n:
        raise ValueError(f"ODMA requires 0 < d <= n, got d={d}, n={spec.n}")
    msg = torch.arange(spec.num_codewords)
    return [ComponentSpec(Q=num_blocks, d=d, V=spec.num_codewords, N=spec.num_codewords,
                            R_init="random_placements", C_init="random_gaussian",
                            U_init="explicit", T_init="identity",
                            learn_R=learn_R, learn_C=learn_C,
                            explicit_atom_q=msg % num_blocks,
                            explicit_atom_v=msg)]


def ccs_component_specs(spec: URASpec, num_sections: int, learn_C: bool) -> list[ComponentSpec]:
    if spec.n % num_sections != 0:
        raise ValueError(
            f"n ({spec.n}) must be divisible by num_sections ({num_sections}) for the ccs preset")
    d = spec.n // num_sections
    J = max(int(math.ceil(spec.num_codewords ** (1.0 / num_sections))), 2)
    while J ** num_sections < spec.num_codewords:
        J += 1
    components: list[ComponentSpec] = []
    for l in range(num_sections):
        section_R = torch.zeros(1, spec.n, d)
        section_R[0, l * d:(l + 1) * d, torch.arange(d)] = 1.0
        msg_to_atom = (torch.arange(spec.num_codewords) // (J ** l)) % J
        components.append(ComponentSpec(Q=1, d=d, V=J, N=J,
                                          R_init="explicit", C_init="random_gaussian",
                                          U_init="all_pairs", T_init="explicit",
                                          learn_R=False, learn_C=learn_C,
                                          explicit_R=section_R,
                                          explicit_msg_to_atom=msg_to_atom))
    return components


def component_constraints(component_specs: list[ComponentSpec],
                            constrain_C_unit_norm: bool) -> list[ComponentConstraints]:
    out: list[ComponentConstraints] = []
    for cs in component_specs:
        c = ComponentConstraints()
        if constrain_C_unit_norm and cs.learn_C:
            c.C = "unit_norm_columns"
        out.append(c)
    return out


# --- argparse -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train", "infer"], default="infer")
    p.add_argument("--preset", choices=["dense", "odma", "ccs"], default="odma")
    # URASpec
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--num-codewords", type=int, default=32)
    p.add_argument("--num-active", type=int, default=4)
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--ebn0-db", type=float, default=5.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    # preset-specific
    p.add_argument("--num-blocks", type=int, default=4, help="odma preset only")
    p.add_argument("--d", type=int, default=None, help="local codeword length for odma preset")
    p.add_argument("--num-sections", type=int, default=2, help="ccs preset only")
    # learnability
    p.add_argument("--learn-C", action="store_true")
    p.add_argument("--learn-R", action="store_true")
    p.add_argument("--constrain-C-unit-norm", action="store_true", default=True)
    p.add_argument("--no-constrain-C-unit-norm", dest="constrain_C_unit_norm", action="store_false")
    # data / channel
    p.add_argument("--fading", choices=["constant", "iid_gaussian"], default="constant")
    p.add_argument("--data-source", choices=["uniform"], default="uniform")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dataset-size", type=int, default=1024,
                   help="if set, overrides --batches-per-epoch as ceil(dataset_size/batch_size)")
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--dataset-path", type=str, default=None,
                   help="optional .pt count dataset to load instead of generating uniform data")
    p.add_argument("--save-dataset", action="store_true",
                   help="save the generated/loaded count dataset into --out-dir")
    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batches-per-epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-min-factor", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lambda-power", type=float, default=0.0)
    p.add_argument("--lambda-coherence", type=float, default=0.0)
    p.add_argument("--lambda-row-load", type=float, default=0.0)
    p.add_argument("--eval-batches", type=int, default=8)
    # eval / inference
    p.add_argument("--num-trials", type=int, default=16,
                   help="number of evaluation batches in --mode infer")
    p.add_argument("--decoder", default="oracle_k_omp")
    p.add_argument("--max-list-size", type=int, default=None,
                   help="optional URA list-size cap (e.g. K_a)")
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results/framework_run")
    return p.parse_args(argv)


# --- main -----------------------------------------------------------------


def build_for_args(args: argparse.Namespace, gen: torch.Generator) -> tuple:
    spec = URASpec(n=args.n, num_codewords=args.num_codewords,
                    num_active=args.num_active, num_antennas=args.num_antennas)
    if args.preset == "dense":
        component_specs = dense_component_specs(spec, args.learn_C)
    elif args.preset == "odma":
        d = args.d if args.d is not None else max(spec.n // args.num_blocks, 1)
        component_specs = odma_component_specs(spec, d, args.num_blocks, args.learn_C, args.learn_R)
    elif args.preset == "ccs":
        component_specs = ccs_component_specs(spec, args.num_sections, args.learn_C)
    else:
        raise ValueError(f"unknown preset '{args.preset}'")
    constraints = component_constraints(component_specs, args.constrain_C_unit_norm)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    encoder = build_encoder(spec, component_specs, constraints=constraints,
                              dtype=dtype, generator=gen)
    if args.dataset_path is None:
        dataset_cfg = DatasetConfig(
            num_samples=int(args.dataset_size),
            num_active=spec.num_active,
            num_codewords=spec.num_codewords,
            train_fraction=float(args.train_fraction),
            val_fraction=float(args.val_fraction),
        )
        dataset = generate_uniform_count_dataset(dataset_cfg, gen, encoder.device, dtype)
    else:
        dataset = load_count_dataset(args.dataset_path, encoder.device, dtype)
    train_sampler = make_dataset_sampler(dataset, "train", args.batch_size, True, gen)
    val_sampler = make_dataset_sampler(dataset, "val", args.batch_size, False, gen)
    test_sampler = make_dataset_sampler(dataset, "test", args.batch_size, False, gen)
    if args.fading == "constant":
        fading_sampler = constant_fading(spec.num_antennas, dtype, encoder.device)
    else:
        fading_sampler = iid_gaussian_fading(spec.num_antennas, dtype, encoder.device, gen)
    return spec, encoder, dataset, train_sampler, val_sampler, test_sampler, fading_sampler


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(int(args.seed))
    gen = torch.Generator().manual_seed(int(args.seed))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    spec, encoder, dataset, train_sampler, val_sampler, test_sampler, fading_sampler = build_for_args(args, gen)
    print(f"[framework] preset={args.preset}  n={spec.n}  M={spec.num_codewords}  "
          f"K_a={spec.num_active}  M_ant={spec.num_antennas}  Eb/N0={args.ebn0_db} dB")
    Phi_summary = {
        "n_codewords": spec.num_codewords,
        "n_resources": spec.n,
        "num_learnable_params": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        "components": [
            {"Q": int(c.R.shape[0]), "d": int(c.R.shape[2]), "V": int(c.C.shape[1]),
             "N": int(c.atom_q.numel()), "learn_R": isinstance(c.R, torch.nn.Parameter),
             "learn_C": isinstance(c.C, torch.nn.Parameter)}
            for c in encoder.components],
    }

    if args.mode == "train":
        train_size = int(dataset.split()["train"].counts.shape[0])
        batches_per_epoch = max(math.ceil(train_size / args.batch_size), 1)
        cfg = TrainConfig(epochs=args.epochs, batches_per_epoch=batches_per_epoch,
                            batch_size=args.batch_size, lr=args.lr,
                            weight_decay=args.weight_decay,
                            lr_min_factor=args.lr_min_factor,
                            grad_clip=args.grad_clip,
                            lambda_power=args.lambda_power,
                            lambda_coherence=args.lambda_coherence,
                            lambda_row_load=args.lambda_row_load,
                            eval_batches=args.eval_batches,
                            eval_max_list_size=args.max_list_size)
        train(encoder, counts_sampler=train_sampler, validation_counts_sampler=val_sampler,
              fading_sampler=fading_sampler,
              ebn0_db=args.ebn0_db, cfg=cfg, generator=gen)
        plot_training_curves(cfg.progress, out_dir / "training_curves.png")
        (out_dir / "training_progress.json").write_text(json.dumps(cfg.progress, indent=2))

    # always run inference at the end so results files are written either way
    decoder = get_decoder(args.decoder)
    summary = evaluate(encoder, counts_sampler=test_sampler,
                         fading_sampler=fading_sampler, ebn0_db=args.ebn0_db,
                         num_batches=args.num_trials, batch_size=args.batch_size,
                         decoder=decoder, max_list_size=args.max_list_size,
                         generator=gen)
    print("[framework] eval:", "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))

    # one extra batch saved as a count comparison plot for diagnostic purposes
    batch = sample_batch(encoder, 1, test_sampler, fading_sampler, args.ebn0_db, gen)
    with torch.no_grad():
        out = decoder(encoder, batch.Y, batch.H, num_active=spec.num_active)
    plot_count_estimate(batch.counts[0].cpu(), out.counts[0].cpu(),
                         out_dir / "count_estimate.png")
    if args.save_dataset:
        dataset.save(out_dir / "count_dataset.pt")
    (out_dir / "summary.json").write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "spec": asdict(spec),
        "dataset": asdict(dataset.config),
        "phi_summary": Phi_summary,
        "eval_metrics": summary,
    }, indent=2, default=str))
    analyze_encoder(encoder, out_dir / "encoding_analysis")
    print(f"[framework] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
