"""Train a learned global decoder with the framework encoder frozen."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.channel import constant_fading, sample_batch, uniform_counts_generator  # noqa: E402
from framework.core import URASpec  # noqa: E402
from framework.decoders import oracle_k_omp  # noqa: E402
from framework.encoder import build_encoder  # noqa: E402
from framework.learned_decoders import UnrolledNonnegativeISTA, matched_filter_decoder  # noqa: E402
from framework.metrics import aggregate_metrics, batch_evaluate  # noqa: E402
from framework.pipeline import dense_component_specs, odma_component_specs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["dense", "odma"], default="dense")
    p.add_argument("-B", "--payload-bits", type=int, default=6)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--num-active", type=int, default=5)
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--ebn0-db", type=float, default=4.0)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batches-per-epoch", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda-support", type=float, default=0.1)
    p.add_argument("--lambda-sum", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--out-dir", default="results/framework_unrolled_decoder_test")
    return p.parse_args(argv)


def build_frozen_encoder(args: argparse.Namespace, gen: torch.Generator):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    M = 1 << int(args.payload_bits)
    spec = URASpec(n=int(args.n), num_codewords=M, num_active=int(args.num_active),
                   num_antennas=int(args.num_antennas), payload_bits=int(args.payload_bits))
    if args.preset == "dense":
        components = dense_component_specs(spec, learn_C=False)
    else:
        components = odma_component_specs(spec, int(args.d), int(args.num_blocks), learn_C=False, learn_R=False)
    encoder = build_encoder(spec, components, dtype=dtype, generator=gen)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def decoder_loss(out, counts_true: torch.Tensor, num_active: int,
                 lambda_support: float, lambda_sum: float) -> tuple[torch.Tensor, dict]:
    soft = out.meta["soft_counts"].to(dtype=counts_true.dtype, device=counts_true.device)
    logits = out.meta["support_logits"].to(dtype=counts_true.dtype, device=counts_true.device)
    count_loss = F.mse_loss(soft, counts_true)
    target = (counts_true > 0).to(counts_true.dtype)
    pos_weight = torch.as_tensor((counts_true.shape[1] - num_active) / max(num_active, 1),
                                 dtype=counts_true.dtype, device=counts_true.device)
    support_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    sum_loss = (((soft.sum(dim=1) - float(num_active)) / max(float(num_active), 1.0)) ** 2).mean()
    loss = count_loss + float(lambda_support) * support_loss + float(lambda_sum) * sum_loss
    return loss, {"loss": float(loss.detach()), "loss_count": float(count_loss.detach()),
                  "loss_support": float(support_loss.detach()), "loss_sum": float(sum_loss.detach())}


def evaluate_decoder(encoder, decoder_fn, counts_sampler, fading_sampler, ebn0_db: float,
                     num_batches: int, batch_size: int, gen: torch.Generator) -> dict:
    per = []
    with torch.no_grad():
        for _ in range(num_batches):
            batch = sample_batch(encoder, batch_size, counts_sampler, fading_sampler, ebn0_db, gen)
            out = decoder_fn(encoder, batch.Y, batch.H, encoder.spec.num_active)
            counts_est = out.counts.to(dtype=batch.counts.dtype, device=batch.counts.device)
            rows, _ = batch_evaluate(batch.counts, counts_est, max_list_size=encoder.spec.num_active)
            per.extend(rows)
    return aggregate_metrics(per)


def plot_progress(progress: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in progress]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["learned_l1_acc"] for r in progress], marker="o", label="learned ISTA")
    axes[0].axhline(progress[0]["matched_l1_acc"], color="gray", ls="--", label="matched filter")
    axes[0].axhline(progress[0]["nnomp_l1_acc"], color="black", ls=":", label="NNOMP oracle-K")
    axes[0].set_ylabel("L1 accuracy")
    axes[1].plot(epochs, [r["learned_pupe"] for r in progress], marker="o", label="learned ISTA")
    axes[1].axhline(progress[0]["matched_pupe"], color="gray", ls="--", label="matched filter")
    axes[1].axhline(progress[0]["nnomp_pupe"], color="black", ls=":", label="NNOMP oracle-K")
    axes[1].set_ylabel("PUPE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_antennas < 2:
        raise SystemExit("--num-antennas must be >= 2")
    if args.preset == "odma" and (args.d <= 0 or args.d > args.n):
        raise SystemExit(f"invalid ODMA geometry: d={args.d}, n={args.n}")
    torch.manual_seed(int(args.seed))
    gen = torch.Generator().manual_seed(int(args.seed))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_frozen_encoder(args, gen)
    counts_sampler = uniform_counts_generator(encoder.spec.num_active, encoder.spec.num_codewords, gen, encoder.device)
    fading_sampler = constant_fading(encoder.spec.num_antennas, encoder.dtype, encoder.device)
    model = UnrolledNonnegativeISTA(num_layers=int(args.num_layers)).to(device=encoder.device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    def learned_fn(enc, Y, H, K): return model(enc, Y, H, K)
    def nnomp_fn(enc, Y, H, K): return oracle_k_omp(enc, Y, H, K)

    matched = evaluate_decoder(encoder, matched_filter_decoder, counts_sampler, fading_sampler,
                               args.ebn0_db, args.eval_batches, args.batch_size, gen)
    nnomp = evaluate_decoder(encoder, nnomp_fn, counts_sampler, fading_sampler,
                             args.ebn0_db, args.eval_batches, args.batch_size, gen)

    print(f"Frozen {args.preset} encoder: n={encoder.n}, M={encoder.num_codewords}, "
          f"K={encoder.spec.num_active}, Eb/N0={args.ebn0_db:g} dB")
    print(f"matched filter: L1={matched.get('l1_acc', float('nan')):.4f} PUPE={matched.get('pupe', float('nan')):.4f}")
    print(f"NNOMP oracleK : L1={nnomp.get('l1_acc', float('nan')):.4f} PUPE={nnomp.get('pupe', float('nan')):.4f}")

    progress = []
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        parts_sum: dict[str, float] = {}
        model.train()
        for _ in range(int(args.batches_per_epoch)):
            batch = sample_batch(encoder, int(args.batch_size), counts_sampler, fading_sampler, float(args.ebn0_db), gen)
            out = model(encoder, batch.Y, batch.H, encoder.spec.num_active)
            loss, parts = decoder_loss(out, batch.counts, encoder.spec.num_active, args.lambda_support, args.lambda_sum)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        learned = evaluate_decoder(encoder, learned_fn, counts_sampler, fading_sampler,
                                   args.ebn0_db, args.eval_batches, args.batch_size, gen)
        rec = {"epoch": epoch, **{k: v / args.batches_per_epoch for k, v in parts_sum.items()},
               "learned_l1_acc": learned.get("l1_acc", float("nan")),
               "learned_pupe": learned.get("pupe", float("nan")),
               "matched_l1_acc": matched.get("l1_acc", float("nan")),
               "matched_pupe": matched.get("pupe", float("nan")),
               "nnomp_l1_acc": nnomp.get("l1_acc", float("nan")),
               "nnomp_pupe": nnomp.get("pupe", float("nan"))}
        progress.append(rec)
        print(f"epoch={epoch:<3d} loss={rec['loss']:.4f} "
              f"learned L1={rec['learned_l1_acc']:.4f} PUPE={rec['learned_pupe']:.4f}", flush=True)

    payload = {"args": vars(args), "matched_filter": matched, "nnomp_oracle_k": nnomp,
               "progress": progress, "wall_s": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_progress(progress, out_dir / "training_progress.png")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'training_progress.png'}")


if __name__ == "__main__":
    main()
