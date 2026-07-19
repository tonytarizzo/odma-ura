r"""Training and inference loops for the URA framework.

Design choice: the *evaluation* decoder (e.g. oracle-K NNOMP) is generally not
differentiable. For codebook training we therefore drive gradients through a
*surrogate* decoder that is differentiable but uses only encoder-supplied
operations. The simplest such surrogate is a single matched-filter pass,
    \hat a_surr = Phi^H Phi a_true,
which gives gradients into R and C and remains exact for orthonormal Phi. More
realistic surrogates (unrolled NNOMP, learned decoders) can be plugged in by
swapping `surrogate_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .channel import matched_filter_collapse, sample_batch
from .core import DecoderOutput, URABatch
from .decoders import oracle_k_omp
from .encoder import Encoder
from .losses import (coherence_penalty, count_mse_loss, power_penalty,
                     row_load_penalty)
from .metrics import aggregate_metrics, batch_evaluate


SurrogateFn = Callable[[Encoder, torch.Tensor, torch.Tensor], torch.Tensor]
DecoderFn = Callable[..., DecoderOutput]


def matched_filter_surrogate(encoder: Encoder, Y: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Phi^H y_mf, the simplest differentiable count proxy."""
    y_mf = matched_filter_collapse(Y, H)
    return encoder.rmatvec(y_mf)


@dataclass
class TrainConfig:
    epochs: int = 5
    batches_per_epoch: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_min_factor: float = 0.1            # cosine schedule end factor
    power_target: float = 1.0
    lambda_power: float = 0.0
    lambda_coherence: float = 0.0
    lambda_row_load: float = 0.0
    grad_clip: float | None = None
    log_every: int = 1
    eval_batches: int = 8
    eval_max_list_size: int | None = None
    surrogate: str = "matched_filter"
    progress: list[dict] = field(default_factory=list)


SURROGATES: dict[str, SurrogateFn] = {"matched_filter": matched_filter_surrogate}


def make_optimiser(encoder: Encoder, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in encoder.parameters() if p.requires_grad]
    if not params:
        raise ValueError("encoder has no learnable parameters; nothing to train")
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def total_loss(encoder: Encoder, batch: URABatch, surrogate: SurrogateFn,
                cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    pred = surrogate(encoder, batch.Y, batch.H)
    loss_dec = count_mse_loss(pred, batch.counts)
    parts = {"loss_dec": float(loss_dec.detach().cpu())}
    loss = loss_dec
    if cfg.lambda_power > 0.0:
        lp = power_penalty(encoder, cfg.power_target)
        loss = loss + cfg.lambda_power * lp
        parts["loss_pow"] = float(lp.detach().cpu())
    if cfg.lambda_coherence > 0.0:
        lc = coherence_penalty(encoder)
        loss = loss + cfg.lambda_coherence * lc
        parts["loss_coh"] = float(lc.detach().cpu())
    if cfg.lambda_row_load > 0.0:
        lr_ = row_load_penalty(encoder)
        loss = loss + cfg.lambda_row_load * lr_
        parts["loss_row"] = float(lr_.detach().cpu())
    parts["loss"] = float(loss.detach().cpu())
    return loss, parts


def train(encoder: Encoder, counts_sampler, validation_counts_sampler, fading_sampler,
           ebn0_db: float, cfg: TrainConfig,
           generator: torch.Generator | None = None) -> TrainConfig:
    """Train `encoder` in place. The same `cfg` object is returned with the
    `progress` list filled in for the caller's bookkeeping."""
    surrogate = SURROGATES[cfg.surrogate]
    opt = make_optimiser(encoder, cfg)
    total_steps = max(cfg.epochs * cfg.batches_per_epoch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=cfg.lr * cfg.lr_min_factor)

    step = 0
    for epoch in range(cfg.epochs):
        epoch_parts: dict[str, float] = {}
        for _ in range(cfg.batches_per_epoch):
            batch = sample_batch(encoder, cfg.batch_size, counts_sampler, fading_sampler, ebn0_db, generator)
            loss, parts = total_loss(encoder, batch, surrogate, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
            opt.step()
            encoder.apply_constraints()
            sched.step()
            step += 1
            for k, v in parts.items():
                epoch_parts[k] = epoch_parts.get(k, 0.0) + v
        avg = {k: v / cfg.batches_per_epoch for k, v in epoch_parts.items()}
        eval_summary = evaluate(encoder, counts_sampler=validation_counts_sampler,
                                  fading_sampler=fading_sampler,
                                  ebn0_db=ebn0_db, num_batches=cfg.eval_batches,
                                  batch_size=cfg.batch_size,
                                  max_list_size=cfg.eval_max_list_size,
                                  generator=generator)
        record = {"epoch": epoch + 1, "lr": float(opt.param_groups[0]["lr"]),
                   **avg, **{f"eval_{k}": v for k, v in eval_summary.items()}}
        cfg.progress.append(record)
        if cfg.log_every and (epoch + 1) % cfg.log_every == 0:
            keys = ["loss", "loss_dec", "eval_pupe", "eval_f1", "eval_l1_err"]
            shown = "  ".join(f"{k}={record.get(k, float('nan')):.4f}" for k in keys)
            print(f"epoch {epoch + 1:3d}  lr={record['lr']:.2e}  {shown}")
    return cfg


# --- inference / evaluation ----------------------------------------------


def evaluate(encoder: Encoder, counts_sampler, fading_sampler, ebn0_db: float,
              num_batches: int, batch_size: int,
              decoder: DecoderFn = oracle_k_omp,
              max_list_size: int | None = None,
              generator: torch.Generator | None = None) -> dict:
    """Run the configured decoder over fresh batches and aggregate metrics."""
    if num_batches <= 0:
        return {}
    all_per_sample: list[dict] = []
    encoder.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            batch = sample_batch(encoder, batch_size, counts_sampler, fading_sampler, ebn0_db, generator)
            out = decoder(encoder, batch.Y, batch.H, num_active=batch.num_active, noise_var=batch.noise_var)
            counts_est = out.counts.to(dtype=batch.counts.dtype, device=batch.counts.device)
            per, _ = batch_evaluate(batch.counts, counts_est, max_list_size=max_list_size)
            all_per_sample.extend(per)
    encoder.train()
    return aggregate_metrics(all_per_sample)
