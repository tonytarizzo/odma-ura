"""Paired before/after decoder evaluation for geometry-optimised encoders.

Given a run directory from ``tests.framework_geometry_optimisation``, this
script loads ``encoder_before.pt`` and ``encoder_after.pt`` and evaluates them
on the same sampled active-message counts and noise realisations. The point is
to test whether decoder-free geometry improvements translate into actual
oracle-K sparse-recovery performance.
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

from framework.channel import ebn0_db_to_noise_var  # noqa: E402
from framework.decoders import get_decoder  # noqa: E402
from framework.metrics import aggregate_metrics, batch_evaluate  # noqa: E402
from tests.framework_geometry_optimisation import build_geometry_encoder  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[-4, -2, 0, 2, 4, 6, 8])
    p.add_argument("--decoder", default="oracle_k_omp")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-name", default="decoder_eval_summary")
    return p.parse_args(argv)


def namespace_from_saved_args(saved: dict) -> argparse.Namespace:
    return argparse.Namespace(**saved)


def load_encoder(run_dir: Path, which: str):
    ckpt_path = run_dir / f"encoder_{which}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"missing {ckpt_path}; rerun tests.framework_geometry_optimisation with the current script first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    opt_args = namespace_from_saved_args(ckpt["args"])
    gen = torch.Generator().manual_seed(int(opt_args.seed))
    encoder, _ = build_geometry_encoder(opt_args, gen)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, opt_args


def sample_counts(batch_size: int, K: int, M: int, dtype: torch.dtype,
                  device: torch.device, generator: torch.Generator) -> torch.Tensor:
    active = torch.randint(M, (batch_size, K), generator=generator, device=device)
    counts = torch.zeros(batch_size, M, dtype=dtype, device=device)
    counts.scatter_add_(1, active.long(), torch.ones_like(active, dtype=dtype))
    return counts


def paired_batch(encoder, counts: torch.Tensor, ebn0_db: float,
                 base_noise: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = encoder.encode(counts.to(dtype=encoder.dtype, device=encoder.device))
    Y_clean = y.unsqueeze(-1) * H.unsqueeze(1)
    Phi = encoder.explicit_matrix().detach()
    energy = float(torch.mean(torch.sum(torch.abs(Phi) ** 2, dim=0)).cpu())
    noise_var = ebn0_db_to_noise_var(float(ebn0_db), encoder.spec.payload_bits, energy)
    Y = Y_clean + base_noise.to(dtype=encoder.dtype, device=encoder.device) * math.sqrt(noise_var)
    return Y, H


def evaluate_point(before, after, decoder_fn, K: int, ebn0_db: float,
                   num_samples: int, batch_size: int, seed: int) -> dict:
    if before.num_codewords != after.num_codewords or before.n != after.n:
        raise ValueError("before/after encoders have different (n, M)")
    if before.dtype != after.dtype:
        raise ValueError("before/after encoders have different dtypes")
    if before.spec.num_antennas != after.spec.num_antennas:
        raise ValueError("before/after encoders have different antenna counts")

    gen = torch.Generator(device=before.device).manual_seed(int(seed))
    rows = {"before": [], "after": []}
    walls = {"before": [], "after": []}
    done = 0
    while done < int(num_samples):
        bsz = min(int(batch_size), int(num_samples) - done)
        counts = sample_counts(bsz, K, before.num_codewords, before.dtype, before.device, gen)
        H = torch.ones(bsz, before.spec.num_antennas, dtype=before.dtype, device=before.device)
        base_noise = torch.randn(bsz, before.n, before.spec.num_antennas,
                                 dtype=before.dtype, device=before.device, generator=gen)
        for label, encoder in (("before", before), ("after", after)):
            Y, H_pair = paired_batch(encoder, counts, ebn0_db, base_noise, H)
            t0 = time.time()
            out = decoder_fn(encoder, Y, H_pair, num_active=K)
            walls[label].append(time.time() - t0)
            per, _ = batch_evaluate(counts, out.counts.to(dtype=counts.dtype, device=counts.device), max_list_size=K)
            rows[label].extend(per)
        done += bsz

    out = {"ebn0_db": float(ebn0_db), "num_samples": int(num_samples), "K": int(K)}
    for label in ("before", "after"):
        metrics = aggregate_metrics(rows[label])
        metrics["wall_s_total"] = float(sum(walls[label]))
        metrics["wall_s_per_sample"] = float(sum(walls[label]) / max(num_samples, 1))
        out[label] = metrics
    out["delta_after_minus_before"] = {
        k: float(out["after"][k] - out["before"][k])
        for k in out["before"]
        if isinstance(out["before"][k], (int, float)) and isinstance(out["after"].get(k), (int, float))
    }
    return out


def plot_curves(points: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted(points, key=lambda p: p["ebn0_db"])
    x = [p["ebn0_db"] for p in points]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, ylabel in [
        (axes[0], "pupe", "PUPE"),
        (axes[1], "l1_acc", "L1 accuracy"),
        (axes[2], "false_alarm_rate", "False alarm rate"),
    ]:
        ax.plot(x, [p["before"].get(key, float("nan")) for p in points], marker="o", label="before")
        ax.plot(x, [p["after"].get(key, float("nan")) for p in points], marker="o", label="after")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    run_dir = Path(args.run_dir)
    before, opt_args = load_encoder(run_dir, "before")
    after, opt_args_after = load_encoder(run_dir, "after")
    if vars(opt_args) != vars(opt_args_after):
        raise ValueError("before/after checkpoints have different saved args")

    decoder_fn = get_decoder(args.decoder)
    K = int(opt_args.active_k)
    print(f"[decoder-eval] run={run_dir} decoder={args.decoder} n={before.n} M={before.num_codewords} "
          f"K={K} samples={args.num_samples}")

    points = []
    for i, ebn0_db in enumerate(args.ebn0_grid):
        point = evaluate_point(before, after, decoder_fn, K, float(ebn0_db),
                               int(args.num_samples), int(args.batch_size), int(args.seed) + 1009 * i)
        points.append(point)
        b = point["before"]; a = point["after"]; d = point["delta_after_minus_before"]
        print(f"Eb/N0={float(ebn0_db):>6.2f}  "
              f"PUPE {b['pupe']:.4f}->{a['pupe']:.4f} ({d['pupe']:+.4f})  "
              f"L1 {b['l1_acc']:.4f}->{a['l1_acc']:.4f} ({d['l1_acc']:+.4f})", flush=True)

    payload = {
        "eval_args": vars(args),
        "optimisation_args": vars(opt_args),
        "points": points,
    }
    json_path = run_dir / f"{args.out_name}.json"
    png_path = run_dir / f"{args.out_name}.png"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    plot_curves(points, png_path)
    print(f"[decoder-eval] wrote {json_path}")
    print(f"[decoder-eval] wrote {png_path}")


if __name__ == "__main__":
    main()
