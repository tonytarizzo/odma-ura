"""Decoder-free geometry optimisation for framework codebooks.

This script optimises framework encoder factors directly against global
codebook geometry objectives. No channel samples or decoder are used. The first
intended use is conservative: keep the message plumbing fixed, keep ODMA
placements fixed, learn only the local codebook C, and ask whether the induced
global Phi can move toward recovery-friendly geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.analysis import analyze_encoder  # noqa: E402
from framework.core import URASpec  # noqa: E402
from framework.encoder import ComponentConstraints, build_encoder  # noqa: E402
from framework.pipeline import dense_component_specs, odma_component_specs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["dense", "odma"], default="odma")
    p.add_argument("--objective", choices=["amp", "vamp", "support_margin", "mixed"], default="amp")
    p.add_argument("-B", "--payload-bits", type=int, default=8)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--active-k", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-supports", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--learn-C", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--learn-R-values", action="store_true",
                   help="learn values on the initial R support mask; off-mask entries are projected back to zero")
    p.add_argument("--lambda-r-l1", type=float, default=0.0,
                   help="optional sparsity pressure on learnable R values; default is off")
    p.add_argument("--lambda-column-energy", type=float, default=0.0,
                   help="optional global codeword-energy penalty; usually unnecessary with fixed placements and unit C")
    p.add_argument("--mixed-amp-weight", type=float, default=1.0)
    p.add_argument("--mixed-vamp-weight", type=float, default=1.0)
    p.add_argument("--mixed-margin-weight", type=float, default=1.0)
    p.add_argument("--margin-target", type=float, default=0.0)
    p.add_argument("--eval-supports", type=int, default=256)
    p.add_argument("--analysis-supports", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--out-dir", default="results/framework_geometry_optimisation")
    return p.parse_args(argv)


def unit_columns(Phi: torch.Tensor) -> torch.Tensor:
    return Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-12)


def sample_supports(M: int, K: int, num_samples: int, device: torch.device,
                    generator: torch.Generator | None) -> list[torch.Tensor]:
    K = min(int(K), int(M))
    return [torch.randperm(M, device=device, generator=generator)[:K] for _ in range(int(num_samples))]


def active_gram_loss(Phi: torch.Tensor, active_k: int, num_samples: int,
                     generator: torch.Generator | None) -> torch.Tensor:
    P = unit_columns(Phi)
    losses = []
    for idx in sample_supports(P.shape[1], active_k, num_samples, P.device, generator):
        A = P.index_select(1, idx)
        G = A.conj().T @ A if A.is_complex() else A.T @ A
        I = torch.eye(G.shape[0], dtype=G.real.dtype if G.is_complex() else G.dtype, device=G.device)
        losses.append(((G.real - I) ** 2).mean())
    return torch.stack(losses).mean()


def vamp_spectral_loss(Phi: torch.Tensor) -> torch.Tensor:
    P = unit_columns(Phi)
    cov = P @ P.conj().T if P.is_complex() else P @ P.T
    cov = cov.real
    target = float(P.shape[1]) / float(P.shape[0])
    I = torch.eye(P.shape[0], dtype=cov.dtype, device=cov.device)
    return (((cov - target * I) / max(target, 1e-12)) ** 2).mean()


def support_margin_loss(Phi: torch.Tensor, active_k: int, num_samples: int,
                        margin_target: float, generator: torch.Generator | None) -> torch.Tensor:
    P = unit_columns(Phi)
    M = P.shape[1]
    if active_k >= M:
        raise ValueError("support_margin requires active_k < M")
    losses = []
    for idx in sample_supports(M, active_k, num_samples, P.device, generator):
        active = torch.zeros(M, dtype=torch.bool, device=P.device)
        active[idx] = True
        y = P.index_select(1, idx).sum(dim=1)
        scores = ((P.conj().T @ y) if P.is_complex() else (P.T @ y)).real
        min_true = scores[active].min()
        max_false = scores[~active].max()
        losses.append(torch.nn.functional.softplus(max_false - min_true + float(margin_target)))
    return torch.stack(losses).mean()


def column_energy_loss(Phi: torch.Tensor) -> torch.Tensor:
    energy = (Phi.conj() * Phi).sum(dim=0).real if Phi.is_complex() else (Phi ** 2).sum(dim=0)
    return ((energy - 1.0) ** 2).mean()


def r_l1_loss(encoder) -> torch.Tensor:
    vals = [c.R.abs().mean() for c in encoder.components if isinstance(c.R, torch.nn.Parameter)]
    if not vals:
        return torch.zeros((), dtype=encoder.dtype, device=encoder.device)
    return torch.stack(vals).mean()


def compute_losses(encoder, args: argparse.Namespace, generator: torch.Generator | None) -> tuple[torch.Tensor, dict[str, float]]:
    Phi = encoder.explicit_matrix()
    parts = {
        "amp": active_gram_loss(Phi, args.active_k, args.batch_supports, generator),
        "vamp": vamp_spectral_loss(Phi),
        "support_margin": support_margin_loss(Phi, args.active_k, args.batch_supports, args.margin_target, generator),
    }
    if args.objective == "mixed":
        loss = (float(args.mixed_amp_weight) * parts["amp"]
                + float(args.mixed_vamp_weight) * parts["vamp"]
                + float(args.mixed_margin_weight) * parts["support_margin"])
    else:
        loss = parts[args.objective]
    if args.lambda_r_l1 > 0.0:
        parts["r_l1"] = r_l1_loss(encoder)
        loss = loss + float(args.lambda_r_l1) * parts["r_l1"]
    if args.lambda_column_energy > 0.0:
        parts["column_energy"] = column_energy_loss(Phi)
        loss = loss + float(args.lambda_column_energy) * parts["column_energy"]
    return loss, {k: float(v.detach().cpu()) for k, v in parts.items()} | {"loss": float(loss.detach().cpu())}


def build_geometry_encoder(args: argparse.Namespace, gen: torch.Generator):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    M = 1 << int(args.payload_bits)
    spec = URASpec(n=int(args.n), num_codewords=M, num_active=int(args.active_k), num_antennas=2,
                   payload_bits=int(args.payload_bits))
    if args.preset == "dense":
        components = dense_component_specs(spec, learn_C=bool(args.learn_C))
        if args.learn_R_values:
            raise ValueError("dense preset uses identity R; use learn-C for dense geometry optimisation")
    else:
        components = odma_component_specs(spec, int(args.d), int(args.num_blocks),
                                          learn_C=bool(args.learn_C), learn_R=bool(args.learn_R_values))
    constraints = []
    for cs in components:
        c = ComponentConstraints()
        if cs.learn_C:
            c.C = "unit_norm_columns"
        constraints.append(c)
    encoder = build_encoder(spec, components, constraints=constraints, dtype=dtype, generator=gen)
    r_masks = [c.R.detach().abs() > 1e-12 for c in encoder.components]
    return encoder, r_masks


def project_after_step(encoder, r_masks: list[torch.Tensor]) -> None:
    encoder.apply_constraints()
    with torch.no_grad():
        for comp, mask in zip(encoder.components, r_masks):
            if isinstance(comp.R, torch.nn.Parameter):
                comp.R.mul_(mask.to(device=comp.R.device, dtype=comp.R.dtype))
                norms = comp.R.norm(dim=1, keepdim=True).clamp_min(1e-12)
                comp.R.div_(norms)


def evaluate_all_objectives(encoder, args: argparse.Namespace, num_supports: int, seed: int) -> dict[str, float]:
    old = args.batch_supports
    args.batch_supports = int(num_supports)
    gen = torch.Generator(device=encoder.device).manual_seed(int(seed))
    with torch.no_grad():
        _, parts = compute_losses(encoder, args, gen)
    args.batch_supports = old
    return parts


def plot_progress(progress: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("loss", "amp", "vamp", "support_margin", "r_l1", "column_energy") if k in progress[0]]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [r["step"] for r in progress]
    for key in keys:
        ax.plot(x, [r[key] for r in progress], label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("objective")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.payload_bits <= 0:
        raise SystemExit("--payload-bits must be positive")
    if args.active_k <= 0 or args.active_k >= (1 << int(args.payload_bits)):
        raise SystemExit("--active-k must lie in [1, M)")
    if args.preset == "odma" and (args.d <= 0 or args.d > args.n):
        raise SystemExit(f"invalid ODMA geometry: d={args.d}, n={args.n}")

    torch.manual_seed(int(args.seed))
    gen = torch.Generator().manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, r_masks = build_geometry_encoder(args, gen)
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    if num_params == 0:
        raise SystemExit("no learnable parameters; enable --learn-C or --learn-R-values")

    print(f"[geometry] preset={args.preset} objective={args.objective} n={args.n} M={encoder.num_codewords} "
          f"K={args.active_k} learnable_params={num_params}")

    analyze_encoder(encoder, out_dir / "encoding_analysis_before", active_k=int(args.active_k),
                    num_active_samples=int(args.analysis_supports))
    before = evaluate_all_objectives(encoder, args, int(args.eval_supports), int(args.seed) + 10_000)
    torch.save({"args": vars(args), "state_dict": encoder.state_dict()}, out_dir / "encoder_before.pt")

    opt = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad],
                           lr=float(args.lr), weight_decay=float(args.weight_decay))
    progress = []
    for step in range(1, int(args.steps) + 1):
        loss, parts = compute_losses(encoder, args, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(args.grad_clip))
        opt.step()
        project_after_step(encoder, r_masks)
        if step == 1 or step == args.steps or step % max(1, args.steps // 100) == 0:
            rec = {"step": int(step), **parts}
            progress.append(rec)
            print(" ".join([f"step={step:04d}", *(f"{k}={v:.4e}" for k, v in parts.items())]), flush=True)

    after = evaluate_all_objectives(encoder, args, int(args.eval_supports), int(args.seed) + 10_000)
    torch.save({"args": vars(args), "state_dict": encoder.state_dict()}, out_dir / "encoder_after.pt")
    analyze_encoder(encoder, out_dir / "encoding_analysis_after", active_k=int(args.active_k),
                    num_active_samples=int(args.analysis_supports))
    plot_progress(progress, out_dir / "geometry_optimisation_progress.png")

    payload = {
        "args": vars(args),
        "num_learnable_params": int(num_params),
        "before": before,
        "after": after,
        "progress": progress,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"[geometry] before: {before}")
    print(f"[geometry] after : {after}")
    print(f"[geometry] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
