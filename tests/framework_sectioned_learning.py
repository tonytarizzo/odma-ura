"""Train and evaluate the scalable L>1 encoder, structured D0, outer BP, and valid-path list decoder.

The runner never allocates a tensor indexed by all ``M=2^B`` messages.  The
``smoke`` preset is a fast wiring check; ``laptop`` is a deterministic local
learning/performance test intended to finish in a few minutes on a CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.channel import constant_fading  # noqa: E402
from framework.core import ComponentSpec, OuterBPOutput, SectionedURASpec  # noqa: E402
from framework.encoder import ComponentConstraints  # noqa: E402
from framework.outer_code import random_sparse_outer_code, triadic_outer_code  # noqa: E402
from framework.outer_decoder import (SectionedOuterDecoder, ValidPathListDecoder, path_list_pupe,
                                     sectioned_outer_training_loss)  # noqa: E402
from framework.sectioned import (build_orthogonal_sectioned_encoder, build_sectioned_encoder,
                                 outer_code_path_generator, sample_sectioned_batch, sampled_energy_report)  # noqa: E402


PRESETS = {
    "smoke": {"payload_bits": 8, "section_bits": 2, "n": 64, "num_active": 2, "steps": 8,
              "batch_size": 4, "eval_batches": 2, "eval_ebn0": [4.0, 8.0], "d0_layers": 2,
              "bp_iterations": 2, "power_iters": 3, "num_path_negatives": 4, "log_every": 4,
              "beam_width": 64, "list_extra": 4, "learn_encoder": True, "assert_reasonable": False,
              "outer_warmup_steps": 0, "outer_code": "triadic", "energy_mode": "orthogonal_exact",
              "bank_type": "explicit", "mixing_stages": 4},
    "scale_smoke": {"payload_bits": 128, "section_bits": 16, "n": 256, "num_active": 2, "steps": 1,
                    "batch_size": 1, "eval_batches": 1, "eval_ebn0": [8.0], "d0_layers": 1,
                    "bp_iterations": 1, "power_iters": 2, "num_path_negatives": 2, "log_every": 1,
                    "beam_width": 64, "list_extra": 4, "candidate_cap": 32, "learn_encoder": False,
                    "assert_reasonable": False, "outer_warmup_steps": 0, "outer_code": "triadic",
                    "energy_mode": "orthogonal_exact", "bank_type": "subsampled_hadamard", "mixing_stages": 2},
    "laptop": {"payload_bits": 16, "section_bits": 4, "n": 512, "num_active": 2, "steps": 400,
               "outer_warmup_steps": 250, "batch_size": 32, "eval_batches": 12,
               "eval_ebn0": [6.0, 10.0, 14.0], "d0_layers": 6, "bp_iterations": 3, "power_iters": 8,
               "num_path_negatives": 16, "log_every": 25, "beam_width": 512, "list_extra": 16,
               "learn_encoder": True, "assert_reasonable": True, "outer_code": "triadic",
               "energy_mode": "orthogonal_exact", "bank_type": "explicit", "mixing_stages": 4},
    "custom": {},
}


def float_grid(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), default="laptop")
    p.add_argument("-B", "--payload-bits", type=int, default=None)
    p.add_argument("-J", "--section-bits", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--num-active", type=int, default=None)
    p.add_argument("--outer-code", choices=["triadic", "random_sparse"], default=None)
    p.add_argument("--num-parity-sections", type=int, default=None)
    p.add_argument("--check-degree", type=int, default=2)
    p.add_argument("--random-coefficients", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--energy-mode", choices=["orthogonal_exact", "overlapping_sampled"], default=None)
    p.add_argument("--bank-type", choices=["explicit", "subsampled_hadamard"], default=None)
    p.add_argument("--learn-encoder", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mixing-stages", type=int, default=None)
    p.add_argument("--d0-layers", type=int, default=None)
    p.add_argument("--bp-iterations", type=int, default=None)
    p.add_argument("--power-iters", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--outer-warmup-steps", type=int, default=None,
                   help="train D0 (and the optional encoder) alone first, then freeze the encoder and ramp outer losses")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--train-ebn0-min", type=float, default=2.0)
    p.add_argument("--train-ebn0-max", type=float, default=14.0)
    p.add_argument("--eval-ebn0", type=float_grid, default=None)
    p.add_argument("--eval-batches", type=int, default=None)
    p.add_argument("--lambda-d0-count", type=float, default=0.1)
    p.add_argument("--lambda-marginal", type=float, default=0.1)
    p.add_argument("--lambda-path", type=float, default=0.02)
    p.add_argument("--lambda-power", type=float, default=1.0)
    p.add_argument("--num-path-negatives", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument("--list-extra", type=int, default=None)
    p.add_argument("--candidate-cap", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--assert-reasonable", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--seed", type=int, default=7301)
    p.add_argument("--out-dir", default="results/framework_sectioned_learning")
    args = p.parse_args(argv)
    values = PRESETS[args.preset]
    for name, value in values.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    required = ["payload_bits", "section_bits", "n", "num_active", "steps", "batch_size", "eval_batches",
                "eval_ebn0", "d0_layers", "bp_iterations", "power_iters", "num_path_negatives", "log_every",
                "beam_width", "list_extra", "learn_encoder", "assert_reasonable", "outer_warmup_steps",
                "outer_code", "energy_mode", "bank_type", "mixing_stages"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        p.error(f"custom preset requires {', '.join('--' + name.replace('_', '-') for name in missing)}")
    return args


def build_system(args: argparse.Namespace, generator: torch.Generator):
    del generator
    outer_generator = torch.Generator().manual_seed(args.seed + 1001)
    encoder_generator = torch.Generator().manual_seed(args.seed + 2001)
    if args.payload_bits <= 0 or args.section_bits <= 0 or args.payload_bits % args.section_bits:
        raise ValueError("B and J must be positive, with J dividing B")
    if args.outer_code == "triadic":
        code = triadic_outer_code(args.payload_bits, args.section_bits)
    else:
        num_information = args.payload_bits // args.section_bits
        num_parity = num_information if args.num_parity_sections is None else int(args.num_parity_sections)
        code = random_sparse_outer_code(args.payload_bits, args.section_bits, num_parity, args.check_degree,
                                        outer_generator, args.random_coefficients)
    spec = SectionedURASpec(n=args.n, payload_bits=args.payload_bits, num_active=args.num_active)
    if args.energy_mode == "orthogonal_exact":
        encoder = build_orthogonal_sectioned_encoder(spec, code, learn_C=args.learn_encoder,
                                                     mixing_stages=args.mixing_stages, bank_type=args.bank_type,
                                                     dtype=torch.float32, generator=encoder_generator)
    else:
        if args.bank_type != "explicit":
            raise ValueError("overlapping_sampled currently requires explicit local banks")
        q = 1 << args.section_bits
        components = [ComponentSpec(Q=1, d=args.n, V=q, N=q, R_init="identity", C_init="random_gaussian",
                                    U_init="all_pairs", learn_C=args.learn_encoder) for _ in range(code.num_sections)]
        constraints = [ComponentConstraints(C="unit_norm_columns" if args.learn_encoder else "none") for _ in components]
        energies = [1.0 / code.num_sections] * code.num_sections
        encoder = build_sectioned_encoder(spec, components, constraints, energies, torch.float32, encoder_generator)
    decoder = SectionedOuterDecoder(args.d0_layers, args.bp_iterations, args.power_iters)
    return encoder, code, decoder


def random_ebn0(args: argparse.Namespace, generator: torch.Generator) -> float:
    u = float(torch.rand((), generator=generator))
    return args.train_ebn0_min + u * (args.train_ebn0_max - args.train_ebn0_min)


def loss_for_batch(args: argparse.Namespace, encoder, code, decoder, batch, generator: torch.Generator,
                   outer_scale: float = 1.0):
    output = decoder(encoder, code, batch.Y, batch.H, batch.num_active, batch.noise_var)
    lambda_power = args.lambda_power if args.energy_mode == "overlapping_sampled" else 0.0
    return sectioned_outer_training_loss(output, batch.section_counts, batch.active_paths, code, encoder,
                                         args.lambda_d0_count, outer_scale * args.lambda_marginal,
                                         outer_scale * args.lambda_path, lambda_power,
                                         args.num_path_negatives, generator), output


def evaluate(args: argparse.Namespace, encoder, code, decoder, seed: int) -> dict:
    rows = []
    decoder.eval(); encoder.eval()
    extractor = ValidPathListDecoder(args.beam_width, args.list_extra, args.candidate_cap)
    with torch.no_grad():
        for snr_index, ebn0_db in enumerate(args.eval_ebn0):
            generator = torch.Generator().manual_seed(seed + 1009 * snr_index)
            sampler = outer_code_path_generator(args.num_active, code, generator, encoder.device)
            fading = constant_fading(encoder.spec.num_antennas, encoder.dtype, encoder.device)
            sums: dict[str, float] = {}
            pupe_values, d0_pupe_values = [], []
            for _ in range(args.eval_batches):
                batch = sample_sectioned_batch(encoder, args.batch_size, sampler, fading, ebn0_db, generator)
                (loss, parts), output = loss_for_batch(args, encoder, code, decoder, batch, generator)
                listed = extractor.decode(output, code, output.meta["d0_output"].meta["soft_section_counts"], batch.num_active)
                d0_logits = torch.stack(output.meta["d0_output"].meta["section_support_logits"], dim=1)
                d0_beliefs = OuterBPOutput(log_beliefs=torch.log_softmax(d0_logits, dim=-1))
                d0_listed = extractor.decode(d0_beliefs, code, output.meta["d0_output"].meta["soft_section_counts"],
                                             batch.num_active)
                pupe = path_list_pupe(listed, batch.active_paths)
                pupe_values.append(pupe.cpu()); d0_pupe_values.append(path_list_pupe(d0_listed, batch.active_paths).cpu())
                for name, value in parts.items():
                    sums[name] = sums.get(name, 0.0) + float(value)
            pupe = torch.cat(pupe_values)
            d0_pupe = torch.cat(d0_pupe_values)
            row = {"ebn0_db": float(ebn0_db), "pupe": float(pupe.mean()),
                   "d0_without_bp_pupe": float(d0_pupe.mean()),
                   "pupe_standard_error": float(pupe.std(unbiased=False) / math.sqrt(pupe.numel())),
                   "exact_batch_recovery": float((pupe == 0).to(torch.float32).mean()),
                   **{name: value / args.eval_batches for name, value in sums.items()}}
            rows.append(row)
    encoder.train(); decoder.train()
    return {"rows": rows, "mean_pupe": sum(row["pupe"] for row in rows) / len(rows),
            "mean_loss": sum(row["total"] for row in rows) / len(rows)}


def train(args: argparse.Namespace, encoder, code, decoder, generator: torch.Generator) -> list[dict]:
    parameters = list(decoder.parameters())
    if args.learn_encoder:
        parameters += [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimiser = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.steps, 1), eta_min=args.lr * 0.1)
    sampler = outer_code_path_generator(args.num_active, code, generator, encoder.device)
    fading = constant_fading(encoder.spec.num_antennas, encoder.dtype, encoder.device)
    interval_sums: dict[str, float] = {}
    progress = []
    for step in range(1, args.steps + 1):
        if args.learn_encoder and args.outer_warmup_steps and step == args.outer_warmup_steps + 1:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
        if step <= args.outer_warmup_steps:
            outer_scale = 0.0
            phase = "d0_warmup"
        else:
            ramp_steps = max(args.steps - args.outer_warmup_steps, 1)
            outer_scale = min((step - args.outer_warmup_steps) / max(ramp_steps // 2, 1), 1.0)
            phase = "outer_ramp"
        batch = sample_sectioned_batch(encoder, args.batch_size, sampler, fading, random_ebn0(args, generator), generator)
        (loss, parts), _ = loss_for_batch(args, encoder, code, decoder, batch, generator, outer_scale)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimiser.step(); scheduler.step()
        if args.learn_encoder and (args.outer_warmup_steps == 0 or step <= args.outer_warmup_steps):
            encoder.apply_constraints()
        for name, value in parts.items():
            interval_sums[name] = interval_sums.get(name, 0.0) + float(value.detach())
        interval_sums["grad_norm"] = interval_sums.get("grad_norm", 0.0) + float(grad_norm)
        if step % args.log_every == 0 or step == args.steps:
            count = args.log_every if step % args.log_every == 0 else args.steps % args.log_every
            record = {"step": step, "phase": phase, "outer_scale": outer_scale,
                      "lr": float(optimiser.param_groups[0]["lr"]),
                      **{name: value / count for name, value in interval_sums.items()}}
            progress.append(record); interval_sums = {}
            print(f"step={step:4d} phase={phase:10s} loss={record['total']:.5f} d0={record['d0']:.5f} "
                  f"marginal={record['outer_marginal']:.5f} path={record['outer_path']:.5f}", flush=True)
    return progress


def plot_summary(progress: list[dict], before: dict, after: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot([row["step"] for row in progress], [row["total"] for row in progress], label="total")
    axes[0].plot([row["step"] for row in progress], [row["d0"] for row in progress], label="D0")
    axes[0].set(xlabel="training step", ylabel="loss", title="Scalable end-to-end learning")
    axes[0].grid(alpha=0.25); axes[0].legend()
    for label, result, marker in [("initial", before, "o"), ("trained", after, "s")]:
        axes[1].plot([row["ebn0_db"] for row in result["rows"]], [row["pupe"] for row in result["rows"]],
                     marker=marker, label=label)
    axes[1].set(xlabel="$E_b/N_0$ (dB)", ylabel="PUPE", title="Held-out complete-message recovery", ylim=(0.0, 1.0))
    axes[1].grid(alpha=0.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def validate_result(args: argparse.Namespace, encoder, before: dict, after: dict, progress: list[dict]) -> None:
    values = [before["mean_loss"], before["mean_pupe"], after["mean_loss"], after["mean_pupe"]]
    values += [value for row in progress for value in row.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("learning run produced a nonfinite loss or metric")
    if args.energy_mode == "orthogonal_exact" and not encoder.certify_exact_energy()["guaranteed"]:
        raise AssertionError("projected training violated the exact unit-energy codeword guarantee")
    if args.assert_reasonable:
        if after["mean_loss"] >= 0.95 * before["mean_loss"]:
            raise AssertionError(f"held-out loss did not improve enough: {before['mean_loss']:.4f} -> {after['mean_loss']:.4f}")
        if after["mean_pupe"] > before["mean_pupe"] + 0.02:
            raise AssertionError(f"held-out PUPE regressed: {before['mean_pupe']:.4f} -> {after['mean_pupe']:.4f}")
        if after["mean_pupe"] > 0.65:
            raise AssertionError(f"held-out PUPE {after['mean_pupe']:.4f} is not a useful local sanity result")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.train_ebn0_max < args.train_ebn0_min or args.steps <= 0 or args.batch_size <= 0 or args.eval_batches <= 0:
        raise SystemExit("invalid training range or nonpositive step/batch count")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    encoder, code, decoder = build_system(args, generator)
    if hasattr(encoder, "num_codewords") or any(hasattr(bank, "msg_to_atom") for bank in encoder.banks):
        raise AssertionError("scalable learning runner leaked a global message axis")
    H = code.factor_graph.parity_check_matrix()
    metadata = {"args": vars(args), "num_messages_conceptual": 1 << args.payload_bits,
                "local_state_size": encoder.state_size, "section_sizes": list(encoder.section_sizes),
                "num_information_sections": code.num_information_sections, "num_parity_sections": len(code.parity_supports),
                "num_sections": code.num_sections, "parity_check_shape": list(H.shape),
                "parity_check_matrix": H.tolist(), "outer_code_rate": code.num_information_sections / code.num_sections,
                "energy_certificate_initial": encoder.certify_exact_energy(), "learnable_parameters": sum(
                    parameter.numel() for parameter in list(encoder.parameters()) + list(decoder.parameters())
                    if parameter.requires_grad)}
    print(f"B={args.payload_bits} M=2^{args.payload_bits} local_state={encoder.state_size} "
          f"H={tuple(H.shape)} L={code.num_sections} q={1 << args.section_bits} mode={encoder.energy_mode}")
    started = time.time()
    before = evaluate(args, encoder, code, decoder, args.seed + 100_000)
    progress = train(args, encoder, code, decoder, generator)
    after = evaluate(args, encoder, code, decoder, args.seed + 100_000)
    validate_result(args, encoder, before, after, progress)
    metadata["energy_certificate_final"] = encoder.certify_exact_energy()
    metadata["sampled_energy_final"] = sampled_energy_report(
        encoder, code, min(2048, max(128, args.batch_size * args.eval_batches)),
        torch.Generator().manual_seed(args.seed + 200_000))
    metadata["wall_s"] = time.time() - started
    summary = {"metadata": metadata, "progress": progress, "initial": before, "trained": after}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"metadata": metadata, "encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
               out_dir / "checkpoint.pt")
    plot_summary(progress, before, after, out_dir / "learning_summary.png")
    print(f"held-out loss {before['mean_loss']:.5f} -> {after['mean_loss']:.5f}; "
          f"PUPE {before['mean_pupe']:.4f} -> {after['mean_pupe']:.4f}; wall={metadata['wall_s']:.1f}s")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
