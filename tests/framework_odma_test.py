"""Framework ODMA+URA inference sanity test.

This script checks that the new framework can represent the legacy ODMA+URA
setup and then runs oracle-K OMP inference over job-style parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.analysis import analyze_encoder  # noqa: E402
from framework.channel import constant_fading, sample_batch, uniform_counts_generator  # noqa: E402
from framework.core import ComponentSpec, URASpec  # noqa: E402
from framework.decoders import oracle_k_omp  # noqa: E402
from framework.encoder import build_encoder  # noqa: E402
from framework.metrics import batch_evaluate  # noqa: E402
from framework.pipeline import odma_component_specs  # noqa: E402
from src.scenario import build_scenario  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-B", "--payload-bits", type=int, default=10)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--K-values", nargs="+", type=int, default=[2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200])
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0, 4.0])
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--out-dir", default="results/framework_odma_test")
    p.add_argument("--check-legacy-algebra", action="store_true", default=True)
    p.add_argument("--no-check-legacy-algebra", dest="check_legacy_algebra", action="store_false")
    return p.parse_args(argv)


def build_framework_odma_encoder(n: int, d: int, num_blocks: int, num_codewords: int,
                                 K: int, num_antennas: int, dtype: torch.dtype,
                                 generator: torch.Generator):
    spec = URASpec(n=n, num_codewords=num_codewords, num_active=K, num_antennas=num_antennas)
    components = odma_component_specs(spec, d, num_blocks, learn_C=False, learn_R=False)
    return build_encoder(spec, components, dtype=dtype, generator=generator)


def check_legacy_algebra() -> None:
    n, d, num_blocks, M, K, ant, seed = 32, 8, 4, 16, 3, 2, 123
    legacy = build_scenario(
        n=n, d=d, num_blocks=num_blocks, num_codewords=M,
        num_devices_active=K, num_antennas=ant, esn0_db=0.0, seed=seed)
    R = torch.stack([torch.as_tensor(legacy.P_mats[b], dtype=torch.float64) for b in range(num_blocks)])
    C = torch.as_tensor(legacy.codebook.T, dtype=torch.float64)
    msg = torch.arange(M)
    component = ComponentSpec(
        Q=num_blocks, d=d, V=M, N=M,
        R_init="explicit", C_init="explicit", U_init="explicit", T_init="identity",
        explicit_R=R, explicit_C=C,
        explicit_atom_q=msg % num_blocks,
        explicit_atom_v=msg,
    )
    spec = URASpec(n=n, num_codewords=M, num_active=K, num_antennas=ant)
    encoder = build_encoder(spec, [component], dtype=torch.float64)
    phi_framework = encoder.explicit_matrix()
    phi_legacy = torch.zeros(n, M, dtype=torch.float64)
    for m in range(M):
        phi_legacy[:, m] = torch.as_tensor(legacy.P_mats[m % num_blocks], dtype=torch.float64) @ torch.as_tensor(legacy.codebook[m], dtype=torch.float64)
    if not torch.allclose(phi_framework, phi_legacy, atol=1e-12, rtol=1e-12):
        err = float((phi_framework - phi_legacy).abs().max().item())
        raise AssertionError(f"framework ODMA Phi does not match legacy construction; max error {err:.3e}")


def run_point(args: argparse.Namespace, K: int, ebn0_db: float, seeds: list[int]) -> tuple[list[dict], dict]:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    M = 1 << int(args.payload_bits)
    rows = []
    for seed in seeds:
        gen = torch.Generator().manual_seed(int(seed))
        encoder = build_framework_odma_encoder(args.n, args.d, args.num_blocks, M, K, args.num_antennas, dtype, gen)
        counts_sampler = uniform_counts_generator(K, M, gen, encoder.device)
        fading_sampler = constant_fading(args.num_antennas, dtype, encoder.device)
        batch = sample_batch(encoder, 1, counts_sampler, fading_sampler, ebn0_db, gen)
        t0 = time.time()
        out = oracle_k_omp(encoder, batch.Y, batch.H, K)
        wall = time.time() - t0
        _, summary = batch_evaluate(batch.counts, out.counts.to(batch.counts.dtype), max_list_size=K)
        row = {"K": int(K), "ebn0_db": float(ebn0_db), "seed": int(seed), "metrics": summary, "wall_s": wall}
        rows.append(row)
        print(f"K={K:<4d} Eb/N0={ebn0_db:>5.2f} seed={seed:<4d} "
              f"L1acc={summary['l1_acc']:.4f} PUPE={summary['pupe']:.4f} wall={wall:.2f}s", flush=True)
    metrics = rows[0]["metrics"].keys()
    point = {
        "K": int(K),
        "ebn0_db": float(ebn0_db),
        "num_trials": len(rows),
        "mean_wall_s": float(np.mean([r["wall_s"] for r in rows])),
        **{f"mean_{k}": float(np.mean([r["metrics"][k] for r in rows])) for k in metrics},
    }
    return rows, point


def plot_summary(points: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ebn0_values = sorted({p["ebn0_db"] for p in points})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ebn0 in ebn0_values:
        curve = sorted([p for p in points if p["ebn0_db"] == ebn0], key=lambda p: p["K"])
        axes[0].plot([p["K"] for p in curve], [p["mean_l1_acc"] for p in curve], marker="o", label=f"{ebn0:g} dB")
        axes[1].plot([p["K"] for p in curve], [p["mean_pupe"] for p in curve], marker="o", label=f"{ebn0:g} dB")
    axes[0].set_ylabel("Mean L1 accuracy")
    axes[1].set_ylabel("Mean PUPE")
    for ax in axes:
        ax.set_xlabel("Active devices K")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Eb/N0")
    fig.suptitle("Framework ODMA+URA oracle-K OMP inference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_legacy_algebra:
        check_legacy_algebra()
        print("Legacy algebra check passed: framework Phi matches src ODMA construction.")

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Framework ODMA test: n={args.n}, d={args.d}, blocks={args.num_blocks}, "
          f"B={args.payload_bits}, M={1 << args.payload_bits}, antennas={args.num_antennas}")
    print(f"K values: {args.K_values}")
    print(f"Eb/N0 grid: {args.ebn0_grid}")
    print(f"Seeds: {seeds}")

    all_rows: list[dict] = []
    points: list[dict] = []
    for K in args.K_values:
        for ebn0_db in args.ebn0_grid:
            rows, point = run_point(args, K, ebn0_db, seeds)
            all_rows.extend(rows)
            points.append(point)

    payload = {"args": vars(args), "points": points, "trials": all_rows}
    (out_dir / "framework_odma_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_summary(points, out_dir / "framework_odma_summary.png")
    representative_encoder = build_framework_odma_encoder(
        args.n, args.d, args.num_blocks, 1 << int(args.payload_bits),
        int(args.K_values[0]), args.num_antennas,
        torch.float64 if args.dtype == "float64" else torch.float32,
        torch.Generator().manual_seed(int(args.seed_start)))
    analyze_encoder(representative_encoder, out_dir / "encoding_analysis")
    print(f"Wrote {out_dir / 'framework_odma_summary.json'}")
    print(f"Wrote {out_dir / 'framework_odma_summary.png'}")


if __name__ == "__main__":
    main()
