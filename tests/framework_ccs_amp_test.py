"""Exact reduced-B equivalence: author CCS-AMP code versus explicit framework.

The test uses the authors' Triadic4 outer graph at two bits per section
(``B=8, L=8, J=2``), extracts the exact deterministic subsampled-Hadamard
operator used by their dense inner code, and represents the resulting global
message codebook with framework components.  Both sides decode the same noisy
observation using the unchanged author AMP-BP decoder.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from framework.core import ComponentSpec, URASpec
from framework.encoder import build_encoder
from framework.outer_code import triadic_outer_code
from src.plotting import _configure_matplotlib
from tests.ccs_amp_author import default_author_dir, graph_for_preset, load_author_modules, number_matches


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author-code-dir", type=Path, default=default_author_dir())
    p.add_argument("--allow-unpinned-author-code", action="store_true")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--K-values", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[0.0, 4.0, 8.0])
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--transform-seed", type=int, default=0)
    p.add_argument("--amp-iterations", type=int, default=10)
    p.add_argument("--bp-iterations", type=int, default=1)
    p.add_argument("--list-extra", type=int, default=10)
    p.add_argument("--phi-atol", type=float, default=1e-12)
    p.add_argument("--out-dir", type=Path, default=Path("results/framework_equivalence_ccs_amp"))
    return p.parse_args(argv)


def all_bits(B: int) -> np.ndarray:
    values = np.arange(1 << B, dtype=np.uint64)[:, None]
    shifts = np.arange(B - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((values >> shifts) & 1).astype(int)


def extract_section_matrices(inner, L: int, section_size: int, n: int) -> list[np.ndarray]:
    matrices = [np.zeros((n, section_size), dtype=float) for _ in range(L)]
    for ell in range(L):
        for idx in range(section_size):
            basis = np.zeros(L * section_size)
            basis[ell * section_size + idx] = 1.0
            matrices[ell][:, idx] = inner.Encode(basis).reshape(-1)
    return matrices


def build_framework(section_matrices: list[np.ndarray], encoded_indices: np.ndarray, K: int):
    n, M = section_matrices[0].shape[0], encoded_indices.shape[1]
    eye = torch.eye(n, dtype=torch.float64).unsqueeze(0)
    specs = []
    for ell, matrix in enumerate(section_matrices):
        specs.append(ComponentSpec(Q=1, d=n, V=matrix.shape[1], N=matrix.shape[1], R_init="explicit", C_init="explicit",
                                   U_init="all_pairs", T_init="explicit", explicit_R=eye,
                                   explicit_C=torch.as_tensor(matrix, dtype=torch.float64),
                                   explicit_msg_to_atom=torch.as_tensor(encoded_indices[ell], dtype=torch.long)))
    return build_encoder(URASpec(n=n, num_codewords=M, num_active=K, payload_bits=8), specs, dtype=torch.float64)


def decode(modules, args, K: int, power: float, observation: np.ndarray):
    graph = graph_for_preset(modules, "explicit_b8")
    inner = modules.inner.DenseInnerCode(args.n, power, 1.0, K, graph)
    estimates, _ = inner.Decode(observation.copy(), args.amp_iterations, True, args.bp_iterations, graph)
    with contextlib.redirect_stdout(io.StringIO()):
        decoded = graph.decoder(estimates.copy(), min(1 << 2, K + args.list_extra))
    return estimates, decoded


def plot_points(points: list[dict], out_path: Path) -> None:
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = ["#0E7490", "#7C3AED", "#B45309"]
    for color, K in zip(colors, sorted({p["K"] for p in points})):
        curve = sorted([p for p in points if p["K"] == K], key=lambda p: p["ebn0_db"])
        x = [p["ebn0_db"] for p in curve]; y = [p["mean_pupe"] for p in curve]
        ax.plot(x, y, color=color, marker="o", lw=2.2, label=f"author direct, K={K}")
        ax.plot(x, y, color=color, marker="x", ls="--", lw=1.4, label=f"explicit framework, K={K}")
    ax.set(xlabel="Eb/N0 (dB)", ylabel="Mean PUPE", ylim=(-0.02, 1.02),
           title="Reduced-B CCS-AMP exact equivalence (B=8)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def main(argv=None):
    args = parse_args(argv); args.out_dir.mkdir(parents=True, exist_ok=True)
    modules = load_author_modules(args.author_code_dir, transform_seed=args.transform_seed,
                                  allow_unpinned=args.allow_unpinned_author_code)
    B, L, J, global_M, section_M = 8, 8, 2, 1 << 8, 1 << 2
    bits = all_bits(B)
    rows, max_phi_error, all_estimates_match, all_lists_match = [], 0.0, True, True
    for K in args.K_values:
        for ebn0_db in args.ebn0_grid:
            power = 2.0 * B * 10.0 ** (ebn0_db / 10.0) / args.n
            graph = graph_for_preset(modules, "explicit_b8")
            inner = modules.inner.DenseInnerCode(args.n, power, 1.0, K, graph)
            codewords = graph.encodemessages(bits)
            encoded_indices = np.argmax(codewords.reshape(global_M, L, section_M), axis=2).T
            procedural_indices = triadic_outer_code(B, J).encode_bits(torch.as_tensor(bits)).numpy().T
            if not np.array_equal(procedural_indices, encoded_indices):
                raise AssertionError("procedural modular triadic encoder disagrees with the CCS-AMP author encoder")
            direct_phi = np.column_stack([inner.Encode(codeword).reshape(-1) for codeword in codewords])
            section_matrices = extract_section_matrices(inner, L, section_M, args.n)
            encoder = build_framework(section_matrices, encoded_indices, K)
            framework_phi = encoder.explicit_matrix().detach().cpu().numpy()
            phi_error = float(np.max(np.abs(direct_phi - framework_phi)))
            max_phi_error = max(max_phi_error, phi_error)
            if phi_error > args.phi_atol:
                raise AssertionError(f"direct/framework Phi mismatch {phi_error:.3e} > {args.phi_atol:.3e}")
            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                rng = np.random.default_rng(seed)
                active = rng.integers(0, global_M, size=K)
                counts = np.bincount(active, minlength=global_M).astype(float)
                direct_clean = direct_phi @ counts
                framework_clean = encoder.encode(torch.as_tensor(counts, dtype=torch.float64)).detach().cpu().numpy()
                clean_error = float(np.max(np.abs(direct_clean - framework_clean)))
                observation = (direct_clean + rng.standard_normal(args.n)).reshape(-1, 1)
                est_direct, list_direct = decode(modules, args, K, power, observation)
                est_framework, list_framework = decode(modules, args, K, power, observation)
                estimates_match = bool(np.array_equal(est_direct, est_framework))
                direct_keys = [np.packbits(np.asarray(x, dtype=np.uint8)).tobytes() for x in list_direct]
                framework_keys = [np.packbits(np.asarray(x, dtype=np.uint8)).tobytes() for x in list_framework]
                lists_match = direct_keys == framework_keys
                all_estimates_match &= estimates_match; all_lists_match &= lists_match
                true_codewords = codewords[active]
                matches = number_matches(true_codewords, list_direct, K)
                rows.append({"K": K, "ebn0_db": ebn0_db, "seed": seed, "pupe": (K - matches) / K,
                             "phi_error": phi_error, "clean_error": clean_error,
                             "amp_estimates_match": estimates_match, "decoded_lists_match": lists_match})
                print(f"K={K:<2d} Eb/N0={ebn0_db:>4.1f} seed={seed} PUPE={(K-matches)/K:.4f} "
                      f"PhiErr={phi_error:.1e} listMatch={lists_match}", flush=True)
    if not all_estimates_match or not all_lists_match:
        raise AssertionError("author direct/framework decoded outputs did not match")
    points = []
    for K in args.K_values:
        for ebn0_db in args.ebn0_grid:
            vals = [r["pupe"] for r in rows if r["K"] == K and r["ebn0_db"] == ebn0_db]
            points.append({"K": K, "ebn0_db": ebn0_db, "mean_pupe": float(np.mean(vals))})
    payload = {"args": {**vars(args), "author_code_dir": str(args.author_code_dir)}, "author_commit": modules.commit,
               "max_phi_error": max_phi_error, "all_amp_estimates_match": all_estimates_match,
               "all_decoded_lists_match": all_lists_match, "rows": rows, "points": points}
    (args.out_dir / "ccs_amp_equivalence_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_points(points, args.out_dir / "ccs_amp_equivalence_curves.png")
    print(f"Exact equivalence passed; max Phi error={max_phi_error:.3e}")


if __name__ == "__main__":
    main()
