"""Exact encoder moments and row balancing for Report 8; no receiver training or PUPE claims."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.polar_mapping import make_encoder
from benchmarks.ura_comparison import make_encoder as make_explicit_encoder


def moments(book):
    """Return C=Phi Phi^T/M and mu=Phi 1/M without enumerating payloads."""
    a, positions = book.amplitudes(), book.positions
    patterns = len(positions)
    q = book.symbol_equivalence.to(a.dtype)
    local = a[:, :, None] * q * a[:, None, :]
    indices = positions[:, :, None] * book.n + positions[:, None, :]
    frame = a.new_zeros(book.n * book.n).scatter_add(0, indices.reshape(-1), local.reshape(-1)) / patterns
    generator = book.polar.encode(np.eye(book.polar.payload_bits, dtype=np.uint8)).T
    constant = torch.as_tensor(~generator.any(axis=1), dtype=a.dtype, device=a.device)
    mean = a.new_zeros(book.n).scatter_add(0, positions.reshape(-1), (a * constant).reshape(-1)) / patterns
    return frame.reshape(book.n, book.n), mean


def defect_gradient(book):
    """Analytic gradient of ||C-I/n||_F^2 with respect to energy softmax logits."""
    if book.mapping != "energy": raise ValueError("The derivative requires energy logits")
    c, _ = moments(book)
    a, positions = book.amplitudes(), book.positions
    residual = c - torch.eye(book.n, dtype=c.dtype, device=c.device) / book.n
    local = residual[positions[:, :, None], positions[:, None, :]] * book.symbol_equivalence
    grad_e = (2 / len(positions)) * torch.einsum("pij,pj->pi", local, a) / a
    e = a.square()
    return e * (grad_e - (e * grad_e).sum(-1, keepdim=True))


def balanced_energy(positions, n, tolerance=1e-10, max_iterations=10000):
    """KL-nearest uniform edge weights with unit pattern mass and equal resource mass.

    Structural zeros stay zero. Coverage alone is not sufficient for feasibility;
    lack of convergence is reported, never silently labelled a balanced solution.
    """
    positions = np.asarray(positions)
    if (positions.ndim != 2 or not np.issubdtype(positions.dtype, np.integer) or positions.size == 0
            or n < 1 or np.any(positions < 0) or np.any(positions >= n)
            or any(len(np.unique(row)) != len(row) for row in positions)):
        raise ValueError("Expected distinct, valid channel positions within every pattern")
    if tolerance <= 0 or max_iterations < 1: raise ValueError("Invalid convergence budget")
    p, nc = positions.shape
    if np.any(np.bincount(positions.ravel(), minlength=n) == 0): raise ValueError("Uncovered channel resource")
    e, target = np.full((p, nc), 1 / nc), p / n
    for iteration in range(1, max_iterations + 1):
        mass = np.bincount(positions.ravel(), weights=e.ravel(), minlength=n)
        e *= (target / mass)[positions]
        e /= e.sum(1, keepdims=True)
        mass = np.bincount(positions.ravel(), weights=e.ravel(), minlength=n)
        error = float(np.max(np.abs(mass / target - 1)))
        if error <= tolerance:
            return e, {"iterations": iteration, "max_relative_row_error": error}
    raise ValueError(f"Row balancing did not converge; support feasibility or budget needs review (error={error:.3g})")


@torch.no_grad()
def moment_diagnostics(c, mean):
    n = len(c)
    row_term = n * (c.diag() - 1 / n).square().sum()
    off = c - torch.diag(c.diag())
    off_term = n * off.square().sum()
    eigen = torch.linalg.eigvalsh(c)
    return {"row_defect": float(row_term), "off_diagonal_defect": float(off_term),
            "total_defect": float(row_term + off_term), "frame_participation": float(c.trace().square() / c.square().sum()),
            "mean_codeword_norm2": float(mean.square().sum()), "trace": float(c.trace()),
            "min_eigen_times_n": float(n * eigen[0]), "max_eigen_times_n": float(n * eigen[-1])}


def diagnostics(book): return moment_diagnostics(*moments(book))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("jobs/034_polar_mapping/results"))
    parser.add_argument("--out", type=Path, default=Path("jobs/034_polar_mapping/results/analysis/energy_derivation.json"))
    parser.add_argument("--dense-results", type=Path, default=Path("jobs/032_published_baselines/results/comparison"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for b in (12, 14):
        for seed in (3201, 3202, 3203):
            folder = args.results / f"B{b}_n256_polar_energy_joint_d2_s{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
            if checkpoint["config"] != summary["config"]: raise ValueError("Checkpoint/summary mismatch")
            _, encoder = make_encoder(summary["config"])
            encoder.load_state_dict(checkpoint["encoder"], strict=True)
            book = encoder.codebook.double().eval().requires_grad_(False)
            uniform, balanced = copy.deepcopy(book), copy.deepcopy(book)
            uniform.values.zero_()
            e, info = balanced_energy(book.positions.numpy(), book.n)
            balanced.values.copy_(torch.from_numpy(np.log(e)))
            generator = book.polar.encode(np.eye(book.polar.payload_bits, dtype=np.uint8)).T
            for name, variant in (("uniform", uniform), ("learned", book), ("balanced", balanced)):
                rows.append({"B": b, "seed": seed, "family": name, **diagnostics(variant),
                             "distinct_generator_rows": len(np.unique(generator, axis=0)),
                             "constant_symbols": int((~generator.any(axis=1)).sum()),
                             "balance_convergence": info if name == "balanced" else None})
            dense = torch.load(args.dense_results / f"B{b}_n256_dense_fixed_d2_s{seed}/checkpoint.pt",
                               map_location="cpu", weights_only=True)
            dense_encoder = make_explicit_encoder(dense["config"])
            dense_encoder.load_state_dict(dense["encoder"], strict=True)
            phi = dense_encoder.explicit_matrix().detach().double()
            rows.append({"B": b, "seed": seed, "family": "dense",
                         **moment_diagnostics(phi @ phi.T / phi.shape[1], phi.mean(1))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    for b in (12, 14):
        for family in ("uniform", "balanced", "learned", "dense"):
            group = [r for r in rows if r["B"] == b and r["family"] == family]
            print(b, family, {k: round(float(np.mean([r[k] for r in group])), 8)
                              for k in ("row_defect", "off_diagonal_defect", "total_defect", "mean_codeword_norm2")})
    print(f"Exact encoder diagnostics saved to {args.out}; no new PUPE measurements or training.")


if __name__ == "__main__": main()
