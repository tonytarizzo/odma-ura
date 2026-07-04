"""Global-codebook morphology plots for representative URA constructions.

The script builds small explicit codebooks from several algebraic families and
studies the induced global matrix Phi directly. It is diagnostic: no decoder is
run, and every column is normalised before comparing geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+",
                   default=["dense", "slotted", "odma", "spreading", "ccs", "sparc", "coded_pattern"],
                   choices=["dense", "slotted", "odma", "spreading", "ccs", "sparc", "coded_pattern"])
    p.add_argument("-B", "--payload-bits", type=int, default=8)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--num-patterns", type=int, default=8)
    p.add_argument("--num-sections", type=int, default=4)
    p.add_argument("--active-k", type=int, default=8)
    p.add_argument("--num-active-samples", type=int, default=512)
    p.add_argument("--heatmap-cols", type=int, default=192)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default="results/codebook_morphology")
    return p.parse_args(argv)


def unit_columns(A: np.ndarray) -> np.ndarray:
    return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)


def gaussian_codebook(rng: np.random.Generator, n: int, M: int) -> np.ndarray:
    return unit_columns(rng.standard_normal((n, M)))


def random_placements(rng: np.random.Generator, Q: int, n: int, d: int) -> list[np.ndarray]:
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(Q)]


def build_dense(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    return gaussian_codebook(rng, n, M), {"family": "dense", "description": "iid Gaussian dense columns"}


def build_slotted(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    Q = int(args.num_patterns)
    if n % Q != 0:
        raise ValueError(f"slotted requires n divisible by num_patterns, got n={n}, Q={Q}")
    d = n // Q
    V = math.ceil(M / Q)
    C = gaussian_codebook(rng, d, V)
    Phi = np.zeros((n, M))
    for m in range(M):
        q, v = m % Q, m // Q
        Phi[q * d:(q + 1) * d, m] = C[:, v]
    return unit_columns(Phi), {"family": "slotted", "Q": Q, "slot_len": d, "V": V}


def build_odma(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    Q, d = int(args.num_patterns), int(args.d)
    if d <= 0 or d > n:
        raise ValueError(f"ODMA requires 0 < d <= n, got d={d}, n={n}")
    C = gaussian_codebook(rng, d, M)
    placements = random_placements(rng, Q, n, d)
    Phi = np.zeros((n, M))
    for m in range(M):
        Phi[placements[m % Q], m] = C[:, m]
    return unit_columns(Phi), {"family": "odma", "Q": Q, "d": d, "mapping": "legacy m mod Q"}


def build_spreading(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    Q, d = int(args.num_patterns), int(args.d)
    V = math.ceil(M / Q)
    C = gaussian_codebook(rng, d, V)
    spreaders = [rng.standard_normal((n, d)) / math.sqrt(d) for _ in range(Q)]
    Phi = np.zeros((n, M))
    for m in range(M):
        q, v = m % Q, m // Q
        Phi[:, m] = spreaders[q] @ C[:, v]
    return unit_columns(Phi), {"family": "spreading", "Q": Q, "d": d, "V": V}


def _digits(m: int, base: int, L: int) -> list[int]:
    out = []
    for _ in range(L):
        out.append(m % base)
        m //= base
    return out


def build_ccs(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    L = int(args.num_sections)
    if n % L != 0:
        raise ValueError(f"CCS requires n divisible by num_sections, got n={n}, L={L}")
    ns = n // L
    J = math.ceil(M ** (1.0 / L))
    while J ** L < M:
        J += 1
    A = [gaussian_codebook(rng, ns, J) for _ in range(L)]
    Phi = np.zeros((n, M))
    for m in range(M):
        for ell, j in enumerate(_digits(m, J, L)):
            Phi[ell * ns:(ell + 1) * ns, m] = A[ell][:, j]
    return unit_columns(Phi), {"family": "ccs", "L": L, "section_len": ns, "J": J, "sections": "disjoint"}


def build_sparc(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    L = int(args.num_sections)
    J = math.ceil(M ** (1.0 / L))
    while J ** L < M:
        J += 1
    A = [gaussian_codebook(rng, n, J) / math.sqrt(L) for _ in range(L)]
    Phi = np.zeros((n, M))
    for m in range(M):
        for ell, j in enumerate(_digits(m, J, L)):
            Phi[:, m] += A[ell][:, j]
    return unit_columns(Phi), {"family": "sparc", "L": L, "J": J, "sections": "overlapping full-frame"}


def build_coded_pattern(rng: np.random.Generator, n: int, M: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    Q, d = int(args.num_patterns), int(args.d)
    V = max(math.ceil(1.5 * M / Q), 2)
    C = gaussian_codebook(rng, n, V)
    patterns = random_placements(rng, Q, n, d)
    all_pairs = [(q, v) for q in range(Q) for v in range(V)]
    chosen = rng.choice(len(all_pairs), size=M, replace=False)
    Phi = np.zeros((n, M))
    for col, pair_idx in enumerate(chosen):
        q, v = all_pairs[int(pair_idx)]
        Phi[patterns[q], col] = C[patterns[q], v]
    return unit_columns(Phi), {"family": "coded_pattern", "Q": Q, "d": d, "V": V, "valid_pairs": M}


BUILDERS = {
    "dense": build_dense,
    "slotted": build_slotted,
    "odma": build_odma,
    "spreading": build_spreading,
    "ccs": build_ccs,
    "sparc": build_sparc,
    "coded_pattern": build_coded_pattern,
}


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    qs = np.quantile(x, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1])
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(qs[0]),
            "q01": float(qs[1]), "q05": float(qs[2]), "median": float(qs[3]),
            "q95": float(qs[4]), "q99": float(qs[5]), "max": float(qs[6])}


def active_set_diagnostics(Phi: np.ndarray, K: int, num_samples: int,
                           rng: np.random.Generator) -> dict:
    M = Phi.shape[1]
    K = min(int(K), M)
    frob, cond, mineig = [], [], []
    for _ in range(int(num_samples)):
        idx = rng.choice(M, size=K, replace=False)
        Gs = Phi[:, idx].T @ Phi[:, idx]
        eig = np.linalg.eigvalsh(Gs)
        frob.append(float(np.linalg.norm(Gs - np.eye(K), ord="fro")))
        mineig.append(float(np.min(eig)))
        cond.append(float(np.max(eig) / max(np.min(eig), 1e-12)))
    return {"gram_frobenius_deviation": stats(np.asarray(frob)),
            "condition_number": stats(np.asarray(cond)),
            "min_eigenvalue": stats(np.asarray(mineig))}


def analyse_phi(Phi: np.ndarray, args: argparse.Namespace,
                rng: np.random.Generator) -> tuple[dict, dict]:
    abs_tol = 1e-9
    n, M = Phi.shape
    G = Phi.T @ Phi
    absG = np.abs(G)
    off = absG[~np.eye(M, dtype=bool)]
    support = np.abs(Phi) > abs_tol
    overlap = support.T.astype(float) @ support.astype(float)
    off_overlap = overlap[~np.eye(M, dtype=bool)]
    row_load = support.sum(axis=1)
    row_energy = np.sum(Phi ** 2, axis=1)
    col_support = support.sum(axis=0)
    svals = np.linalg.svd(Phi, compute_uv=False)
    p = svals / max(np.sum(svals), 1e-12)
    effective_rank = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-24)))))
    summary = {
        "shape": {"n": int(n), "M": int(M)},
        "column_support": stats(col_support),
        "row_load": stats(row_load),
        "row_energy": stats(row_energy),
        "coherence_abs": {**stats(off), "rms": float(np.sqrt(np.mean(off ** 2)))},
        "support_overlap": stats(off_overlap),
        "near_duplicate_pairs": int(np.sum(off >= 1.0 - 1e-8) // 2),
        "singular_values": {**stats(svals), "effective_rank": effective_rank,
                            "effective_rank_fraction": effective_rank / max(1, min(n, M)),
                            "stable_rank": float(np.sum(svals ** 2) / max(np.max(svals) ** 2, 1e-12))},
        "active_sets": active_set_diagnostics(Phi, args.active_k, args.num_active_samples, rng),
    }
    return summary, {"absG": absG, "overlap": overlap, "row_load": row_load,
                     "row_energy": row_energy, "off": off, "svals": svals}


def subset_matrix(A: np.ndarray, max_cols: int) -> np.ndarray:
    if A.shape[0] <= max_cols:
        return A
    idx = np.linspace(0, A.shape[0] - 1, max_cols).round().astype(int)
    return A[np.ix_(idx, idx)]


def plot_family(name: str, mats: dict, summary: dict, out_dir: Path, heatmap_cols: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    for key, title, path, vmax in [
        ("absG", "|Phi^T Phi|", "gram_abs_heatmap.png", 1.0),
        ("overlap", "Support overlap count", "support_overlap_heatmap.png", None),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        img = subset_matrix(mats[key], heatmap_cols)
        im = ax.imshow(img, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_title(f"{name}: {title}")
        ax.set_xlabel("message index")
        ax.set_ylabel("message index")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / path, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(mats["row_load"], bins=30)
    axes[0, 0].set_title("Row support load")
    axes[0, 1].hist(mats["row_energy"], bins=30)
    axes[0, 1].set_title("Row energy")
    axes[1, 0].hist(mats["off"], bins=50)
    axes[1, 0].set_title("Off-diagonal |Gram|")
    axes[1, 1].semilogy(np.maximum(mats["svals"], 1e-12), marker=".")
    axes[1, 1].set_title("Singular spectrum")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_dir / "morphology_panels.png", dpi=150)
    plt.close(fig)

    active = summary["active_sets"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes, ["condition_number", "min_eigenvalue", "gram_frobenius_deviation"]):
        vals = [v for k, v in active[key].items() if k in ("min", "q01", "q05", "median", "q95", "q99", "max")]
        ax.plot(vals, marker="o")
        ax.set_xticks(range(7), ["min", "q01", "q05", "med", "q95", "q99", "max"], rotation=35)
        ax.set_title(key.replace("_", " "))
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "active_set_conditioning.png", dpi=150)
    plt.close(fig)


def plot_combined(summaries: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["name"] for s in summaries]
    metrics = [
        ("coherence_abs", "rms", "RMS |Gram offdiag|"),
        ("coherence_abs", "q99", "q99 |Gram offdiag|"),
        ("row_load", "std", "row-load std"),
        ("singular_values", "effective_rank_fraction", "effective-rank frac"),
        ("active_sets.condition_number", "median", "median active cond"),
        ("active_sets.gram_frobenius_deviation", "median", "median active Gram dev"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (group, key, title) in zip(axes.flat, metrics):
        vals = []
        for s in summaries:
            obj = s
            for part in group.split("."):
                obj = obj[part]
            vals.append(obj[key])
        ax.bar(names, vals)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    M = 1 << int(args.payload_bits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    print(f"Codebook morphology: n={args.n}, M={M}, families={args.families}")
    for i, family in enumerate(args.families):
        rng = np.random.default_rng(int(args.seed) + 1009 * i)
        Phi, meta = BUILDERS[family](rng, int(args.n), M, args)
        Phi = unit_columns(Phi)
        summary, mats = analyse_phi(Phi, args, rng)
        summary = {"name": family, "meta": meta, **summary}
        summaries.append(summary)
        fam_dir = out_dir / family
        (fam_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (fam_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        plot_family(family, mats, summary, fam_dir, int(args.heatmap_cols))
        print(f"{family:<14s} rms|Goff|={summary['coherence_abs']['rms']:.4f} "
              f"q99={summary['coherence_abs']['q99']:.4f} "
              f"rowload_std={summary['row_load']['std']:.2f} "
              f"active_cond_med={summary['active_sets']['condition_number']['median']:.2f}", flush=True)

    payload = {"args": vars(args), "summaries": summaries}
    (out_dir / "morphology_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_combined(summaries, out_dir / "combined_morphology_summary.png")
    print(f"Wrote {out_dir / 'morphology_summary.json'}")
    print(f"Wrote {out_dir / 'combined_morphology_summary.png'}")


if __name__ == "__main__":
    main()
