"""Post-run encoding analysis for framework encoders."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .encoder import Encoder


MAX_PAIRWISE_COLUMNS = 2048
MAX_EXACT_SVD_NUMEL = 6_000_000
DEFAULT_ACTIVE_SET_SAMPLES = 128


def tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().flatten().real.to(torch.float64).cpu()
    if x.numel() == 0:
        return {}
    mean = float(x.mean())
    std = float(x.std(unbiased=False))
    centered = x - mean
    qs = torch.quantile(x, torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64))
    return {
        "mean": mean,
        "std": std,
        "min": float(qs[0]),
        "q01": float(qs[1]),
        "q05": float(qs[2]),
        "median": float(qs[3]),
        "q95": float(qs[4]),
        "q99": float(qs[5]),
        "max": float(qs[6]),
        "skew": float((centered ** 3).mean() / max(std ** 3, 1e-24)),
        "excess_kurtosis": float((centered ** 4).mean() / max(std ** 4, 1e-24) - 3.0),
    }


def gini(x: torch.Tensor) -> float:
    x = x.detach().flatten().real.to(torch.float64).cpu().abs()
    if x.numel() == 0 or float(x.sum()) == 0.0:
        return 0.0
    x, _ = torch.sort(x)
    n = x.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2.0 * torch.sum(idx * x) / (n * torch.sum(x))) - (n + 1.0) / n)


def js_divergence_to_gaussian(x: torch.Tensor, bins: int = 80) -> float:
    x = x.detach().flatten().real.to(torch.float64).cpu()
    std = x.std(unbiased=False)
    if x.numel() < 2 or float(std) <= 1e-24:
        return 0.0
    z = ((x - x.mean()) / std).clamp(-6, 6)
    edges = torch.linspace(-6, 6, bins + 1, dtype=torch.float64)
    p = torch.histogram(z, bins=edges).hist.to(torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    q = torch.exp(-0.5 * centers ** 2)
    p = p / p.sum().clamp_min(1e-24)
    q = q / q.sum().clamp_min(1e-24)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log((p / m.clamp_min(1e-24)).clamp_min(1e-24)))
    kl_qm = torch.sum(q * torch.log((q / m.clamp_min(1e-24)).clamp_min(1e-24)))
    return float(0.5 * (kl_pm + kl_qm))


def singular_value_summary(Phi: torch.Tensor) -> dict:
    if Phi.numel() > MAX_EXACT_SVD_NUMEL:
        return {"skipped": f"matrix has {Phi.numel()} entries; exact SVD skipped"}
    s = torch.linalg.svdvals(Phi).detach().cpu().to(torch.float64)
    stats = tensor_stats(s)
    if s.numel() == 0 or float(s.sum()) == 0.0:
        stats.update({"effective_rank": 0.0, "stable_rank": 0.0, "top_singular_fraction": 0.0})
        return stats
    p = s / s.sum()
    entropy_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-24))).sum())
    stable_rank = torch.sum(s ** 2) / s.max().clamp_min(1e-24) ** 2
    stats.update({
        "effective_rank": float(entropy_rank),
        "stable_rank": float(stable_rank),
        "top_singular_fraction": float(s.max() / s.sum().clamp_min(1e-24)),
    })
    return stats


def sampled_columns(Phi: torch.Tensor, max_columns: int = MAX_PAIRWISE_COLUMNS) -> torch.Tensor:
    if Phi.shape[1] <= max_columns:
        return Phi
    idx = torch.linspace(0, Phi.shape[1] - 1, max_columns, device=Phi.device).round().long()
    return Phi.index_select(1, idx)


def codeword_coherence(Phi: torch.Tensor) -> tuple[torch.Tensor, dict]:
    Phi_sample = sampled_columns(Phi)
    P = Phi_sample / Phi_sample.norm(dim=0, keepdim=True).clamp_min(1e-12)
    G = P.conj().T @ P if P.is_complex() else P.T @ P
    abs_G = G.abs()
    M = abs_G.shape[0]
    off = abs_G[~torch.eye(M, dtype=torch.bool, device=abs_G.device)].detach().cpu().to(torch.float64)
    q = torch.quantile(off, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64))
    return abs_G.detach().cpu(), {
        "sampled_columns": int(M),
        "mean_abs": float(off.mean()),
        "rms_abs": float(torch.sqrt((off ** 2).mean())),
        "max_abs": float(off.max()),
        "q50_abs": float(q[0]),
        "q90_abs": float(q[1]),
        "q95_abs": float(q[2]),
        "q99_abs": float(q[3]),
    }


def effective_support(x: torch.Tensor, dim: int) -> torch.Tensor:
    a = x.detach().abs().to(torch.float64)
    return (a.sum(dim=dim) ** 2) / (a ** 2).sum(dim=dim).clamp_min(1e-24)


def unit_columns(Phi: torch.Tensor) -> torch.Tensor:
    return Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-12)


def support_structure(Phi: torch.Tensor, abs_tol: float = 1e-9) -> dict:
    support = sampled_columns(Phi).detach().abs() > abs_tol
    row_load = support.to(torch.float64).sum(dim=1)
    col_support = support.to(torch.float64).sum(dim=0)
    out = {
        "sampled_columns": int(support.shape[1]),
        "column_support": tensor_stats(col_support),
        "row_load": tensor_stats(row_load),
        "row_load_gini": gini(row_load),
    }
    if support.shape[1] > 1:
        overlap = support.to(torch.float64).T @ support.to(torch.float64)
        off = overlap[~torch.eye(overlap.shape[0], dtype=torch.bool, device=overlap.device)]
        out["support_overlap"] = tensor_stats(off)
    else:
        out["support_overlap"] = {}
    return out


def active_set_diagnostics(Phi: torch.Tensor, active_k: int, num_samples: int,
                           generator: torch.Generator | None = None) -> dict:
    n, M = Phi.shape
    K = min(int(active_k), M)
    if K <= 0:
        return {"skipped": "active_k must be positive"}
    P = unit_columns(Phi.detach())
    gram_dev, cond, mineig = [], [], []
    eye = torch.eye(K, dtype=P.real.dtype if P.is_complex() else P.dtype, device=P.device)
    for _ in range(int(num_samples)):
        idx = torch.randperm(M, generator=generator, device=P.device)[:K]
        A = P.index_select(1, idx)
        Gs = A.conj().T @ A if A.is_complex() else A.T @ A
        Gs = Gs.real.to(torch.float64)
        eig = torch.linalg.eigvalsh(Gs)
        min_eig = eig.min().clamp_min(1e-12)
        gram_dev.append(torch.linalg.norm(Gs - eye.to(Gs.dtype), ord="fro") / math.sqrt(K))
        mineig.append(eig.min())
        cond.append(eig.max() / min_eig)
    return {
        "active_k": int(K),
        "num_samples": int(num_samples),
        "gram_deviation_per_active": tensor_stats(torch.stack(gram_dev)),
        "condition_number": tensor_stats(torch.stack(cond)),
        "min_eigenvalue": tensor_stats(torch.stack(mineig)),
    }


def support_search_margin(Phi: torch.Tensor, active_k: int, num_samples: int,
                          generator: torch.Generator | None = None) -> dict:
    M = Phi.shape[1]
    K = min(int(active_k), M)
    if K <= 0:
        return {"skipped": "active_k must be positive"}
    if K >= M:
        return {"skipped": "active_k leaves no inactive columns"}
    P = unit_columns(Phi.detach())
    margins, ratios, true_min, false_max = [], [], [], []
    for _ in range(int(num_samples)):
        idx = torch.randperm(M, generator=generator, device=P.device)[:K]
        active = torch.zeros(M, dtype=torch.bool, device=P.device)
        active[idx] = True
        y = P.index_select(1, idx).sum(dim=1)
        scores = ((P.conj().T @ y) if P.is_complex() else (P.T @ y)).real
        t_min = scores[active].min()
        f_max = scores[~active].max()
        margins.append(t_min - f_max)
        ratios.append(f_max / t_min.clamp_min(1e-12))
        true_min.append(t_min)
        false_max.append(f_max)
    return {
        "active_k": int(K),
        "num_samples": int(num_samples),
        "margin_min_true_minus_max_false": tensor_stats(torch.stack(margins)),
        "false_to_true_score_ratio": tensor_stats(torch.stack(ratios)),
        "min_true_score": tensor_stats(torch.stack(true_min)),
        "max_false_score": tensor_stats(torch.stack(false_max)),
    }


def matrix_features(Phi: torch.Tensor) -> dict:
    Phi = Phi.detach()
    col_energy = (Phi.conj() * Phi).sum(dim=0).real if Phi.is_complex() else (Phi ** 2).sum(dim=0)
    row_energy = (Phi.conj() * Phi).sum(dim=1).real if Phi.is_complex() else (Phi ** 2).sum(dim=1)
    _, coherence = codeword_coherence(Phi)
    sv = singular_value_summary(Phi)
    col_mean = col_energy.mean().clamp_min(1e-24)
    return {
        "entry_near_zero_fraction": float((Phi.abs() <= 1e-9).to(torch.float64).mean()),
        "entry_abs_gini": gini(Phi.abs()),
        "entry_gaussian_js": js_divergence_to_gaussian(Phi.real if Phi.is_complex() else Phi),
        "entry_skew": tensor_stats(Phi.real if Phi.is_complex() else Phi).get("skew", 0.0),
        "entry_excess_kurtosis": tensor_stats(Phi.real if Phi.is_complex() else Phi).get("excess_kurtosis", 0.0),
        "column_energy_cv": float(col_energy.std(unbiased=False) / col_mean),
        "row_energy_gini": gini(row_energy),
        "coherence_mean_abs": coherence["mean_abs"],
        "coherence_max_abs": coherence["max_abs"],
        "effective_rank_fraction": float(sv.get("effective_rank", 0.0) / max(1, min(Phi.shape))),
        "stable_rank_fraction": float(sv.get("stable_rank", 0.0) / max(1, min(Phi.shape))),
        "top_singular_fraction": float(sv.get("top_singular_fraction", 0.0)),
        "codeword_effective_support_mean_fraction": float(effective_support(Phi, dim=0).mean() / max(1, Phi.shape[0])),
    }


def gaussian_like_matrix(Phi: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    G = torch.randn(Phi.shape, dtype=Phi.real.dtype if Phi.is_complex() else Phi.dtype, generator=generator, device=Phi.device)
    target_energy = ((Phi.conj() * Phi).sum(dim=0).real if Phi.is_complex() else (Phi ** 2).sum(dim=0)).mean().sqrt()
    G = G / G.norm(dim=0, keepdim=True).clamp_min(1e-12) * target_energy
    return G.to(dtype=Phi.dtype)


def gaussian_reference(Phi: torch.Tensor, num_refs: int = 4) -> dict:
    observed = matrix_features(Phi)
    refs = []
    gen = torch.Generator(device=Phi.device).manual_seed(2026)
    for _ in range(num_refs):
        refs.append(matrix_features(gaussian_like_matrix(Phi, gen)))
    zscores = {}
    for key, value in observed.items():
        vals = torch.tensor([r[key] for r in refs], dtype=torch.float64)
        mean = float(vals.mean())
        std = float(vals.std(unbiased=False))
        zscores[key] = {
            "observed": float(value),
            "gaussian_mean": mean,
            "gaussian_std": std,
            "z": 0.0 if std <= 1e-12 else float((float(value) - mean) / std),
        }
    z = torch.tensor([abs(v["z"]) for v in zscores.values()], dtype=torch.float64)
    return {
        "num_references": num_refs,
        "unrandomness_rms_z": float(torch.sqrt((z ** 2).mean())),
        "features": zscores,
    }


def matrix_analysis(name: str, Phi: torch.Tensor, include_gaussian_reference: bool = True,
                    active_k: int | None = None, num_active_samples: int = DEFAULT_ACTIVE_SET_SAMPLES,
                    include_recovery: bool = False) -> dict:
    Phi = Phi.detach()
    col_energy = (Phi.conj() * Phi).sum(dim=0).real if Phi.is_complex() else (Phi ** 2).sum(dim=0)
    row_energy = (Phi.conj() * Phi).sum(dim=1).real if Phi.is_complex() else (Phi ** 2).sum(dim=1)
    abs_gram, coherence = codeword_coherence(Phi)
    sampled_abs_gram = abs_gram
    near_duplicate_pairs = 0
    if sampled_abs_gram.shape[0] > 1:
        mask = ~torch.eye(sampled_abs_gram.shape[0], dtype=torch.bool)
        near_duplicate_pairs = int((sampled_abs_gram[mask] >= 1.0 - 1e-8).sum().item() // 2)
    out = {
        "name": name,
        "shape": {"n": int(Phi.shape[0]), "M": int(Phi.shape[1])},
        "column_energy": tensor_stats(col_energy),
        "row_energy": tensor_stats(row_energy),
        "row_energy_gini": gini(row_energy),
        "coherence": coherence,
        "singular_values": singular_value_summary(Phi),
        "entry_value_stats": tensor_stats(Phi.real if Phi.is_complex() else Phi),
        "entry_abs_stats": tensor_stats(Phi.abs()),
        "entry_near_zero_fraction": float((Phi.abs() <= 1e-9).to(torch.float64).mean()),
        "entry_abs_gini": gini(Phi.abs()),
        "entry_gaussian_js": js_divergence_to_gaussian(Phi.real if Phi.is_complex() else Phi),
        "support_structure": support_structure(Phi),
        "codeword_effective_support": tensor_stats(effective_support(Phi, dim=0)),
        "row_effective_participation": tensor_stats(effective_support(Phi.T, dim=0)),
        "sampled_near_duplicate_codeword_pairs": near_duplicate_pairs,
    }
    if include_recovery and active_k is not None:
        gen = torch.Generator(device=Phi.device).manual_seed(2027)
        out["active_set_diagnostics"] = active_set_diagnostics(Phi, active_k, num_active_samples, gen)
        out["support_search_margin"] = support_search_margin(Phi, active_k, num_active_samples, gen)
    if include_gaussian_reference:
        out["gaussian_reference"] = gaussian_reference(Phi)
    return out


def interesting_matrices(encoder: Encoder) -> dict[str, torch.Tensor]:
    matrices = {"global": encoder.explicit_matrix().detach()}
    component_mats = []
    for i, comp in enumerate(encoder.components):
        Phi_i = comp.explicit_matrix().detach()
        component_mats.append(Phi_i)
        matrices[f"component_{i}_global_contribution"] = Phi_i
        matrices[f"component_{i}_local_C"] = comp.C.detach()
        product_cols = int(comp.Q * comp.V)
        if product_cols <= 4096:
            cols = []
            for q in range(comp.Q):
                cols.append(comp.R[q].unsqueeze(1) * comp.C if comp.diagonal_operators else comp.R[q] @ comp.C)
            matrices[f"component_{i}_product_library_B"] = torch.cat(cols, dim=1).detach()
    if 1 < len(component_mats) <= 4:
        running = component_mats[0].clone()
        for i in range(1, len(component_mats)):
            running = running + component_mats[i]
            matrices[f"components_0_to_{i}_sum"] = running.detach()
        for i in range(len(component_mats)):
            for j in range(i + 1, len(component_mats)):
                matrices[f"components_{i}_{j}_sum"] = (component_mats[i] + component_mats[j]).detach()
    return matrices


def component_analysis(encoder: Encoder) -> list[dict]:
    out = []
    for i, comp in enumerate(encoder.components):
        R_stored = comp.R.detach()
        R = comp.materialize_operator_bank().detach()
        C = comp.C.detach()
        support = R.abs() > 1e-9
        overlap = support.reshape(R.shape[0], -1).to(torch.float64) @ support.reshape(R.shape[0], -1).to(torch.float64).T
        atom_counts = torch.bincount(comp.atom_q.detach().cpu(), minlength=R.shape[0]).to(torch.float64)
        out.append({
            "index": i,
            "R_shape": list(R.shape),
            "R_stored_shape": list(R_stored.shape),
            "R_diagonal": comp.diagonal_operators,
            "C_shape": list(C.shape),
            "num_atoms": int(comp.atom_q.numel()),
            "R_nonzero_fraction": float(support.to(torch.float64).mean()),
            "R_operator_overlap_mean": float(overlap.mean()),
            "R_operator_overlap_max_offdiag": (
                float(overlap[~torch.eye(R.shape[0], dtype=torch.bool)].max()) if R.shape[0] > 1 else 0.0),
            "R_operator_load_gini": gini(R.reshape(R.shape[0], -1).abs().sum(dim=1)),
            "atom_operator_count_stats": tensor_stats(atom_counts),
            "C_value_stats": tensor_stats(C),
            "C_column_energy_stats": tensor_stats((C.conj() * C).sum(dim=0).real if C.is_complex() else (C ** 2).sum(dim=0)),
            "C_coherence": codeword_coherence(C)[1] if C.shape[1] > 1 else {},
        })
    return out


def plot_encoding_analysis(Phi: torch.Tensor, abs_gram: torch.Tensor, singular_values: torch.Tensor, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    def draw_hist(ax, x: torch.Tensor, target_bins: int) -> None:
        x = x.detach().cpu().to(torch.float64)
        lo = float(x.min())
        hi = float(x.max())
        if abs(hi - lo) <= 1e-12:
            pad = max(abs(lo) * 1e-6, 1e-6)
            ax.hist(x.numpy(), bins=1, range=(lo - pad, hi + pad))
        else:
            ax.hist(x.numpy(), bins=min(target_bins, max(1, int(x.numel()))))

    out_dir.mkdir(parents=True, exist_ok=True)
    Phi_cpu = Phi.detach().cpu()
    squared = (Phi_cpu.conj() * Phi_cpu).real if Phi_cpu.is_complex() else Phi_cpu ** 2
    col_energy = squared.sum(dim=0).to(torch.float64)
    row_energy = squared.sum(dim=1).to(torch.float64)
    off = abs_gram[~torch.eye(abs_gram.shape[0], dtype=torch.bool)].flatten()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    draw_hist(axes[0, 0], col_energy, 40)
    axes[0, 0].set_title("Codeword energy")
    axes[0, 1].plot(row_energy.numpy())
    axes[0, 1].set_title("Resource row energy")
    draw_hist(axes[1, 0], off, 60)
    axes[1, 0].set_title("Off-diagonal coherence")
    axes[1, 1].semilogy(singular_values.detach().cpu().numpy(), marker="o", linewidth=1)
    axes[1, 1].set_title("Singular spectrum")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "encoding_summary.png", dpi=150)
    plt.close(fig)

    values = Phi_cpu.real.flatten().to(torch.float64)
    abs_values = Phi_cpu.abs().flatten().to(torch.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    draw_hist(axes[0], values, 80)
    axes[0].set_title("Real codebook entries")
    draw_hist(axes[1], abs_values, 80)
    axes[1].set_title("Absolute codebook entries")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "value_distribution.png", dpi=150)
    plt.close(fig)


def plot_component_analysis(encoder: Encoder, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for i, comp in enumerate(encoder.components):
        R_support = comp.materialize_operator_bank().detach().abs() > 1e-9
        if R_support.shape[0] > 1:
            flat_support = R_support.reshape(R_support.shape[0], -1).to(torch.float64)
            overlap = flat_support @ flat_support.T
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(overlap.cpu().numpy(), aspect="auto")
            ax.set_title(f"Component {i} operator support overlap")
            ax.set_xlabel("operator q")
            ax.set_ylabel("operator q")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"component_{i}_operator_overlap.png", dpi=150)
            plt.close(fig)

        C = comp.C.detach()
        if C.shape[1] > 1:
            C_norm = C / C.norm(dim=0, keepdim=True).clamp_min(1e-12)
            G = (C_norm.conj().T @ C_norm if C_norm.is_complex() else C_norm.T @ C_norm).abs()
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(G.cpu().numpy(), aspect="auto", vmin=0)
            ax.set_title(f"Component {i} local C coherence")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"component_{i}_C_coherence.png", dpi=150)
            plt.close(fig)


def plot_matrix_comparison(payload: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [payload["global"], *payload["matrices"].values()]
    names = [r["name"] for r in rows]
    sparsity = [r.get("entry_near_zero_fraction", float("nan")) for r in rows]
    row_gini = [r.get("row_energy_gini", float("nan")) for r in rows]
    coh_mean = [r.get("coherence", {}).get("mean_abs", float("nan")) for r in rows]
    coh_q99 = [r.get("coherence", {}).get("q99_abs", float("nan")) for r in rows]
    coh_max = [r.get("coherence", {}).get("max_abs", float("nan")) for r in rows]
    eff_rank = [
        r.get("singular_values", {}).get("effective_rank", float("nan")) / max(1, min(
            r.get("shape", {}).get("n", 1), r.get("shape", {}).get("M", 1)))
        for r in rows
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    x = range(len(names))
    series = [
        ("Near-zero fraction", sparsity),
        ("Row-energy Gini", row_gini),
        ("Mean |Gram offdiag|", coh_mean),
        ("q99 |Gram offdiag|", coh_q99),
        ("Max |Gram offdiag|", coh_max),
        ("Effective-rank fraction", eff_rank),
    ]
    for ax, (title, vals) in zip(axes.flat, series):
        ax.bar(x, vals)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "matrix_structure_comparison.png", dpi=150)
    plt.close(fig)


def analyze_encoder(encoder: Encoder, out_dir: Path | str, *,
                    active_k: int | None = None,
                    num_active_samples: int = DEFAULT_ACTIVE_SET_SAMPLES) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        Phi = encoder.explicit_matrix().detach()
        abs_gram, coherence = codeword_coherence(Phi)
        singular_values = torch.linalg.svdvals(Phi).detach().cpu().to(torch.float64)
        matrices = interesting_matrices(encoder)
        payload = {
            "global": matrix_analysis("global", Phi, active_k=active_k,
                                      num_active_samples=num_active_samples,
                                      include_recovery=active_k is not None),
            "matrices": {name: matrix_analysis(name, matrix, include_gaussian_reference=(matrix.numel() <= MAX_EXACT_SVD_NUMEL))
                         for name, matrix in matrices.items() if name != "global"},
            "components": component_analysis(encoder),
        }
        (out_dir / "encoding_analysis.json").write_text(json.dumps(payload, indent=2, default=str))
        plot_encoding_analysis(Phi, abs_gram, singular_values, out_dir)
        plot_component_analysis(encoder, out_dir)
        plot_matrix_comparison(payload, out_dir)
    return payload
