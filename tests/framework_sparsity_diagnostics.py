"""Bounded-memory diagnostics for explicit sparse-global codebooks."""

from __future__ import annotations

import math

import numpy as np
import torch


def log2_binomial(n: int, s: int) -> float:
    if s < 0 or s > n:
        raise ValueError(f"s must lie in [0,{n}], got {s}")
    return (math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1)) / math.log(2.0)


def _packed_pattern_counts(mask: torch.Tensor) -> np.ndarray:
    packed = np.packbits(mask.transpose(0, 1).contiguous().numpy(), axis=1)
    _, counts = np.unique(packed, axis=0, return_counts=True)
    return counts


def _sample_pair_geometry(Phi: torch.Tensor, support: torch.Tensor, num_pairs: int,
                          generator: torch.Generator) -> dict:
    if num_pairs <= 0:
        return {}
    M = Phi.shape[1]
    left = torch.randint(M, (num_pairs,), generator=generator)
    right = torch.randint(M - 1, (num_pairs,), generator=generator)
    right += (right >= left).long()
    overlap_parts, correlation_parts = [], []
    for start in range(0, num_pairs, 2048):
        i = left[start:start + 2048]; j = right[start:start + 2048]
        overlap_parts.append((support[:, i] & support[:, j]).sum(dim=0).to(torch.float32))
        inner = torch.sum(Phi[:, i].conj() * Phi[:, j], dim=0)
        correlation_parts.append(torch.abs(inner).to(torch.float32))
    overlaps = torch.cat(overlap_parts); correlations = torch.cat(correlation_parts)
    return {
        "sampled_pair_count": int(num_pairs),
        "sampled_support_overlap_mean": float(overlaps.mean()),
        "sampled_support_overlap_q95": float(torch.quantile(overlaps, 0.95)),
        "sampled_abs_correlation_mean": float(correlations.mean()),
        "sampled_abs_correlation_q95": float(torch.quantile(correlations, 0.95)),
        "sampled_abs_correlation_q99": float(torch.quantile(correlations, 0.99)),
        "sampled_abs_correlation_q999": float(torch.quantile(correlations, 0.999)),
        "sampled_abs_correlation_max": float(correlations.max()),
        "sampled_nonzero_correlation_fraction": float((correlations > 1e-8).to(torch.float32).mean()),
    }


def _sample_active_geometry(support: torch.Tensor, K: int, num_samples: int,
                            generator: torch.Generator) -> dict:
    if num_samples <= 0:
        return {}
    n, M = support.shape
    if K <= 0 or K > M:
        raise ValueError(f"active-set K must lie in [1,{M}], got {K}")
    occupied, collided_rows, singleton_rows, reused = [], [], [], []
    for _ in range(num_samples):
        messages = torch.randperm(M, generator=generator)[:K]
        row_multiplicity = support[:, messages].sum(dim=1)
        occupied.append(float((row_multiplicity > 0).sum()))
        collided_rows.append(float((row_multiplicity > 1).sum()))
        singleton_rows.append(float((row_multiplicity == 1).sum()))
        reused.append(float((row_multiplicity - 1).clamp_min(0).sum()))
    placements = K * float(support.sum(dim=0).to(torch.float32).mean())
    return {
        "active_geometry_samples": int(num_samples),
        "active_geometry_K": int(K),
        "active_occupied_rows_mean": float(np.mean(occupied)),
        "active_occupied_fraction_mean": float(np.mean(occupied) / n),
        "active_expansion_ratio_mean": float(np.mean(occupied) / K),
        "active_collided_rows_mean": float(np.mean(collided_rows)),
        "active_singleton_rows_mean": float(np.mean(singleton_rows)),
        "active_singleton_placement_fraction_mean": float(np.mean(singleton_rows) / max(placements, 1.0)),
        "active_reused_placement_fraction_mean": float(np.mean(reused) / max(placements, 1.0)),
        "active_messages_sampled_without_replacement": True,
    }


def _sample_active_gram(Phi: torch.Tensor, K: int, num_samples: int, generator: torch.Generator) -> dict:
    if num_samples <= 0:
        return {}
    M = Phi.shape[1]
    if K <= 0 or K > M:
        raise ValueError(f"active-set K must lie in [1,{M}], got {K}")
    min_eigenvalues, max_eigenvalues, conditions = [], [], []
    for _ in range(num_samples):
        messages = torch.randperm(M, generator=generator)[:K]
        active = Phi[:, messages]
        gram = active.conj().transpose(0, 1) @ active
        eigenvalues = torch.linalg.eigvalsh(gram).real.to(torch.float64).clamp_min(0.0)
        minimum, maximum = float(eigenvalues.min()), float(eigenvalues.max())
        min_eigenvalues.append(minimum); max_eigenvalues.append(maximum)
        conditions.append(maximum / max(minimum, 1e-12))
    return {
        "active_gram_samples": int(num_samples), "active_gram_K": int(K),
        "active_gram_min_eigenvalue_mean": float(np.mean(min_eigenvalues)),
        "active_gram_min_eigenvalue_q05": float(np.quantile(min_eigenvalues, 0.05)),
        "active_gram_min_eigenvalue_min": float(np.min(min_eigenvalues)),
        "active_gram_max_eigenvalue_mean": float(np.mean(max_eigenvalues)),
        "active_gram_condition_median": float(np.median(conditions)),
        "active_gram_condition_q95": float(np.quantile(conditions, 0.95)),
        "active_gram_condition_max": float(np.max(conditions)),
    }


def _sample_sum_separation(Phi: torch.Tensor, K: int, num_pairs: int, generator: torch.Generator) -> dict:
    if num_pairs <= 0:
        return {}
    M = Phi.shape[1]
    if K <= 0 or 2 * K > M:
        raise ValueError(f"disjoint K-sum sampling requires 1 <= 2K <= M, got K={K}, M={M}")
    separations = []
    for _ in range(num_pairs):
        messages = torch.randperm(M, generator=generator)[:2 * K]
        difference = Phi[:, messages[:K]].sum(dim=1) - Phi[:, messages[K:]].sum(dim=1)
        separations.append(float(torch.linalg.vector_norm(difference) / math.sqrt(2.0 * K)))
    values = np.asarray(separations)
    return {
        "sum_separation_pairs": int(num_pairs), "sum_separation_K": int(K),
        "normalised_disjoint_K_sum_distance_mean": float(values.mean()),
        "normalised_disjoint_K_sum_distance_q05": float(np.quantile(values, 0.05)),
        "normalised_disjoint_K_sum_distance_min": float(values.min()),
    }


def analyse_encoder_sparsity(encoder, num_pairs: int = 0, active_samples: int = 0,
                             active_k: int | None = None, active_gram_samples: int = 0,
                             sum_pair_samples: int = 0, seed: int = 0) -> dict:
    """Materialise only ``Phi`` and compute support/correlation diagnostics, never its Gram matrix."""
    Phi = encoder.explicit_matrix().detach().cpu()
    support = torch.abs(Phi) > 0
    n, M = Phi.shape
    support_sizes = support.sum(dim=0)
    if int(support_sizes.min()) != int(support_sizes.max()):
        raise ValueError("density sweep expects one exact support size per codeword")
    s = int(support_sizes[0])
    energies = torch.sum(torch.abs(Phi) ** 2, dim=0).real.to(torch.float64)
    row_load = support.sum(dim=1).to(torch.float64)
    row_energy = torch.sum(torch.abs(Phi) ** 2, dim=1).real.to(torch.float64)
    support_counts = _packed_pattern_counts(support)
    signed_counts = None
    if not Phi.is_complex():
        signed_support = support & (Phi > 0)
        signed_signature = torch.cat((support, signed_support), dim=0)
        signed_counts = _packed_pattern_counts(signed_signature)
    log2_masks = log2_binomial(n, s)
    diagnostics = {
        "n": int(n), "M": int(M), "support_size": s, "nonzero_fraction": float(s / n),
        "log2_available_support_masks": float(log2_masks),
        "support_masks_sufficient_for_M": bool(log2_masks + 1e-10 >= math.log2(M)),
        "low_density_support_mask_shortfall": bool(s <= n // 2 and log2_masks + 1e-10 < math.log2(M)),
        "distinct_support_masks": int(support_counts.size),
        "support_mask_duplicate_fraction": float(1.0 - support_counts.size / M),
        "largest_support_mask_multiplicity": int(support_counts.max()),
        "distinct_signed_support_patterns": int(signed_counts.size) if signed_counts is not None else None,
        "signed_support_duplicate_fraction": float(1.0 - signed_counts.size / M) if signed_counts is not None else None,
        "largest_signed_support_multiplicity": int(signed_counts.max()) if signed_counts is not None else None,
        "column_energy_min": float(energies.min()), "column_energy_max": float(energies.max()),
        "max_unit_energy_deviation": float(torch.max(torch.abs(energies - 1.0))),
        "row_nonzero_load_mean": float(row_load.mean()),
        "row_nonzero_load_cv": float(row_load.std(unbiased=False) / row_load.mean().clamp_min(1e-12)),
        "row_energy_cv": float(row_energy.std(unbiased=False) / row_energy.mean().clamp_min(1e-12)),
        "independent_support_pair_overlap_expectation": float(s * s / n),
        "real_s1_max_distinct_unit_directions": int(2 * n) if not Phi.is_complex() and s == 1 else None,
        "diagnostics_materialised_phi": True, "diagnostics_materialised_gram": False,
    }
    generator = torch.Generator().manual_seed(int(seed))
    diagnostics.update(_sample_pair_geometry(Phi, support, int(num_pairs), generator))
    diagnostics.update(_sample_active_geometry(support, int(active_k), int(active_samples), generator)
                       if active_k is not None else {})
    diagnostics.update(_sample_active_gram(Phi, int(active_k), int(active_gram_samples), generator)
                       if active_k is not None else {})
    diagnostics.update(_sample_sum_separation(Phi, int(active_k), int(sum_pair_samples), generator)
                       if active_k is not None else {})
    return diagnostics
