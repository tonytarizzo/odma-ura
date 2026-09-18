"""Coordinate-wise amplitude sharing on fixed affine hash supports.

Each table t looks up V[t, P_t w]. Normalisation is across the gathered entries
of a message, never across a stored bank column. Only V is gradient-trained.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .core import URASpec
from .hash_skeleton import (all_message_bits, gf2_rank, hash_skeleton_rows, linear_hash_bins,
                            materialize_sparse_codebook, random_linear_hash_bank)
from .initializers import nonzero_gaussian
from .prototype_amplitudes import ProceduralHashCodebook, PrototypeHashEncoder, _projection


def coordinate_projections(A: torch.Tensor, label_bits: int, seed: int) -> torch.Tensor:
    """Nested prefixes, independent of the local hash up to the available B bits.

Each table first completes A_t to a basis, then appends A_t. A retry is needed
only if the stacked prefixes lack the maximum possible rank. The full sequence
is certified once, so different J calls with the same seed remain nested.
"""
    T, r, B = A.shape
    if not 0 <= label_bits <= min(B, 62):
        raise ValueError("label_bits must lie in [0, min(B,62)]")
    generator = torch.Generator().manual_seed(int(seed) + 610_013)
    for _ in range(100):
        bases = []
        for table in range(T):
            if gf2_rank(A[table]) != r:
                raise ValueError("every support hash must have full row rank")
            basis = A[table].clone()
            extra = []
            for _ in range(10_000):
                if len(extra) == B - r:
                    break
                row = torch.randint(2, (1, B), dtype=torch.uint8, generator=generator)
                candidate = torch.cat((basis, row))
                if gf2_rank(candidate) > basis.shape[0]:
                    basis = candidate; extra.append(row)
            if len(extra) != B - r:
                raise RuntimeError("failed to complete a binary basis")
            tail = A[table] if extra else A[table][torch.randperm(r, generator=generator)]
            bases.append(torch.cat((*extra, tail)))
        Q = torch.stack(bases)
        # Once full rank is reached, longer prefixes remain full rank.
        if all(gf2_rank(Q[:, :j].reshape(-1, B)) == min(T * j, B)
               for j in range(1, min(B, math.ceil(B / T)) + 1)):
            return Q[:, :label_bits].clone()
    raise RuntimeError("failed to construct maximum-rank stacked amplitude labels")


def balance_rows(values: torch.Tensor) -> torch.Tensor:
    """Initialisation only: zero mean and unit RMS within each table's bank."""
    if values.shape[1] < 2:
        raise ValueError("centering requires at least two labels")
    centered = values - values.mean(dim=1, keepdim=True)
    rms = centered.square().mean(dim=1, keepdim=True).sqrt()
    if bool((rms == 0).any()):
        raise ValueError("cannot balance a constant bank row")
    return centered / rms


class CoordinateAmplitudeBank(nn.Module):
    def __init__(self, projection: torch.Tensor, values: torch.Tensor, learnable: bool, target_energy: float = 1.0):
        super().__init__()
        T, J, B = projection.shape
        if values.shape != (T, 1 << J) or target_energy <= 0 or not values.is_floating_point():
            raise ValueError("expected real V=(T,2^J) and positive target energy")
        self.register_buffer("projection", projection.to(torch.uint8).clone())
        if learnable:
            self.prototypes = nn.Parameter(values.clone())
        else:
            self.register_buffer("prototypes", values.clone())
        self.target_energy = float(target_energy)

    @property
    def tables(self): return int(self.projection.shape[0])

    @property
    def label_bits(self): return int(self.projection.shape[1])

    @property
    def payload_bits(self): return int(self.projection.shape[2])

    def labels(self, message_bits: torch.Tensor) -> torch.Tensor:
        offsets = torch.zeros(self.tables, self.label_bits, dtype=torch.uint8, device=self.projection.device)
        return linear_hash_bins(self.projection, offsets, message_bits.to(self.projection.device))

    def from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        raw = self.prototypes.gather(1, labels)
        norms = raw.norm(dim=0, keepdim=True)
        nonzero = norms > 0
        normalized = raw / torch.where(nonzero, norms, torch.ones_like(norms))
        # Define even the exceptional all-zero vector to have unit energy.
        fallback = torch.zeros_like(raw); fallback[0] = 1
        return math.sqrt(self.target_energy) * torch.where(nonzero, normalized, fallback)

    def forward(self, message_bits: torch.Tensor) -> torch.Tensor:
        return self.from_labels(self.labels(message_bits))

    def apply_constraint(self) -> None:
        # A stored V column is NOT a codeword: all energy constraints live in forward().
        pass


class CoordinateHashEncoder(PrototypeHashEncoder):
    """Small-B certification adapter; only this adapter allocates message lookups."""

    def _all_amplitudes(self) -> torch.Tensor:
        return self.codebook.amplitude_bank.from_labels(self._message_labels)

    def message_columns(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        rows = self._message_rows.index_select(1, indices)
        amplitudes = self.codebook.amplitude_bank.from_labels(self._message_labels.index_select(1, indices))
        return materialize_sparse_codebook(rows, amplitudes, self.n)

    def spectral_norm_squared(self, num_iters=20, generator=None, use_cache=True):
        # Rebuilding a learned operator must not inject random calibration noise into validation.
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(730_019)
        return super().spectral_norm_squared(num_iters, generator, use_cache)

    def prototype_diagnostics(self) -> dict:
        with torch.no_grad():
            V = self.codebook.amplitude_bank.prototypes
            raw = V.gather(1, self._message_labels)
            energy = self._all_amplitudes().square()
            table_energy = energy.mean(dim=1)
            return {
                "prototype_shape": list(V.shape), "prototype_parameters": V.numel(),
                "raw_amplitude_rank": int(torch.linalg.matrix_rank(raw.to(torch.float64))),
                "table_energy_cv": float(table_energy.std(unbiased=False) / table_energy.mean()),
                "bank_row_mean_rms": float(V.mean(dim=1).square().mean().sqrt()),
                "bank_row_rms_cv": float(V.square().mean(dim=1).sqrt().std(unbiased=False)
                                         / V.square().mean(dim=1).sqrt().mean().clamp_min(1e-12)),
                "effective_coordinates_mean": float((self.spec.energy_per_codeword ** 2
                                                      / energy.square().sum(dim=0)).mean()),
                "max_energy_deviation": float((energy.sum(dim=0) - self.spec.energy_per_codeword).abs().max()),
            }


def _make_bank(A, label_bits, seed, learnable, balanced, shared, target_energy=1.0):
    T, _, B = A.shape
    P = (_projection(B, label_bits, seed).unsqueeze(0).expand(T, -1, -1).clone() if shared
         else coordinate_projections(A, label_bits, seed))
    generator = torch.Generator().manual_seed(int(seed) + 410_009)
    values = nonzero_gaussian((T, 1 << label_bits), torch.float32, generator)
    bank = CoordinateAmplitudeBank(P, values, learnable, target_energy)
    if label_bits == B:
        # Pair the physical Gaussian columns with the historical unrestricted endpoint.
        labels = bank.labels(all_message_bits(B))
        values = torch.empty_like(values).scatter_(1, labels, values)
    if balanced:
        values = balance_rows(values)
    with torch.no_grad():
        bank.prototypes.copy_(values)
    return bank


def build_coordinate_hash_encoder(spec: URASpec, support_size: int, seed: int, label_bits: int,
                                  learn_amplitudes: bool, balanced: bool = True, shared: bool = False,
                                  search_candidates: int = 128) -> CoordinateHashEncoder:
    if spec.payload_bits > 20:
        raise ValueError("global D0/D1 certification is restricted to B<=20; use the procedural builder for larger B")
    _, support = hash_skeleton_rows(spec, "hash_linear_selected_fixed", support_size, seed, search_candidates)
    A, b = torch.tensor(support["A"], dtype=torch.uint8), torch.tensor(support["b"], dtype=torch.uint8)
    bank = _make_bank(A, label_bits, seed, learn_amplitudes, balanced, shared, spec.energy_per_codeword)
    P = bank.projection
    support.update({
        "amplitude_family": "shared_labels_control" if shared else "coordinate_labels",
        "amplitude_label_bits": label_bits, "balanced_initialization": balanced,
        "projection_shape": list(P.shape), "projection": P.tolist(),
        "projection_ranks": [gf2_rank(p) for p in P],
        "stacked_projection_rank": gf2_rank(P.reshape(-1, spec.payload_bits)),
        "local_joint_ranks": [gf2_rank(torch.cat((a, p))) for a, p in zip(A, P)],
        "prototype_shape": list(bank.prototypes.shape), "prototype_storage_values": bank.prototypes.numel(),
        "learnable_amplitudes": learn_amplitudes, "discrete_maps_learnable": False,
        "exact_column_energy": True, "saved_generator_has_global_message_axis": False,
        "compact_forward_support_and_amplitudes": True, "scalable_claim_applies_to_support_rule_only": False,
        "current_D0_D1_runtime_has_global_message_axis": True,
        "normalization": "gather_then_normalize_each_message; no bank-column projection",
    })
    return CoordinateHashEncoder(spec, ProceduralHashCodebook(A, b, bank, spec.n), support)


def build_random_coordinate_codebook(payload_bits: int, n: int, support_size: int, label_bits: int,
                                     seed: int, learn_amplitudes: bool = False) -> ProceduralHashCodebook:
    B, T = int(payload_bits), int(support_size)
    if T <= 0 or n % T or n // T < 2 or (n // T) & ((n // T) - 1):
        raise ValueError("require positive T and n/T=2^r>=2")
    A, b = random_linear_hash_bank(B, T, (n // T).bit_length() - 1,
                                   torch.Generator().manual_seed(int(seed) + 310_003))
    bank = _make_bank(A, label_bits, seed, learn_amplitudes, True, False)
    return ProceduralHashCodebook(A, b, bank, n)
