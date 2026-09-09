"""Compact amplitude prototypes on fixed affine-hash supports.

For message bits ``w``, table ``t`` selects row ``t R + h_t(w)`` and uses
amplitude ``V[t, P_J w]``.  The discrete maps ``A,b,P_J`` are fixed; only the
prototype bank ``V`` may be trained.  Normalising every prototype column gives
unit-energy codewords for every possible message, including unseen messages.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .core import URASpec
from .hash_skeleton import (all_message_bits, gf2_rank, hash_skeleton_rows, linear_hash_bins,
                            materialize_sparse_codebook, random_linear_hash_bank)
from .initializers import nonzero_gaussian


AMPLITUDE_INITIALIZATIONS = ("gaussian", "equal", "rademacher")


def random_invertible_binary_matrix(payload_bits: int, generator: torch.Generator) -> torch.Tensor:
    """Sample ``Q in GF(2)^(B x B)``; its first J rows define every nested ``P_J``."""
    B = int(payload_bits)
    if B <= 0:
        raise ValueError(f"payload_bits must be positive, got {B}")
    for _ in range(10_000):
        Q = torch.randint(2, (B, B), generator=generator, dtype=torch.uint8)
        if gf2_rank(Q) == B:
            return Q
    raise RuntimeError("could not sample an invertible binary projection matrix")


def amplitude_labels(P: torch.Tensor, message_bits: torch.Tensor) -> torch.Tensor:
    """Return integer labels ``P w`` with least-significant output bit first."""
    P = torch.as_tensor(P, dtype=torch.uint8)
    message_bits = torch.as_tensor(message_bits, dtype=torch.uint8)
    if P.ndim != 2 or message_bits.ndim != 2 or message_bits.shape[1] != P.shape[1]:
        raise ValueError(f"expected P=(J,B), message_bits=(K,B); got {tuple(P.shape)}, {tuple(message_bits.shape)}")
    if P.shape[0] == 0:
        return torch.zeros(message_bits.shape[0], dtype=torch.long, device=message_bits.device)
    bits = torch.einsum("jb,kb->jk", P.to(device=message_bits.device, dtype=torch.int64),
                        message_bits.to(torch.int64)).remainder_(2)
    weights = (1 << torch.arange(P.shape[0], dtype=torch.int64, device=message_bits.device)).unsqueeze(1)
    return torch.sum(bits * weights, dim=0).long()


def _initial_prototypes(tables: int, num_labels: int, kind: str, dtype: torch.dtype,
                        generator: torch.Generator) -> torch.Tensor:
    if kind not in AMPLITUDE_INITIALIZATIONS:
        raise ValueError(f"unknown amplitude initialization '{kind}'")
    if kind == "equal":
        return torch.ones(tables, num_labels, dtype=dtype)
    if kind == "rademacher":
        return (2 * torch.randint(2, (tables, num_labels), generator=generator) - 1).to(dtype)
    return nonzero_gaussian((tables, num_labels), dtype, generator)


class PrototypeAmplitudeBank(nn.Module):
    """The compact map ``w -> normalise(V[:, P_J w])``."""

    def __init__(self, projection: torch.Tensor, prototypes: torch.Tensor, learnable: bool,
                 target_energy: float = 1.0) -> None:
        super().__init__()
        projection = torch.as_tensor(projection, dtype=torch.uint8)
        prototypes = torch.as_tensor(prototypes)
        if projection.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("projection and prototypes must be matrices")
        if prototypes.shape[1] != 1 << projection.shape[0]:
            raise ValueError(f"V must have 2^J columns, got J={projection.shape[0]}, V={tuple(prototypes.shape)}")
        if target_energy <= 0.0:
            raise ValueError(f"target_energy must be positive, got {target_energy}")
        self.register_buffer("projection", projection.clone())
        if learnable:
            self.prototypes = nn.Parameter(prototypes.clone())
        else:
            self.register_buffer("prototypes", prototypes.clone())
        self.target_energy = float(target_energy)
        self.apply_constraint()

    @property
    def label_bits(self) -> int: return int(self.projection.shape[0])

    @property
    def payload_bits(self) -> int: return int(self.projection.shape[1])

    @property
    def tables(self) -> int: return int(self.prototypes.shape[0])

    @property
    def num_labels(self) -> int: return int(self.prototypes.shape[1])

    @property
    def learnable(self) -> bool: return isinstance(self.prototypes, nn.Parameter)

    def normalized_prototypes(self) -> torch.Tensor:
        scale = math.sqrt(self.target_energy)
        norms = self.prototypes.norm(dim=0, keepdim=True)
        normalized = self.prototypes / norms.clamp_min(1e-12)
        if bool((norms <= 1e-12).any()):
            fallback = torch.zeros_like(normalized); fallback[0] = 1.0
            normalized = torch.where((norms <= 1e-12).expand_as(normalized), fallback, normalized)
        return scale * normalized

    def labels(self, message_bits: torch.Tensor) -> torch.Tensor:
        return amplitude_labels(self.projection, message_bits)

    def forward(self, message_bits: torch.Tensor) -> torch.Tensor:
        return self.normalized_prototypes().index_select(1, self.labels(message_bits))

    def apply_constraint(self) -> None:
        with torch.no_grad():
            self.prototypes.copy_(self.normalized_prototypes())

    def diagnostics(self) -> dict:
        with torch.no_grad():
            V = self.normalized_prototypes()
            energy = V.abs().square()
            table_mean = energy.mean(dim=1)
            effective = self.target_energy ** 2 / energy.square().sum(dim=0).clamp_min(1e-12)
            return {
                "prototype_shape": list(V.shape), "prototype_parameters": int(V.numel()),
                "table_energy_cv": float(table_mean.std(unbiased=False) / table_mean.mean().clamp_min(1e-12)),
                "effective_coordinates_mean": float(effective.mean()),
                "effective_coordinates_min": float(effective.min()),
                "max_energy_deviation": float((energy.sum(dim=0) - self.target_energy).abs().max()),
            }


class ProceduralHashCodebook(nn.Module):
    """M-free forward generator storing only ``A,b,P,V``."""

    def __init__(self, A: torch.Tensor, b: torch.Tensor, amplitude_bank: PrototypeAmplitudeBank,
                 n: int) -> None:
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.uint8)
        b = torch.as_tensor(b, dtype=torch.uint8)
        if A.ndim != 3:
            raise ValueError(f"A must have shape (T,r,B), got {tuple(A.shape)}")
        T, r, B = A.shape
        if b.shape != (T, r) or amplitude_bank.tables != T or amplitude_bank.payload_bits != B:
            raise ValueError("support hashes and amplitude bank have incompatible shapes")
        if n != T * (1 << r):
            raise ValueError(f"one-table-row layout requires n=T*2^r, got n={n}, T={T}, r={r}")
        self.register_buffer("A", A.clone())
        self.register_buffer("b", b.clone())
        self.amplitude_bank = amplitude_bank
        self._n = int(n)

    @property
    def n(self) -> int: return self._n

    @property
    def tables(self) -> int: return int(self.A.shape[0])

    @property
    def bins_per_table(self) -> int: return 1 << int(self.A.shape[1])

    @property
    def payload_bits(self) -> int: return int(self.A.shape[2])

    @property
    def dtype(self) -> torch.dtype: return self.amplitude_bank.prototypes.dtype

    @property
    def device(self) -> torch.device: return self.amplitude_bank.prototypes.device

    def support_rows(self, message_bits: torch.Tensor) -> torch.Tensor:
        bits = message_bits.to(device=self.device, dtype=torch.uint8)
        bins = linear_hash_bins(self.A.to(self.device), self.b.to(self.device), bits)
        offsets = torch.arange(self.tables, dtype=torch.long, device=self.device).unsqueeze(1) * self.bins_per_table
        return offsets + bins

    def codewords(self, message_bits: torch.Tensor) -> torch.Tensor:
        bits = message_bits.to(device=self.device, dtype=torch.uint8)
        return materialize_sparse_codebook(self.support_rows(bits), self.amplitude_bank(bits), self.n)

    def apply_constraints(self) -> None:
        self.amplitude_bank.apply_constraint()


class PrototypeHashEncoder(nn.Module):
    """Small-B global interface for D0/D1, backed by a compact procedural codebook.

    The length-M row and label tensors are non-persistent runtime aids required
    by the current global decoders.  They are not part of the saved generator.
    """

    def __init__(self, spec: URASpec, codebook: ProceduralHashCodebook, construction_metadata: dict) -> None:
        super().__init__()
        if spec.n != codebook.n or spec.payload_bits != codebook.payload_bits or spec.num_codewords != 1 << spec.payload_bits:
            raise ValueError("PrototypeHashEncoder requires n-compatible codebook and M=2^B")
        self.spec = spec
        self.codebook = codebook
        bits = all_message_bits(spec.payload_bits).to(codebook.device)
        self.register_buffer("_message_rows", codebook.support_rows(bits), persistent=False)
        self.register_buffer("_message_labels", codebook.amplitude_bank.labels(bits), persistent=False)
        self.construction_metadata = construction_metadata
        self._spectral_cache: dict[int, torch.Tensor] = {}

    @property
    def n(self) -> int: return self.spec.n

    @property
    def num_codewords(self) -> int: return self.spec.num_codewords

    @property
    def dtype(self) -> torch.dtype: return self.codebook.dtype

    @property
    def device(self) -> torch.device: return self.codebook.device

    def _all_amplitudes(self) -> torch.Tensor:
        return self.codebook.amplitude_bank.normalized_prototypes().index_select(1, self._message_labels)

    def message_columns(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        rows = self._message_rows.index_select(1, indices)
        amplitudes = self._all_amplitudes().index_select(1, indices)
        return materialize_sparse_codebook(rows, amplitudes, self.n)

    def explicit_matrix(self) -> torch.Tensor:
        return materialize_sparse_codebook(self._message_rows, self._all_amplitudes(), self.n)

    def matvec(self, a: torch.Tensor) -> torch.Tensor:
        if a.ndim not in (1, 2):
            raise ValueError(f"a must have shape (M,) or (batch,M), got {tuple(a.shape)}")
        squeezed = a.ndim == 1
        a_batch = a.unsqueeze(0) if squeezed else a
        if a_batch.shape[1] != self.num_codewords:
            raise ValueError(f"a must have M={self.num_codewords} columns, got {tuple(a.shape)}")
        values = a_batch.to(self.dtype).unsqueeze(1) * self._all_amplitudes().unsqueeze(0)
        out = torch.zeros(a_batch.shape[0], self.n, dtype=self.dtype, device=self.device)
        for table in range(self.codebook.tables):
            rows = self._message_rows[table].unsqueeze(0).expand(a_batch.shape[0], -1)
            out.scatter_add_(1, rows, values[:, table])
        return out.squeeze(0) if squeezed else out

    def rmatvec(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim not in (1, 2):
            raise ValueError(f"residual must have shape (n,) or (batch,n), got {tuple(residual.shape)}")
        squeezed = residual.ndim == 1
        residual_batch = residual.unsqueeze(0) if squeezed else residual
        if residual_batch.shape[1] != self.n:
            raise ValueError(f"residual must have n={self.n} columns, got {tuple(residual.shape)}")
        amplitudes = self._all_amplitudes().conj() if self.dtype.is_complex else self._all_amplitudes()
        out = torch.zeros(residual_batch.shape[0], self.num_codewords, dtype=self.dtype, device=self.device)
        for table in range(self.codebook.tables):
            out += residual_batch.to(self.dtype).index_select(1, self._message_rows[table]) * amplitudes[table]
        return out.squeeze(0) if squeezed else out

    def encode(self, counts: torch.Tensor) -> torch.Tensor:
        return self.matvec(counts)

    def message_factor_ids(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.num_codewords, dtype=torch.long, device=self.device),
                torch.arange(self.num_codewords, dtype=torch.long, device=self.device))

    def apply_constraints(self) -> None:
        self.codebook.apply_constraints()
        self._spectral_cache.clear()

    def mean_codeword_energy(self, chunk_size: int = 256, use_cache: bool = True) -> float:
        return float(self.spec.energy_per_codeword)

    def spectral_norm_squared(self, num_iters: int = 20, generator: torch.Generator | None = None,
                              use_cache: bool = True) -> torch.Tensor:
        if num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {num_iters}")
        if use_cache and int(num_iters) in self._spectral_cache:
            return self._spectral_cache[int(num_iters)]
        real_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64 if self.dtype == torch.complex128 else self.dtype
        x = torch.randn(self.num_codewords, dtype=real_dtype, device=self.device, generator=generator)
        x = x / x.norm().clamp_min(1e-12)
        with torch.no_grad():
            for _ in range(int(num_iters)):
                x = self.rmatvec(self.matvec(x).to(self.dtype)).real
                x = x / x.norm().clamp_min(1e-12)
            value = self.matvec(x.to(self.dtype)).abs().square().sum().real.clamp_min(1e-12)
        if use_cache:
            self._spectral_cache[int(num_iters)] = value
        return value

    def prototype_diagnostics(self) -> dict:
        return self.codebook.amplitude_bank.diagnostics()


def _projection(payload_bits: int, label_bits: int, seed: int) -> torch.Tensor:
    B, J = int(payload_bits), int(label_bits)
    if J < 0 or J > B:
        raise ValueError(f"amplitude label bits J must lie in [0,{B}], got {J}")
    generator = torch.Generator().manual_seed(int(seed) + 510_011)
    return random_invertible_binary_matrix(B, generator)[:J]


def _prototype_values(tables: int, payload_bits: int, projection: torch.Tensor, seed: int,
                      kind: str, dtype: torch.dtype) -> torch.Tensor:
    J = int(projection.shape[0])
    generator = torch.Generator().manual_seed(int(seed) + 410_009)
    if kind == "equal" and J != 0:
        raise ValueError("the equal-amplitude control is defined at J=0")
    if kind == "gaussian" and J == payload_bits:
        per_message = nonzero_gaussian((tables, 1 << payload_bits), dtype, generator)
        per_message /= per_message.norm(dim=0, keepdim=True).clamp_min(1e-12)
        labels = amplitude_labels(projection, all_message_bits(payload_bits))
        values = torch.empty_like(per_message)
        values[:, labels] = per_message
        return values
    return _initial_prototypes(tables, 1 << J, kind, dtype, generator)


def build_prototype_hash_encoder(spec: URASpec, support_size: int, seed: int, label_bits: int,
                                 learn_amplitudes: bool, amplitude_init: str = "gaussian",
                                 search_candidates: int = 128) -> PrototypeHashEncoder:
    """Build the exact small-B selected-hash reference used by D0/D1 experiments."""
    _, support = hash_skeleton_rows(spec, "hash_linear_selected_fixed", support_size, seed, search_candidates)
    A = torch.tensor(support["A"], dtype=torch.uint8)
    b = torch.tensor(support["b"], dtype=torch.uint8)
    P = _projection(spec.payload_bits, label_bits, seed)
    V = _prototype_values(support_size, spec.payload_bits, P, seed, amplitude_init, torch.float32)
    bank = PrototypeAmplitudeBank(P, V, learn_amplitudes, spec.energy_per_codeword)
    codebook = ProceduralHashCodebook(A, b, bank, spec.n)
    support.update({
        "amplitude_family": "nested_linear_label_prototypes", "amplitude_initialization": amplitude_init,
        "amplitude_label_bits": int(label_bits), "amplitude_labels": int(1 << label_bits),
        "projection_shape": list(P.shape), "projection_rank": gf2_rank(P), "projection": P.tolist(),
        "combined_support_projection_rank": gf2_rank(torch.cat((A.reshape(-1, spec.payload_bits), P), dim=0)),
        "prototype_shape": list(V.shape), "prototype_storage_values": int(V.numel()),
        "learnable_amplitudes": bool(learn_amplitudes), "discrete_maps_learnable": False,
        "exact_column_energy": True, "saved_generator_has_global_message_axis": False,
        "compact_forward_support_and_amplitudes": True, "scalable_claim_applies_to_support_rule_only": False,
        "current_D0_D1_runtime_has_global_message_axis": True,
    })
    encoder = PrototypeHashEncoder(spec, codebook, support)
    encoder.apply_constraints()
    return encoder


def build_random_procedural_hash_codebook(payload_bits: int, n: int, support_size: int, label_bits: int,
                                          seed: int, learn_amplitudes: bool = False,
                                          amplitude_init: str = "gaussian") -> ProceduralHashCodebook:
    """Build an M-free generator, including at B values where enumeration is impossible."""
    B, T = int(payload_bits), int(support_size)
    if n % T:
        raise ValueError(f"support size T={T} must divide n={n}")
    R = n // T
    if R <= 0 or R & (R - 1):
        raise ValueError(f"R=n/T must be a power of two, got R={R}")
    structure_generator = torch.Generator().manual_seed(int(seed) + 310_003)
    A, b = random_linear_hash_bank(B, T, int(math.log2(R)), structure_generator)
    P = _projection(B, label_bits, seed)
    V = _prototype_values(T, B, P, seed, amplitude_init, torch.float32)
    return ProceduralHashCodebook(A, b, PrototypeAmplitudeBank(P, V, learn_amplitudes), n)
