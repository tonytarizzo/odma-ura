"""Scalable section-domain execution with no global ``M = 2^B`` axis.

The explicit backend represents ``Phi = sum_l F_l T_l`` and stores one global
count per message. This module executes the same physical model from the local
section counts ``s_l`` directly:

    y = sum_l F_l s_l,  F_l = B_l U_l.

A procedural outer code defines legal paths through the local atom alphabets.
It remains separate from this physical linear operator, while ``encode_bits``
composes the two operations without constructing a global lookup table.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import torch
from torch import nn

from .channel import ebn0_db_to_noise_var
from .constraints import apply_constraints
from .core import ComponentSpec, SectionedURABatch, SectionedURASpec
from .encoder import ComponentConstraints, Encoder, LocalAtomBank, SubsampledHadamardAtomBank
from .initializers import init_C, init_R, init_U
from .outer_code import OuterCode, SparseLinearOuterCode, ccs_amp_paper_outer_code


class FixedOrthogonalMixer(nn.Module):
    """Implicit orthogonal mixing built from fixed permutations and 2x2 Hadamard rotations."""

    def __init__(self, n: int, num_stages: int, generator: torch.Generator | None = None) -> None:
        super().__init__()
        if n <= 0 or num_stages < 0:
            raise ValueError(f"n must be positive and num_stages nonnegative, got n={n}, stages={num_stages}")
        permutations = [torch.randperm(n, generator=generator) for _ in range(num_stages)]
        if permutations:
            permutation = torch.stack(permutations)
            inverse = torch.argsort(permutation, dim=1)
        else:
            permutation = torch.empty(0, n, dtype=torch.long)
            inverse = torch.empty(0, n, dtype=torch.long)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)
        self.n = int(n)

    @staticmethod
    def _pair_mix(x: torch.Tensor) -> torch.Tensor:
        pair_length = 2 * (x.shape[-1] // 2)
        if pair_length == 0:
            return x
        pairs = x[..., :pair_length].reshape(*x.shape[:-1], -1, 2)
        scale = math.sqrt(0.5)
        mixed = torch.stack(((pairs[..., 0] + pairs[..., 1]) * scale,
                             (pairs[..., 0] - pairs[..., 1]) * scale), dim=-1).flatten(-2)
        return torch.cat((mixed, x[..., pair_length:]), dim=-1) if pair_length < x.shape[-1] else mixed

    def _stage(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        permuted = x.index_select(-1, self.permutation[stage])
        return self._pair_mix(permuted).index_select(-1, self.inverse_permutation[stage])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n:
            raise ValueError(f"mixer expected trailing dimension {self.n}, got {tuple(x.shape)}")
        for stage in range(self.permutation.shape[0]):
            x = self._stage(x, stage)
        return x

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n:
            raise ValueError(f"mixer expected trailing dimension {self.n}, got {tuple(x.shape)}")
        for stage in range(self.permutation.shape[0] - 1, -1, -1):
            x = self._stage(x, stage)
        return x


class SectionedEncoder(nn.Module):
    """A sum of local atom banks whose state size is ``sum_l N_l``, not ``M``."""

    def __init__(self, banks: Sequence[LocalAtomBank], spec: SectionedURASpec,
                 orthogonal_mixer: FixedOrthogonalMixer | None = None,
                 section_energies: Sequence[float] | None = None) -> None:
        super().__init__()
        if not banks:
            raise ValueError("SectionedEncoder requires at least one local atom bank")
        dtype = banks[0].dtype
        device = banks[0].device
        if any(bank.dtype != dtype or bank.device != device for bank in banks):
            raise ValueError("all local atom banks must share dtype and device")
        if orthogonal_mixer is None:
            if any(bank.n != spec.n for bank in banks):
                raise ValueError("overlapping local atom banks must use the full SectionedURASpec resource length")
            energies = ((1.0,) * len(banks) if section_energies is None
                        else tuple(float(value) for value in section_energies))
        else:
            if orthogonal_mixer.n != spec.n or sum(bank.n for bank in banks) > spec.n:
                raise ValueError("orthogonal local dimensions must fit inside the mixer resource length")
            energies = ((1.0 / len(banks),) * len(banks) if section_energies is None
                        else tuple(float(value) for value in section_energies))
        if len(energies) != len(banks) or any(value <= 0.0 for value in energies):
            raise ValueError("positive section energies must have one entry per bank")
        if section_energies is not None and abs(sum(energies) - 1.0) > 1e-10:
            raise ValueError("explicit section energies must sum to one")
        self.banks = nn.ModuleList(banks)
        self.orthogonal_mixer = orthogonal_mixer
        real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
        self.register_buffer("section_scales", torch.sqrt(torch.tensor(energies, dtype=real_dtype, device=device)))
        self.spec = spec
        offsets = [0]
        for bank in banks:
            offsets.append(offsets[-1] + bank.n)
        self._resource_slices = tuple(slice(offsets[ell], offsets[ell + 1]) for ell in range(len(banks)))
        self._spectral_cache: dict[int, torch.Tensor] = {}

    @property
    def n(self) -> int: return self.spec.n

    @property
    def num_sections(self) -> int: return len(self.banks)

    @property
    def section_sizes(self) -> tuple[int, ...]: return tuple(bank.num_atoms for bank in self.banks)

    @property
    def state_size(self) -> int: return sum(self.section_sizes)

    @property
    def energy_mode(self) -> str:
        if self.orthogonal_mixer is not None:
            return "orthogonal_exact"
        return "overlapping_sampled" if abs(float(self.section_scales.square().sum()) - 1.0) <= 1e-6 else "overlapping_unscaled"

    @property
    def dtype(self) -> torch.dtype: return self.banks[0].dtype

    @property
    def device(self) -> torch.device: return self.banks[0].device

    def _validate_section_counts(self, section_counts: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(section_counts) != self.num_sections:
            raise ValueError(f"expected {self.num_sections} section tensors, got {len(section_counts)}")
        counts = tuple(section_counts)
        rank = counts[0].ndim
        if rank not in (1, 2):
            raise ValueError(f"section counts must be vectors or batches, got rank {rank}")
        batch = counts[0].shape[0] if rank == 2 else None
        for ell, (x, size) in enumerate(zip(counts, self.section_sizes)):
            expected = (size,) if rank == 1 else (batch, size)
            if x.ndim != rank or tuple(x.shape) != expected:
                raise ValueError(f"section {ell} counts must have shape {expected}, got {tuple(x.shape)}")
        return counts

    def synthesize(self, section_counts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute ``sum_l F_l s_l`` from local count tensors only."""
        counts = self._validate_section_counts(section_counts)
        if self.orthogonal_mixer is None:
            out = self.section_scales[0].to(self.dtype) * self.banks[0].local_matvec(counts[0].to(self.dtype))
            for scale, bank, local in zip(self.section_scales[1:], self.banks[1:], counts[1:]):
                out = out + scale.to(self.dtype) * bank.local_matvec(local.to(self.dtype))
            return out
        pieces = [scale.to(self.dtype) * bank.local_matvec(local.to(self.dtype))
                  for scale, bank, local in zip(self.section_scales, self.banks, counts)]
        used = sum(piece.shape[-1] for piece in pieces)
        if used < self.n:
            pieces.append(torch.zeros(*pieces[0].shape[:-1], self.n - used, dtype=self.dtype, device=self.device))
        return self.orthogonal_mixer(torch.cat(pieces, dim=-1))

    def local_adjoint(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return ``(F_l^H residual)_l`` without broadcasting to global messages."""
        if self.orthogonal_mixer is None:
            return tuple(scale.to(self.dtype) * bank.local_rmatvec(residual)
                         for scale, bank in zip(self.section_scales, self.banks))
        latent = self.orthogonal_mixer.adjoint(residual)
        return tuple(scale.to(self.dtype) * bank.local_rmatvec(latent[..., resource_slice])
                     for scale, bank, resource_slice in zip(self.section_scales, self.banks, self._resource_slices))

    def counts_from_paths(self, paths: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Scatter active paths ``(batch,K,L)`` directly into local count tensors.

        A negative path entry is treated as padding, allowing batches with
        different realised values of ``K_a`` without any global message vector.
        """
        if paths.ndim != 3 or paths.shape[2] != self.num_sections:
            raise ValueError(
                f"paths must have shape (batch,K,{self.num_sections}), got {tuple(paths.shape)}")
        paths = paths.to(device=self.device, dtype=torch.long)
        valid_by_section = paths >= 0
        if not torch.equal(valid_by_section, valid_by_section[:, :, :1].expand_as(valid_by_section)):
            raise ValueError("path padding must be consistent across every section")
        out: list[torch.Tensor] = []
        for ell, size in enumerate(self.section_sizes):
            indices = paths[:, :, ell]
            valid = indices >= 0
            if valid.any() and int(indices[valid].max()) >= size:
                raise ValueError(f"section {ell} path index outside [0,{size})")
            local = torch.zeros(paths.shape[0], size, dtype=self.dtype, device=self.device)
            safe = indices.clamp_min(0)
            local.scatter_add_(1, safe, valid.to(local.dtype))
            out.append(local)
        return tuple(out)

    def encode_paths(self, paths: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        counts = self.counts_from_paths(paths)
        return self.synthesize(counts), counts

    def _validate_outer_code(self, outer_code: OuterCode) -> None:
        if outer_code.payload_bits != self.spec.payload_bits:
            raise ValueError(f"outer code carries {outer_code.payload_bits} bits, expected {self.spec.payload_bits}")
        if outer_code.section_sizes != self.section_sizes:
            raise ValueError(f"outer-code alphabets {outer_code.section_sizes} do not match atom banks {self.section_sizes}")

    def paths_from_bits(self, bits: torch.Tensor, outer_code: OuterCode) -> torch.Tensor:
        """Apply the procedural outer encoder to payloads shaped ``(batch,K,B)``."""
        self._validate_outer_code(outer_code)
        if bits.ndim != 3 or bits.shape[2] != self.spec.payload_bits:
            raise ValueError(f"bits must have shape (batch,K,{self.spec.payload_bits}), got {tuple(bits.shape)}")
        paths = outer_code.encode_bits(bits.to(self.device))
        if not bool(outer_code.is_valid(paths).all()):
            raise RuntimeError("outer encoder produced an invalid path")
        return paths

    def encode_bits(self, bits: torch.Tensor, outer_code: OuterCode
                    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        """Map payload bits directly to a signal, local counts, and valid paths."""
        paths = self.paths_from_bits(bits, outer_code)
        signal, counts = self.encode_paths(paths)
        return signal, counts, paths

    def codeword_columns(self, paths: torch.Tensor) -> torch.Tensor:
        """Materialise only requested procedural codewords, returning shape ``(...,n)``."""
        paths = torch.as_tensor(paths, dtype=torch.long, device=self.device)
        if paths.ndim == 0 or paths.shape[-1] != self.num_sections:
            raise ValueError(f"paths must have trailing shape ({self.num_sections},), got {tuple(paths.shape)}")
        leading = paths.shape[:-1]
        flat = paths.reshape(-1, self.num_sections)
        if self.orthogonal_mixer is None:
            columns = sum(scale.to(self.dtype) * bank.atom_columns(flat[:, ell])
                          for ell, (scale, bank) in enumerate(zip(self.section_scales, self.banks))).transpose(0, 1)
        else:
            pieces = [scale.to(self.dtype) * bank.atom_columns(flat[:, ell])
                      for ell, (scale, bank) in enumerate(zip(self.section_scales, self.banks))]
            used = sum(piece.shape[0] for piece in pieces)
            if used < self.n:
                pieces.append(torch.zeros(self.n - used, flat.shape[0], dtype=self.dtype, device=self.device))
            columns = self.orthogonal_mixer(torch.cat(pieces, dim=0).transpose(0, 1))
        return columns.reshape(*leading, self.n)

    def path_energies(self, paths: torch.Tensor) -> torch.Tensor:
        columns = self.codeword_columns(paths)
        return torch.sum(torch.abs(columns) ** 2, dim=-1).real

    def certify_exact_energy(self, tolerance: float = 1e-6) -> dict[str, float | bool | str]:
        """Certify the structural unit-energy guarantee without enumerating complete paths."""
        if self.orthogonal_mixer is None:
            return {"guaranteed": False, "mode": self.energy_mode}
        deviations = []
        for bank in self.banks:
            deviations.append(bank.max_unit_energy_deviation())
        max_local_deviation = float(torch.stack(deviations).max().detach())
        energy_sum_deviation = abs(float(torch.sum(self.section_scales.square()).detach()) - 1.0)
        return {"guaranteed": max(max_local_deviation, energy_sum_deviation) <= tolerance,
                "mode": "orthogonal_exact", "max_local_energy_deviation": max_local_deviation,
                "section_energy_sum_deviation": energy_sum_deviation, "tolerance": float(tolerance)}

    def apply_constraints(self) -> None:
        items: list[tuple[str, torch.Tensor, str]] = []
        for ell, bank in enumerate(self.banks):
            for name, tensor, kind in bank.constraint_items():
                items.append((f"section{ell}.{name}", tensor, kind))
        apply_constraints(items)
        self._spectral_cache.clear()

    @staticmethod
    def _state_norm(state: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(sum(torch.sum(torch.abs(x) ** 2).real for x in state)).clamp_min(1e-12)

    def spectral_norm_squared(self, num_iters: int = 20, generator: torch.Generator | None = None,
                              use_cache: bool = True) -> torch.Tensor:
        """Estimate the norm of the concatenated local operator ``[F_1 ... F_L]``."""
        if num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {num_iters}")
        if use_cache and int(num_iters) in self._spectral_cache:
            return self._spectral_cache[int(num_iters)]
        real_dtype = torch.float32 if self.dtype in (torch.float32, torch.complex64) else torch.float64
        state = tuple(torch.randn(size, dtype=real_dtype, device=self.device, generator=generator)
                      for size in self.section_sizes)
        norm = self._state_norm(state)
        state = tuple(x / norm for x in state)
        with torch.no_grad():
            for _ in range(int(num_iters)):
                state = tuple(x.real for x in self.local_adjoint(self.synthesize(state)))
                norm = self._state_norm(state)
                state = tuple(x / norm for x in state)
            y = self.synthesize(state)
            value = torch.sum(torch.abs(y) ** 2).real.clamp_min(1e-12)
        if use_cache:
            self._spectral_cache[int(num_iters)] = value
        return value


def build_sectioned_encoder(spec: SectionedURASpec, component_specs: Sequence[ComponentSpec],
                            constraints: Sequence[ComponentConstraints] | None = None,
                            section_energies: Sequence[float] | None = None,
                            dtype: torch.dtype = torch.float32,
                            generator: torch.Generator | None = None) -> SectionedEncoder:
    """Build only ``R/C/U``; all ``T`` fields in ComponentSpec are ignored."""
    if constraints is not None and len(constraints) != len(component_specs):
        raise ValueError(f"constraints length {len(constraints)} != component count {len(component_specs)}")
    banks: list[LocalAtomBank] = []
    for ell, cs in enumerate(component_specs):
        Q, d, V = int(cs.Q), int(cs.d), int(cs.V)
        N = Q * V if cs.N is None else int(cs.N)
        R = init_R(cs.R_init, Q, spec.n, d, dtype, generator, cs.explicit_R)
        C = init_C(cs.C_init, d, V, dtype, generator, cs.explicit_C)
        atom_q, atom_v = init_U(cs.U_init, Q, V, N, generator, cs.explicit_atom_q, cs.explicit_atom_v)
        bank_constraints = constraints[ell] if constraints is not None else ComponentConstraints()
        banks.append(LocalAtomBank(R, C, atom_q, atom_v, learn_R=cs.learn_R, learn_C=cs.learn_C,
                                   constraints=bank_constraints))
    return SectionedEncoder(banks, spec, section_energies=section_energies)


def build_orthogonal_sectioned_encoder(spec: SectionedURASpec, outer_code: OuterCode,
                                       section_dimensions: int | Sequence[int] | None = None,
                                       section_energies: Sequence[float] | None = None,
                                       learn_C: bool = True, mixing_stages: int | None = None,
                                       bank_type: str = "explicit",
                                       dtype: torch.dtype = torch.float32,
                                       generator: torch.Generator | None = None) -> SectionedEncoder:
    """Build an exact-unit-energy direct sum, optionally spread by a fixed orthogonal mixer."""
    if outer_code.payload_bits != spec.payload_bits:
        raise ValueError("outer-code payload does not match the SectionedURASpec")
    L = outer_code.num_sections
    if section_dimensions is None:
        base, remainder = divmod(spec.n, L)
        dimensions = tuple(base + (ell < remainder) for ell in range(L))
    elif isinstance(section_dimensions, int):
        dimensions = (int(section_dimensions),) * L
    else:
        dimensions = tuple(int(value) for value in section_dimensions)
    if len(dimensions) != L or any(value <= 0 for value in dimensions) or sum(dimensions) > spec.n:
        raise ValueError(f"positive section dimensions must have length {L} and sum to at most n={spec.n}")
    stages = int(math.ceil(math.log2(spec.n))) if mixing_stages is None else int(mixing_stages)
    mixer = FixedOrthogonalMixer(spec.n, stages, generator)
    if bank_type not in {"explicit", "subsampled_hadamard"}:
        raise ValueError(f"bank_type must be explicit or subsampled_hadamard, got {bank_type!r}")
    if bank_type == "subsampled_hadamard" and learn_C:
        raise ValueError("the implicit subsampled-Hadamard bank is fixed; pass learn_C=False")
    banks = []
    for dimension, size in zip(dimensions, outer_code.section_sizes):
        if bank_type == "subsampled_hadamard":
            banks.append(SubsampledHadamardAtomBank(size, dimension, dtype, generator))
        else:
            R = torch.ones(1, dimension, dtype=dtype)
            C = init_C("random_gaussian", dimension, size, dtype, generator)
            atom_q = torch.zeros(size, dtype=torch.long)
            atom_v = torch.arange(size, dtype=torch.long)
            banks.append(LocalAtomBank(R, C, atom_q, atom_v, learn_R=False, learn_C=learn_C,
                                       constraints=ComponentConstraints(C="unit_norm_columns")))
    encoder = SectionedEncoder(banks, spec, mixer, section_energies)
    encoder.apply_constraints()
    return encoder


def build_default_scalable_setup(num_active: int, n: int = 38_400, num_antennas: int = 1,
                                 mixing_stages: int | None = None, dtype: torch.dtype = torch.float32,
                                 generator: torch.Generator | None = None
                                 ) -> tuple[SectionedEncoder, SparseLinearOuterCode]:
    """B=128,J=16 default with exact energy and implicit local banks; not a claim of paper-inner equivalence."""
    outer_code = ccs_amp_paper_outer_code()
    spec = SectionedURASpec(n=n, payload_bits=128, num_active=num_active, num_antennas=num_antennas)
    encoder = build_orthogonal_sectioned_encoder(spec, outer_code, learn_C=False, mixing_stages=mixing_stages,
                                                 bank_type="subsampled_hadamard", dtype=dtype, generator=generator)
    return encoder, outer_code


def sectioned_from_explicit(encoder: Encoder) -> SectionedEncoder:
    """Clone an explicit encoder's physical banks for small-B equivalence checks."""
    banks = [LocalAtomBank(c.R.detach(), c.C.detach(), c.atom_q, c.atom_v,
                           learn_R=isinstance(c.R, nn.Parameter), learn_C=isinstance(c.C, nn.Parameter),
                           constraints=c.constraints) for c in encoder.components]
    spec = SectionedURASpec(n=encoder.n, payload_bits=int(encoder.spec.payload_bits),
                            num_active=encoder.spec.num_active, num_antennas=encoder.spec.num_antennas,
                            energy_per_codeword=encoder.spec.energy_per_codeword)
    return SectionedEncoder(banks, spec)


def uniform_section_paths_generator(num_active: int, section_sizes: Sequence[int],
                                    generator: torch.Generator | None = None,
                                    device: torch.device | str | None = None) -> Callable[[int], torch.Tensor]:
    """Sample independent local tuples without outer-code validity constraints."""
    sizes = tuple(int(size) for size in section_sizes)
    if num_active <= 0 or not sizes or any(size <= 1 for size in sizes):
        raise ValueError("num_active must be positive and every section alphabet must contain at least two atoms")

    def sample(batch_size: int) -> torch.Tensor:
        sections = [torch.randint(size, (batch_size, num_active), generator=generator, device=device) for size in sizes]
        return torch.stack(sections, dim=-1)

    return sample


def outer_code_path_generator(num_active: int, outer_code: OuterCode,
                              generator: torch.Generator | None = None,
                              device: torch.device | str | None = None) -> Callable[[int], torch.Tensor]:
    """Sample uniform payload bits and procedurally encode only their valid paths."""
    if num_active <= 0:
        raise ValueError(f"num_active must be positive, got {num_active}")

    def sample(batch_size: int) -> torch.Tensor:
        bits = torch.randint(2, (batch_size, num_active, outer_code.payload_bits),
                             dtype=torch.long, generator=generator, device=device)
        return outer_code.encode_bits(bits)

    return sample


def sampled_energy_report(encoder: SectionedEncoder, outer_code: OuterCode, num_samples: int = 1024,
                          generator: torch.Generator | None = None) -> dict[str, float | int | str]:
    """Measure procedural codeword energies on uniformly sampled messages without an M-axis."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    encoder._validate_outer_code(outer_code)
    bits = torch.randint(2, (num_samples, outer_code.payload_bits), generator=generator, device=encoder.device)
    with torch.no_grad():
        energies = encoder.path_energies(outer_code.encode_bits(bits))
    return {"mode": encoder.energy_mode, "num_samples": int(num_samples),
            "minimum": float(energies.min()), "mean": float(energies.mean()),
            "maximum": float(energies.max()), "max_abs_unit_deviation": float(torch.max(torch.abs(energies - 1.0)))}


def sample_sectioned_batch(encoder: SectionedEncoder, batch_size: int,
                           path_sampler: Callable[[int], torch.Tensor],
                           fading_sampler: Callable[[int], torch.Tensor], ebn0_db: float,
                           generator: torch.Generator | None = None) -> SectionedURABatch:
    """Generate a channel batch without allocating any object indexed by global messages."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    paths = path_sampler(batch_size).to(device=encoder.device, dtype=torch.long)
    y, section_counts = encoder.encode_paths(paths)
    H = fading_sampler(batch_size).to(dtype=encoder.dtype, device=encoder.device)
    Y_clean = y.unsqueeze(-1) * H.unsqueeze(1)
    noise_var = ebn0_db_to_noise_var(ebn0_db, encoder.spec.payload_bits, encoder.spec.energy_per_codeword)
    if encoder.dtype.is_complex:
        noise = torch.randn(Y_clean.shape, dtype=Y_clean.dtype, device=Y_clean.device,
                            generator=generator) * (noise_var / 2.0) ** 0.5
    else:
        noise = torch.randn(Y_clean.shape, dtype=Y_clean.dtype, device=Y_clean.device,
                            generator=generator) * noise_var ** 0.5
    Y = Y_clean + noise
    num_active = (paths[:, :, 0] >= 0).sum(dim=1).to(torch.long)
    return SectionedURABatch(section_counts=section_counts, active_paths=paths, y_clean=y, Y_clean=Y_clean,
                             Y=Y, H=H, noise_var=noise_var, num_active=num_active, ebn0_db=float(ebn0_db))
