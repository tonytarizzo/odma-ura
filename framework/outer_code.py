"""Procedural outer codes mapping payload bits to valid local-section paths.

The scalable encoder never constructs a table with one row per message.  It
computes a path directly,

    f_out : {0,1}^B -> X_1 x ... x X_L,

and the physical encoder transmits ``sum_l F_l[:, f_out(w)_l]``.  The classes
below make that operation, its validity constraints, and its inverse explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch


def _validate_binary(bits: torch.Tensor, payload_bits: int) -> torch.Tensor:
    bits = torch.as_tensor(bits)
    if bits.ndim == 0 or bits.shape[-1] != payload_bits:
        raise ValueError(f"bits must have trailing shape ({payload_bits},), got {tuple(bits.shape)}")
    if bits.numel() and not torch.all((bits == 0) | (bits == 1)):
        raise ValueError("payload must contain only binary values")
    return bits.to(torch.long)


def _bits_to_symbols(bits: torch.Tensor, section_bits: int) -> torch.Tensor:
    if bits.shape[-1] % section_bits:
        raise ValueError(f"bit length {bits.shape[-1]} is not divisible by section width {section_bits}")
    chunks = bits.reshape(*bits.shape[:-1], -1, section_bits)
    weights = 2 ** torch.arange(section_bits - 1, -1, -1, dtype=torch.long, device=bits.device)
    return torch.sum(chunks.to(torch.long) * weights, dim=-1)


def _symbols_to_bits(symbols: torch.Tensor, section_bits: int) -> torch.Tensor:
    shifts = torch.arange(section_bits - 1, -1, -1, dtype=torch.long, device=symbols.device)
    bits = ((symbols.to(torch.long).unsqueeze(-1) >> shifts) & 1)
    return bits.reshape(*symbols.shape[:-1], symbols.shape[-1] * section_bits)


@dataclass(frozen=True)
class LinearCheck:
    """One sparse modular row: ``sum_i coefficients[i] * x[variables[i]] = 0 mod q``."""

    variables: tuple[int, ...]
    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.variables) < 2 or len(self.variables) != len(self.coefficients):
            raise ValueError("a linear check needs equal-length variable and coefficient tuples of length >= 2")
        if len(set(self.variables)) != len(self.variables) or any(index < 0 for index in self.variables):
            raise ValueError("linear-check variables must be distinct nonnegative indices")
        if any(coefficient <= 0 for coefficient in self.coefficients):
            raise ValueError("linear-check coefficients must be nonzero")


@dataclass(frozen=True)
class OuterFactorGraph:
    """Sparse representation of ``H x = 0 mod 2^J`` used by CCS-AMP."""

    section_sizes: tuple[int, ...]
    checks: tuple[LinearCheck, ...]
    section_bits: int

    def __post_init__(self) -> None:
        expected_size = 1 << self.section_bits
        if not self.section_sizes or any(size != expected_size for size in self.section_sizes):
            raise ValueError(f"every linear-code section must have size 2^{self.section_bits}={expected_size}")
        for check in self.checks:
            if max(check.variables) >= len(self.section_sizes):
                raise ValueError("linear check references a nonexistent section")
            if max(check.coefficients) >= expected_size:
                raise ValueError("linear-check coefficient lies outside the modular alphabet")
            if any(math.gcd(coefficient, expected_size) != 1 for coefficient in check.coefficients):
                raise ValueError("linear-check coefficients must be invertible modulo 2^J (therefore odd)")

    @property
    def num_variables(self) -> int: return len(self.section_sizes)

    @property
    def modulus(self) -> int: return 1 << self.section_bits

    def parity_check_matrix(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Materialise the small ``H`` matrix; this is section-sized, never message-sized."""
        H = torch.zeros(len(self.checks), self.num_variables, dtype=torch.long, device=device)
        for row, check in enumerate(self.checks):
            H[row, list(check.variables)] = torch.tensor(check.coefficients, dtype=torch.long, device=device)
        return H

    def syndrome(self, paths: torch.Tensor) -> torch.Tensor:
        paths = torch.as_tensor(paths, dtype=torch.long)
        if paths.ndim == 0 or paths.shape[-1] != self.num_variables:
            raise ValueError(f"paths must have trailing shape ({self.num_variables},), got {tuple(paths.shape)}")
        syndrome = torch.zeros(*paths.shape[:-1], len(self.checks), dtype=torch.long, device=paths.device)
        for row, check in enumerate(self.checks):
            value = torch.zeros(paths.shape[:-1], dtype=torch.long, device=paths.device)
            for variable, coefficient in zip(check.variables, check.coefficients):
                value += coefficient * paths[..., variable]
            syndrome[..., row] = value.remainder(self.modulus)
        return syndrome

    def is_valid(self, paths: torch.Tensor) -> torch.Tensor:
        paths = torch.as_tensor(paths, dtype=torch.long)
        in_range = ((paths >= 0) & (paths < self.modulus)).all(dim=-1)
        safe_paths = paths.clamp(0, self.modulus - 1)
        return in_range & (self.syndrome(safe_paths) == 0).all(dim=-1)


class OuterCode:
    """Common interface for direct payload-to-path encoders."""

    payload_bits: int
    section_sizes: tuple[int, ...]

    @property
    def num_sections(self) -> int: return len(self.section_sizes)

    @property
    def factor_graph(self) -> OuterFactorGraph | None: return None

    def encode_bits(self, bits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        """Alias for the single procedural outer-encoding operation."""
        return self.encode_bits(bits)

    def decode_bits(self, paths: torch.Tensor, validate: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, paths: torch.Tensor, validate: bool = True) -> torch.Tensor:
        return self.decode_bits(paths, validate)

    def is_valid(self, paths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def enumerate_paths(self, max_messages: int = 1 << 20,
                        device: torch.device | str | None = None) -> torch.Tensor:
        """Enumerate all paths only for small-system certification."""
        num_messages = 1 << self.payload_bits
        if num_messages > max_messages:
            raise ValueError(f"refusing to enumerate 2^{self.payload_bits}={num_messages} messages")
        messages = torch.arange(num_messages, dtype=torch.long, device=device)
        shifts = torch.arange(self.payload_bits - 1, -1, -1, dtype=torch.long, device=device)
        bits = ((messages.unsqueeze(1) >> shifts) & 1)
        return self.encode_bits(bits)


class IdentityOuterCode(OuterCode):
    """Split payload bits into sections; every local tuple is valid."""

    def __init__(self, payload_bits: int, section_bits: int | Sequence[int]) -> None:
        if payload_bits <= 0:
            raise ValueError(f"payload_bits must be positive, got {payload_bits}")
        if isinstance(section_bits, int):
            if section_bits <= 0:
                raise ValueError(f"section_bits must be positive, got {section_bits}")
            widths = (section_bits,) * (payload_bits // section_bits)
            if payload_bits % section_bits:
                widths += (payload_bits % section_bits,)
        else:
            widths = tuple(int(width) for width in section_bits)
            if not widths or any(width <= 0 for width in widths) or sum(widths) != payload_bits:
                raise ValueError("explicit section widths must be positive and sum to payload_bits")
        self.payload_bits = int(payload_bits)
        self.section_bits = widths
        self.section_sizes = tuple(1 << width for width in widths)

    def encode_bits(self, bits: torch.Tensor) -> torch.Tensor:
        bits = _validate_binary(bits, self.payload_bits)
        symbols = []; start = 0
        for width in self.section_bits:
            symbols.append(_bits_to_symbols(bits[..., start:start + width], width).squeeze(-1))
            start += width
        return torch.stack(symbols, dim=-1)

    def _validate_paths(self, paths: torch.Tensor) -> torch.Tensor:
        paths = torch.as_tensor(paths, dtype=torch.long)
        if paths.ndim == 0 or paths.shape[-1] != self.num_sections:
            raise ValueError(f"paths must have trailing shape ({self.num_sections},), got {tuple(paths.shape)}")
        return paths

    def is_valid(self, paths: torch.Tensor) -> torch.Tensor:
        paths = self._validate_paths(paths)
        valid = torch.ones(paths.shape[:-1], dtype=torch.bool, device=paths.device)
        for ell, size in enumerate(self.section_sizes):
            valid &= (paths[..., ell] >= 0) & (paths[..., ell] < size)
        return valid

    def decode_bits(self, paths: torch.Tensor, validate: bool = True) -> torch.Tensor:
        paths = self._validate_paths(paths)
        if validate and not bool(self.is_valid(paths).all()):
            raise ValueError("identity-code path contains an out-of-range section symbol")
        chunks = [_symbols_to_bits(paths[..., ell:ell + 1], width) for ell, width in enumerate(self.section_bits)]
        return torch.cat(chunks, dim=-1)


class SparseLinearOuterCode(OuterCode):
    """Systematic sparse linear outer code over integers modulo ``q=2^J``.

    The payload becomes ``k=B/J`` information symbols.  Each configured check
    adds one parity symbol, so the transmitted path has ``L=k+r`` sections.
    Graph supports and coefficients are fixed structural configuration; the
    physical atom banks and decoder calibration remain learnable.
    """

    def __init__(self, payload_bits: int, section_bits: int,
                 parity_supports: Sequence[Sequence[int]],
                 parity_coefficients: Sequence[Sequence[int]] | None = None,
                 info_positions: Sequence[int] | None = None,
                 parity_positions: Sequence[int] | None = None) -> None:
        if payload_bits <= 0 or section_bits <= 0 or payload_bits % section_bits:
            raise ValueError("payload_bits and section_bits must be positive, with section_bits dividing payload_bits")
        self.payload_bits = int(payload_bits)
        self.uniform_section_bits = int(section_bits)
        self.num_information_sections = payload_bits // section_bits
        supports = tuple(tuple(int(index) for index in support) for support in parity_supports)
        if any(not support or len(set(support)) != len(support)
               or min(support) < 0 or max(support) >= self.num_information_sections for support in supports):
            raise ValueError("each parity support must contain distinct valid information-section indices")
        if parity_coefficients is None:
            coefficients = tuple((1,) * len(support) for support in supports)
        else:
            coefficients = tuple(tuple(int(value) for value in row) for row in parity_coefficients)
            if len(coefficients) != len(supports) or any(len(row) != len(support) for row, support in zip(coefficients, supports)):
                raise ValueError("parity coefficients must match parity supports")
        modulus = 1 << section_bits
        if any(value <= 0 or value >= modulus or math.gcd(value, modulus) != 1
               for row in coefficients for value in row):
            raise ValueError(f"parity coefficients must be odd integers in [1,{modulus})")
        num_sections = self.num_information_sections + len(supports)
        info = tuple(range(self.num_information_sections)) if info_positions is None else tuple(int(x) for x in info_positions)
        parity = (tuple(range(self.num_information_sections, num_sections)) if parity_positions is None
                  else tuple(int(x) for x in parity_positions))
        if (len(info) != self.num_information_sections or len(parity) != len(supports)
                or sorted(info + parity) != list(range(num_sections))):
            raise ValueError("information and parity positions must partition all path sections")
        self.parity_supports = supports
        self.parity_coefficients = coefficients
        self.info_positions = info
        self.parity_positions = parity
        self.section_sizes = (modulus,) * num_sections
        checks = []
        for support, coefficient_row, parity_position in zip(supports, coefficients, parity):
            entries = [(info[index], coefficient) for index, coefficient in zip(support, coefficient_row)]
            entries.append((parity_position, 1))
            entries.sort()
            checks.append(LinearCheck(tuple(variable for variable, _ in entries),
                                      tuple(coefficient for _, coefficient in entries)))
        self._factor_graph = OuterFactorGraph(self.section_sizes, tuple(checks), section_bits)

    @property
    def factor_graph(self) -> OuterFactorGraph: return self._factor_graph

    def _validate_paths(self, paths: torch.Tensor) -> torch.Tensor:
        paths = torch.as_tensor(paths, dtype=torch.long)
        if paths.ndim == 0 or paths.shape[-1] != self.num_sections:
            raise ValueError(f"paths must have trailing shape ({self.num_sections},), got {tuple(paths.shape)}")
        return paths

    def encode_bits(self, bits: torch.Tensor) -> torch.Tensor:
        bits = _validate_binary(bits, self.payload_bits)
        return self.encode_symbols(_bits_to_symbols(bits, self.uniform_section_bits))

    def encode_symbols(self, information: torch.Tensor) -> torch.Tensor:
        """Encode systematic section symbols directly; useful for list extraction."""
        information = torch.as_tensor(information, dtype=torch.long)
        if information.ndim == 0 or information.shape[-1] != self.num_information_sections:
            raise ValueError(f"information must have trailing shape ({self.num_information_sections},)")
        modulus = 1 << self.uniform_section_bits
        if information.numel() and (int(information.min()) < 0 or int(information.max()) >= modulus):
            raise ValueError(f"information symbols must lie in [0,{modulus})")
        paths = torch.zeros(*information.shape[:-1], self.num_sections, dtype=torch.long, device=information.device)
        paths[..., list(self.info_positions)] = information
        for support, coefficients, position in zip(self.parity_supports, self.parity_coefficients, self.parity_positions):
            parity = torch.zeros(information.shape[:-1], dtype=torch.long, device=information.device)
            for index, coefficient in zip(support, coefficients):
                parity += coefficient * information[..., index]
            paths[..., position] = (-parity).remainder(modulus)
        return paths

    def is_valid(self, paths: torch.Tensor) -> torch.Tensor:
        return self.factor_graph.is_valid(self._validate_paths(paths))

    def decode_bits(self, paths: torch.Tensor, validate: bool = True) -> torch.Tensor:
        paths = self._validate_paths(paths)
        if validate and not bool(self.is_valid(paths).all()):
            raise ValueError("path violates the configured outer-code parity checks")
        return _symbols_to_bits(paths[..., list(self.info_positions)], self.uniform_section_bits)


def random_sparse_outer_code(payload_bits: int, section_bits: int, num_parity_sections: int,
                             check_degree: int, generator: torch.Generator | None = None,
                             random_coefficients: bool = True) -> SparseLinearOuterCode:
    """Sample a fixed sparse graph configuration for experiment-level search."""
    if num_parity_sections < 0:
        raise ValueError(f"num_parity_sections must be nonnegative, got {num_parity_sections}")
    if payload_bits % section_bits:
        raise ValueError("section_bits must divide payload_bits")
    num_information = payload_bits // section_bits
    if check_degree <= 0 or check_degree > num_information:
        raise ValueError(f"check_degree must lie in [1,{num_information}], got {check_degree}")
    supports = [tuple(torch.randperm(num_information, generator=generator)[:check_degree].tolist())
                for _ in range(num_parity_sections)]
    coefficients = None
    if random_coefficients:
        half = 1 << max(section_bits - 1, 0)
        coefficients = [tuple((2 * torch.randint(0, half, (check_degree,), generator=generator) + 1).tolist())
                        for _ in range(num_parity_sections)]
    return SparseLinearOuterCode(payload_bits, section_bits, supports, coefficients)


def triadic_outer_code(payload_bits: int, section_bits: int) -> SparseLinearOuterCode:
    """Cyclic degree-three CCS graph: each parity joins two adjacent information symbols."""
    if payload_bits % section_bits:
        raise ValueError("section_bits must divide payload_bits")
    num_information = payload_bits // section_bits
    if num_information < 2:
        raise ValueError("the triadic graph needs at least two information sections")
    supports = [(ell, (ell + 1) % num_information) for ell in range(num_information)]
    info_positions = tuple(2 * ell for ell in range(num_information))
    parity_positions = tuple(2 * ell + 1 for ell in range(num_information))
    return SparseLinearOuterCode(payload_bits, section_bits, supports,
                                 info_positions=info_positions, parity_positions=parity_positions)


def ccs_amp_paper_outer_code() -> SparseLinearOuterCode:
    """Published CCS-AMP outer dimensions: B=128, J=16, eight information and eight parity sections."""
    return triadic_outer_code(payload_bits=128, section_bits=16)
