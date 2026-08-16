"""Encoder class: one experiment ↔ one Encoder instance.

The Encoder stores the (R, C, U, T) factors of every product component and
exposes the standard global-codebook interface:
    explicit_matrix() -> Phi in C^{n x M}
    matvec(a)         -> Phi a            (in C^n or batched in C^{B x n})
    rmatvec(r)        -> Phi^H r
    encode(counts)    -> Phi counts       (the noiseless transmitted signal)

It also owns constraint metadata, applied via `apply_constraints()` after each
optimiser step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

from .constraints import apply_constraints
from .core import ComponentSpec, URASpec
from .initializers import init_C, init_R, init_T, init_U


@dataclass
class ComponentConstraints:
    R: str = "none"
    C: str = "none"


class LocalAtomBank(nn.Module):
    """One scalable physical factor ``F_l = B_l U_l`` without a global message axis.

    The bank owns the existing ``R``, ``C``, and valid-atom ``U`` data. Its
    numerical state is indexed only by the ``N`` local atoms. In particular it
    has no ``msg_to_atom`` lookup and no dependency on ``M = 2^B``.

    Parameters or buffers (depending on ``learn_R`` / ``learn_C``):
        R          (Q, n, d), or (Q, n) for diagonal operators with d=n
        C          (d, V)

    Buffers:
        atom_q     (N,)   operator index of each valid atom
        atom_v     (N,)   local-codeword index of each valid atom
    """

    def __init__(self, R: torch.Tensor, C: torch.Tensor, atom_q: torch.Tensor,
                 atom_v: torch.Tensor, learn_R: bool = False, learn_C: bool = False,
                 constraints: ComponentConstraints | None = None) -> None:
        super().__init__()
        if R.ndim not in (2, 3):
            raise ValueError(f"R must have shape (Q, n, d) or diagonal shape (Q, n), got {tuple(R.shape)}")
        if C.ndim != 2:
            raise ValueError(f"C must have shape (d, V), got {tuple(C.shape)}")
        if R.ndim == 3 and R.shape[2] != C.shape[0]:
            raise ValueError(f"R/C local dim mismatch: {R.shape[2]} vs {C.shape[0]}")
        if R.ndim == 2 and R.shape[1] != C.shape[0]:
            raise ValueError(f"diagonal R requires d=n={R.shape[1]}, got C local dimension {C.shape[0]}")
        if not (atom_q.shape == atom_v.shape and atom_q.ndim == 1):
            raise ValueError("atom_q and atom_v must be 1-D tensors of equal length")
        if atom_q.numel() == 0:
            raise ValueError("a local atom bank must contain at least one valid atom")
        if int(atom_q.min()) < 0 or int(atom_v.min()) < 0:
            raise ValueError("atom_q and atom_v must be nonnegative")
        if int(atom_q.max()) >= R.shape[0] or int(atom_v.max()) >= C.shape[1]:
            raise ValueError("atom_q or atom_v contains an out-of-range index")

        R = R.clone(); C = C.clone().to(dtype=R.dtype, device=R.device)
        if learn_R:
            self.R = nn.Parameter(R)
        else:
            self.register_buffer("R", R)
        if learn_C:
            self.C = nn.Parameter(C)
        else:
            self.register_buffer("C", C)
        self.register_buffer("atom_q", atom_q.long().clone())
        self.register_buffer("atom_v", atom_v.long().clone())
        self.constraints = constraints or ComponentConstraints()

    @property
    def n(self) -> int: return int(self.R.shape[1])

    @property
    def dtype(self) -> torch.dtype: return self.R.dtype

    @property
    def device(self) -> torch.device: return self.R.device

    @property
    def d(self) -> int: return int(self.C.shape[0])

    @property
    def Q(self) -> int: return int(self.R.shape[0])

    @property
    def V(self) -> int: return int(self.C.shape[1])

    @property
    def num_atoms(self) -> int: return int(self.atom_q.numel())

    @property
    def diagonal_operators(self) -> bool: return self.R.ndim == 2

    def materialize_operator_bank(self) -> torch.Tensor:
        """Return the operator bank as (Q,n,d), expanding diagonal masks only for diagnostics."""
        return torch.diag_embed(self.R) if self.diagonal_operators else self.R

    @staticmethod
    def _as_batch(x: torch.Tensor, expected: int, name: str) -> tuple[torch.Tensor, bool]:
        if x.ndim == 1:
            if x.shape[0] != expected:
                raise ValueError(f"{name} must have length {expected}, got {tuple(x.shape)}")
            return x.unsqueeze(0), True
        if x.ndim == 2 and x.shape[1] == expected:
            return x, False
        raise ValueError(f"{name} must have shape ({expected},) or (B,{expected}), got {tuple(x.shape)}")

    def _atoms_to_pairs(self, atom: torch.Tensor) -> torch.Tensor:
        """Apply U, scattering valid-atom coefficients onto the complete (Q,V) grid."""
        atom, squeezed = self._as_batch(atom, self.num_atoms, "atom")
        pair = torch.zeros(atom.shape[0], self.Q * self.V, dtype=atom.dtype, device=atom.device)
        pair_idx = self.atom_q * self.V + self.atom_v
        pair.scatter_add_(1, pair_idx.unsqueeze(0).expand(atom.shape[0], -1), atom)
        pair = pair.reshape(atom.shape[0], self.Q, self.V)
        return pair.squeeze(0) if squeezed else pair

    def _pairs_to_atoms(self, pair: torch.Tensor) -> torch.Tensor:
        """Apply U^H, returning scores only for the N valid local atoms."""
        if pair.ndim == 2:
            pair = pair.unsqueeze(0); squeezed = True
        elif pair.ndim == 3:
            squeezed = False
        else:
            raise ValueError(f"pair must have shape (Q,V) or (B,Q,V), got {tuple(pair.shape)}")
        if pair.shape[1:] != (self.Q, self.V):
            raise ValueError(f"pair must have trailing shape ({self.Q},{self.V}), got {tuple(pair.shape)}")
        atom = pair[:, self.atom_q, self.atom_v]
        return atom.squeeze(0) if squeezed else atom

    def apply_operators(self, local: torch.Tensor) -> torch.Tensor:
        """Apply every R_q and sum over q. local has shape (Q,d) or (B,Q,d)."""
        if local.ndim == 2:
            local = local.unsqueeze(0); squeezed = True
        elif local.ndim == 3:
            squeezed = False
        else:
            raise ValueError(f"local must have shape (Q,d) or (B,Q,d), got {tuple(local.shape)}")
        if self.diagonal_operators:
            out = (local * self.R.unsqueeze(0)).sum(dim=1)
        else:
            out = torch.einsum("qnd,bqd->bn", self.R, local)
        return out.squeeze(0) if squeezed else out

    def adjoint_operators(self, r: torch.Tensor) -> torch.Tensor:
        """Return (R_q^H r)_q with shape (Q,d) or (B,Q,d)."""
        r, squeezed = self._as_batch(r, self.n, "r")
        if self.diagonal_operators:
            local = r.unsqueeze(1) * self.R.conj().unsqueeze(0)
        else:
            local = torch.einsum("qnd,bn->bqd", self.R.conj(), r)
        return local.squeeze(0) if squeezed else local

    def atom_columns(self, indices: torch.Tensor) -> torch.Tensor:
        """Materialise selected columns of F_l without constructing its complete product grid."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.R.device).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.num_atoms):
            raise ValueError(f"atom index outside [0,{self.num_atoms})")
        q = self.atom_q.index_select(0, indices)
        v = self.atom_v.index_select(0, indices)
        cols = torch.empty(self.n, indices.numel(), dtype=self.C.dtype, device=self.C.device)
        for q_value in torch.unique(q, sorted=True).tolist():
            mask = q == int(q_value)
            local = self.C.index_select(1, v[mask])
            placed = self.R[q_value].unsqueeze(1) * local if self.diagonal_operators else self.R[q_value] @ local
            cols[:, mask] = placed
        return cols

    def explicit_local_matrix(self) -> torch.Tensor:
        """Materialise F_l in C^{n x N_l}; intended only for diagnostics and small tests."""
        return self.atom_columns(torch.arange(self.num_atoms, device=self.R.device))

    def local_matvec(self, atom_counts: torch.Tensor) -> torch.Tensor:
        """Apply F_l to local atom counts shaped (N_l,) or (B,N_l)."""
        pair = self._atoms_to_pairs(atom_counts)
        local = torch.einsum("...qv,dv->...qd", pair.to(self.C.dtype), self.C)
        return self.apply_operators(local)

    def local_rmatvec(self, r: torch.Tensor) -> torch.Tensor:
        """Apply F_l^H to a resource residual without constructing global-message scores."""
        local = self.adjoint_operators(r.to(self.R.dtype))
        pair = torch.einsum("...qd,dv->...qv", local, self.C.conj())
        return self._pairs_to_atoms(pair)

    def constraint_items(self) -> list[tuple[str, torch.Tensor, str]]:
        out: list[tuple[str, torch.Tensor, str]] = []
        if isinstance(self.R, nn.Parameter):
            out.append(("R", self.R.data, self.constraints.R))
        if isinstance(self.C, nn.Parameter):
            out.append(("C", self.C.data, self.constraints.C))
        return out

    def max_unit_energy_deviation(self) -> torch.Tensor:
        energies = torch.sum(torch.abs(self.explicit_local_matrix()) ** 2, dim=0).real
        return torch.max(torch.abs(energies - 1.0))


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised Walsh-Hadamard transform on the final power-of-two axis."""
    length = x.shape[-1]
    if length <= 0 or length & (length - 1):
        raise ValueError(f"Walsh-Hadamard length must be a positive power of two, got {length}")
    y = x
    block = 1
    while block < length:
        pairs = y.reshape(*y.shape[:-1], -1, 2, block)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(*y.shape[:-1], length)
        block *= 2
    return y


class SubsampledHadamardAtomBank(nn.Module):
    """Implicit ``d x 2^J`` unit-column dictionary with no stored dense matrix."""

    def __init__(self, num_atoms: int, output_dimension: int, dtype: torch.dtype = torch.float32,
                 generator: torch.Generator | None = None) -> None:
        super().__init__()
        if num_atoms <= 0 or num_atoms & (num_atoms - 1):
            raise ValueError(f"num_atoms must be a positive power of two, got {num_atoms}")
        if output_dimension <= 0 or output_dimension > num_atoms:
            raise ValueError(f"output_dimension must lie in [1,{num_atoms}], got {output_dimension}")
        rows = torch.randperm(num_atoms, generator=generator)[:output_dimension]
        real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
        signs = (2 * torch.randint(2, (num_atoms,), generator=generator) - 1).to(real_dtype)
        if dtype.is_complex:
            signs = signs.to(dtype)
        self.register_buffer("rows", rows)
        self.register_buffer("signs", signs)
        self._num_atoms = int(num_atoms)
        self._output_dimension = int(output_dimension)

    @property
    def n(self) -> int: return self._output_dimension

    @property
    def num_atoms(self) -> int: return self._num_atoms

    @property
    def dtype(self) -> torch.dtype: return self.signs.dtype

    @property
    def device(self) -> torch.device: return self.signs.device

    @staticmethod
    def _as_batch(x: torch.Tensor, expected: int, name: str) -> tuple[torch.Tensor, bool]:
        if x.ndim == 1 and x.shape[0] == expected:
            return x.unsqueeze(0), True
        if x.ndim == 2 and x.shape[1] == expected:
            return x, False
        raise ValueError(f"{name} must have shape ({expected},) or (batch,{expected}), got {tuple(x.shape)}")

    def local_matvec(self, atom_counts: torch.Tensor) -> torch.Tensor:
        atom_counts, squeezed = self._as_batch(atom_counts, self.num_atoms, "atom_counts")
        transformed = _fwht(atom_counts.to(self.dtype) * self.signs)
        output = transformed.index_select(-1, self.rows) / math.sqrt(self.n)
        return output.squeeze(0) if squeezed else output

    def local_rmatvec(self, residual: torch.Tensor) -> torch.Tensor:
        residual, squeezed = self._as_batch(residual, self.n, "residual")
        embedded = torch.zeros(residual.shape[0], self.num_atoms, dtype=self.dtype, device=self.device)
        embedded.index_copy_(1, self.rows, residual.to(self.dtype))
        output = self.signs.conj() * _fwht(embedded) / math.sqrt(self.n)
        return output.squeeze(0) if squeezed else output

    def atom_columns(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.num_atoms):
            raise ValueError(f"atom index outside [0,{self.num_atoms})")
        parity = torch.zeros(self.n, indices.numel(), dtype=torch.long, device=self.device)
        bits = self.num_atoms.bit_length() - 1
        for shift in range(bits):
            parity ^= ((self.rows >> shift) & 1).unsqueeze(1) * ((indices >> shift) & 1).unsqueeze(0)
        columns = (1 - 2 * parity).to(self.dtype) * self.signs.index_select(0, indices).unsqueeze(0)
        return columns / math.sqrt(self.n)

    def explicit_local_matrix(self, max_elements: int = 20_000_000) -> torch.Tensor:
        if self.n * self.num_atoms > max_elements:
            raise ValueError("refusing to materialise a large implicit Hadamard dictionary; request selected atom_columns instead")
        return self.atom_columns(torch.arange(self.num_atoms, device=self.device))

    def constraint_items(self) -> list[tuple[str, torch.Tensor, str]]:
        return []

    def max_unit_energy_deviation(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.signs.real.dtype, device=self.device)


class ProductComponent(LocalAtomBank):
    """One explicit factor ``B_l U_l T_l`` used by the small-B global backend.

    Buffers:
        atom_q     (N,)   operator index of each valid atom
        atom_v     (N,)   local-codeword index of each valid atom
        msg_to_atom (M,)  global-message -> atom index

    The physical ``R/C/U`` operations live in :class:`LocalAtomBank`. This
    wrapper adds only the length-M ``T`` lookup required by legacy global
    count-vector decoders and exact small-system certification.
    """

    def __init__(self, R: torch.Tensor, C: torch.Tensor, atom_q: torch.Tensor,
                 atom_v: torch.Tensor, msg_to_atom: torch.Tensor,
                 learn_R: bool = False, learn_C: bool = False,
                 constraints: ComponentConstraints | None = None) -> None:
        super().__init__(R, C, atom_q, atom_v, learn_R=learn_R, learn_C=learn_C, constraints=constraints)
        if msg_to_atom.ndim != 1:
            raise ValueError(f"msg_to_atom must be 1-D, got shape {tuple(msg_to_atom.shape)}")
        if int(msg_to_atom.min()) < 0 or int(msg_to_atom.max()) >= self.num_atoms:
            raise ValueError("msg_to_atom contains an out-of-range atom index")
        self.register_buffer("msg_to_atom", msg_to_atom.long().clone())

    @property
    def num_codewords(self) -> int: return int(self.msg_to_atom.numel())

    def _messages_to_atoms(self, a: torch.Tensor) -> torch.Tensor:
        """Apply T, returning coefficients over the N valid local atoms."""
        a, squeezed = self._as_batch(a, self.num_codewords, "a")
        atom = torch.zeros(a.shape[0], self.num_atoms, dtype=a.dtype, device=a.device)
        atom.scatter_add_(1, self.msg_to_atom.unsqueeze(0).expand(a.shape[0], -1), a)
        return atom.squeeze(0) if squeezed else atom

    def _atoms_to_messages(self, atom: torch.Tensor) -> torch.Tensor:
        """Apply T^H, broadcasting local-atom scores back to global messages."""
        atom, squeezed = self._as_batch(atom, self.num_atoms, "atom")
        msg = atom[:, self.msg_to_atom]
        return msg.squeeze(0) if squeezed else msg

    def message_columns(self, indices: torch.Tensor) -> torch.Tensor:
        """Materialise selected component columns without an (M,n,d) intermediate."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.R.device).reshape(-1)
        return self.atom_columns(self.msg_to_atom.index_select(0, indices))

    def explicit_matrix(self) -> torch.Tensor:
        """Column m of B_l U_l T_l: R_{l, q(i_l(m))} c_{l, v(i_l(m))}."""
        return self.message_columns(torch.arange(self.num_codewords, device=self.R.device))

    def matvec(self, a: torch.Tensor) -> torch.Tensor:
        return self.local_matvec(self._messages_to_atoms(a))

    def rmatvec(self, r: torch.Tensor) -> torch.Tensor:
        return self._atoms_to_messages(self.local_rmatvec(r))


def matvec_with_matrix(Phi: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    if a.ndim == 1:
        return Phi @ a
    if a.ndim == 2:
        return a @ Phi.transpose(-1, -2)
    raise ValueError(f"a must have shape (M,) or (B, M), got {tuple(a.shape)}")


def rmatvec_with_matrix(Phi: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    if r.ndim == 1:
        return Phi.conj().transpose(-1, -2) @ r if Phi.is_complex() else Phi.transpose(-1, -2) @ r
    if r.ndim == 2:
        return r @ Phi.conj() if Phi.is_complex() else r @ Phi
    raise ValueError(f"r must have shape (n,) or (B, n), got {tuple(r.shape)}")


class Encoder(nn.Module):
    """Container for one URA experiment.

    Owns all `(R, C, U, T)` factors across components and exposes a single
    interface to the channel and decoders.
    """

    def __init__(self, components: list[ProductComponent], spec: URASpec) -> None:
        super().__init__()
        if not components:
            raise ValueError("Encoder requires at least one ProductComponent")
        n0 = components[0].n
        M0 = components[0].num_codewords
        for c in components:
            if c.n != n0 or c.num_codewords != M0:
                raise ValueError("all components must share (n, M)")
        if n0 != spec.n or M0 != spec.num_codewords:
            raise ValueError(
                f"component shape ({n0}, {M0}) disagrees with URASpec (n={spec.n}, M={spec.num_codewords})")
        self.components = nn.ModuleList(components)
        self.spec = spec
        self._mean_energy_cache: float | None = None
        self._spectral_cache: dict[int, torch.Tensor] = {}

    @property
    def n(self) -> int: return self.spec.n

    @property
    def num_codewords(self) -> int: return self.spec.num_codewords

    @property
    def dtype(self) -> torch.dtype: return self.components[0].R.dtype

    @property
    def device(self) -> torch.device: return self.components[0].R.device

    def explicit_matrix(self) -> torch.Tensor:
        out = self.components[0].explicit_matrix()
        for c in self.components[1:]:
            out = out + c.explicit_matrix()
        return out

    def matvec(self, a: torch.Tensor) -> torch.Tensor:
        out = self.components[0].matvec(a)
        for c in self.components[1:]:
            out = out + c.matvec(a)
        return out

    def rmatvec(self, r: torch.Tensor) -> torch.Tensor:
        out = self.components[0].rmatvec(r)
        for c in self.components[1:]:
            out = out + c.rmatvec(r)
        return out

    def encode(self, counts: torch.Tensor) -> torch.Tensor:
        """Map a count vector a to the noiseless transmitted signal y = Phi a."""
        return self.matvec(counts)

    def apply_constraints(self) -> None:
        items: list[tuple[str, torch.Tensor, str]] = []
        for i, c in enumerate(self.components):
            for name, t, kind in c.constraint_items():
                items.append((f"comp{i}.{name}", t, kind))
        apply_constraints(items)
        self._mean_energy_cache = None
        self._spectral_cache.clear()

    def mean_codeword_energy(self, chunk_size: int = 256, use_cache: bool = True) -> float:
        """Compute mean global column energy in bounded memory, caching until parameters change."""
        if use_cache and self._mean_energy_cache is not None:
            return self._mean_energy_cache
        total = 0.0
        with torch.no_grad():
            for start in range(0, self.num_codewords, int(chunk_size)):
                idx = torch.arange(start, min(start + int(chunk_size), self.num_codewords), device=self.device)
                cols = self.components[0].message_columns(idx)
                for c in self.components[1:]:
                    cols = cols + c.message_columns(idx)
                total += float(torch.sum(torch.abs(cols) ** 2).cpu())
        value = total / float(self.num_codewords)
        if use_cache:
            self._mean_energy_cache = value
        return value

    def spectral_norm_squared(self, num_iters: int = 20, generator: torch.Generator | None = None,
                              use_cache: bool = True) -> torch.Tensor:
        """Estimate ||Phi||_2^2 by power iteration using only implicit forward/adjoint operations."""
        if num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {num_iters}")
        if use_cache and int(num_iters) in self._spectral_cache:
            return self._spectral_cache[int(num_iters)]
        real_dtype = self.components[0].R.real.dtype if self.dtype.is_complex else self.dtype
        x = torch.randn(self.num_codewords, dtype=real_dtype, device=self.device, generator=generator)
        x = x / x.norm().clamp_min(1e-12)
        with torch.no_grad():
            for _ in range(int(num_iters)):
                x = self.rmatvec(self.matvec(x).to(self.dtype)).real
                x = x / x.norm().clamp_min(1e-12)
            y = self.matvec(x.to(self.dtype))
            value = torch.sum(torch.abs(y) ** 2).real.clamp_min(1e-12)
        if use_cache:
            self._spectral_cache[int(num_iters)] = value
        return value


def build_encoder(spec: URASpec, component_specs: list[ComponentSpec],
                  constraints: list[ComponentConstraints] | None = None,
                  dtype: torch.dtype = torch.float32,
                  generator: torch.Generator | None = None) -> Encoder:
    """Instantiate an Encoder from declarative per-component specs."""
    if constraints is not None and len(constraints) != len(component_specs):
        raise ValueError(
            f"constraints length {len(constraints)} != component count {len(component_specs)}")
    components: list[ProductComponent] = []
    for i, cs in enumerate(component_specs):
        Q, d, V = int(cs.Q), int(cs.d), int(cs.V)
        N = Q * V if cs.N is None else int(cs.N)
        R = init_R(cs.R_init, Q, spec.n, d, dtype, generator, cs.explicit_R)
        C = init_C(cs.C_init, d, V, dtype, generator, cs.explicit_C)
        atom_q, atom_v = init_U(cs.U_init, Q, V, N, generator, cs.explicit_atom_q, cs.explicit_atom_v)
        msg_to_atom = init_T(cs.T_init, atom_q.numel(), spec.num_codewords, generator, cs.explicit_msg_to_atom)
        comp_c = constraints[i] if constraints is not None else ComponentConstraints()
        components.append(ProductComponent(R, C, atom_q, atom_v, msg_to_atom,
                                            learn_R=cs.learn_R, learn_C=cs.learn_C,
                                            constraints=comp_c))
    return Encoder(components, spec)
