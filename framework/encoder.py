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
import torch
from torch import nn

from .constraints import apply_constraints
from .core import ComponentSpec, URASpec
from .initializers import init_C, init_R, init_T, init_U


@dataclass
class ComponentConstraints:
    R: str = "none"
    C: str = "none"


class ProductComponent(nn.Module):
    """One factor (B_l U_l T_l).

    Buffers:
        atom_q     (N,)   operator index of each valid atom
        atom_v     (N,)   local-codeword index of each valid atom
        msg_to_atom (M,)  global-message -> atom index

    Parameters or buffers (depending on `learn_R` / `learn_C`):
        R          (Q, n, d)
        C          (d, V)
    """

    def __init__(self, R: torch.Tensor, C: torch.Tensor, atom_q: torch.Tensor,
                 atom_v: torch.Tensor, msg_to_atom: torch.Tensor,
                 learn_R: bool = False, learn_C: bool = False,
                 constraints: ComponentConstraints | None = None) -> None:
        super().__init__()
        if R.ndim != 3:
            raise ValueError(f"R must have shape (Q, n, d), got {tuple(R.shape)}")
        if C.ndim != 2:
            raise ValueError(f"C must have shape (d, V), got {tuple(C.shape)}")
        if R.shape[2] != C.shape[0]:
            raise ValueError(f"R/C local dim mismatch: {R.shape[2]} vs {C.shape[0]}")
        if not (atom_q.shape == atom_v.shape and atom_q.ndim == 1):
            raise ValueError("atom_q and atom_v must be 1-D tensors of equal length")
        if msg_to_atom.ndim != 1:
            raise ValueError(f"msg_to_atom must be 1-D, got shape {tuple(msg_to_atom.shape)}")
        if int(atom_q.max()) >= R.shape[0] or int(atom_v.max()) >= C.shape[1]:
            raise ValueError("atom_q or atom_v contains an out-of-range index")
        if int(msg_to_atom.max()) >= atom_q.numel():
            raise ValueError("msg_to_atom contains an out-of-range atom index")

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
        self.register_buffer("msg_to_atom", msg_to_atom.long().clone())
        self.constraints = constraints or ComponentConstraints()

    @property
    def n(self) -> int: return int(self.R.shape[1])

    @property
    def num_codewords(self) -> int: return int(self.msg_to_atom.numel())

    def explicit_matrix(self) -> torch.Tensor:
        """Column m of B_l U_l T_l: R_{l, q(i_l(m))} c_{l, v(i_l(m))}."""
        atoms = self.msg_to_atom
        q = self.atom_q[atoms]                  # (M,)
        v = self.atom_v[atoms]                  # (M,)
        R_sel = self.R.index_select(0, q)       # (M, n, d)
        C_sel = self.C.index_select(1, v)       # (d, M)
        return torch.einsum("mnd,dm->nm", R_sel, C_sel)

    def matvec(self, a: torch.Tensor) -> torch.Tensor:
        return matvec_with_matrix(self.explicit_matrix(), a)

    def rmatvec(self, r: torch.Tensor) -> torch.Tensor:
        return rmatvec_with_matrix(self.explicit_matrix(), r)

    def constraint_items(self) -> list[tuple[str, torch.Tensor, str]]:
        out: list[tuple[str, torch.Tensor, str]] = []
        if isinstance(self.R, nn.Parameter):
            out.append(("R", self.R.data, self.constraints.R))
        if isinstance(self.C, nn.Parameter):
            out.append(("C", self.C.data, self.constraints.C))
        return out


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
        return matvec_with_matrix(self.explicit_matrix(), a)

    def rmatvec(self, r: torch.Tensor) -> torch.Tensor:
        return rmatvec_with_matrix(self.explicit_matrix(), r)

    def encode(self, counts: torch.Tensor) -> torch.Tensor:
        """Map a count vector a to the noiseless transmitted signal y = Phi a."""
        return self.matvec(counts)

    def apply_constraints(self) -> None:
        items: list[tuple[str, torch.Tensor, str]] = []
        for i, c in enumerate(self.components):
            for name, t, kind in c.constraint_items():
                items.append((f"comp{i}.{name}", t, kind))
        apply_constraints(items)


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
