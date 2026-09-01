"""Initialisation strategies for the (R, C, U, T) factors of one component.

Each builder takes a `torch.Generator` so experiments stay reproducible. The
caller is responsible for placing the returned tensors on the right device.
"""

from __future__ import annotations

import math
import torch


def nonzero_gaussian(shape: tuple[int, ...], dtype: torch.dtype,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw Gaussian values while preserving an intended exact nonzero support."""
    values = torch.randn(shape, dtype=dtype, generator=generator)
    zeros = values == 0
    while bool(zeros.any()):
        values[zeros] = torch.randn((int(zeros.sum()),), dtype=dtype, generator=generator)
        zeros = values == 0
    return values


# --- R: operator/resource bank ----------------------------------------------


def init_R(strategy: str, Q: int, n: int, d: int, dtype: torch.dtype,
           generator: torch.Generator | None = None,
           value: torch.Tensor | None = None) -> torch.Tensor:
    if Q <= 0 or n <= 0 or d <= 0:
        raise ValueError(f"R requires Q,n,d > 0; got Q={Q}, n={n}, d={d}")
    if strategy == "random_gaussian":
        R = torch.randn(Q, n, d, dtype=dtype, generator=generator) / math.sqrt(d)
    elif strategy == "random_sign_diagonal":
        if d != n:
            raise ValueError(f"random_sign_diagonal requires d=n, got d={d}, n={n}")
        signs = torch.randint(0, 2, (Q, n), generator=generator)
        R = (2 * signs - 1).to(dtype=dtype)
    elif strategy == "random_phase_diagonal":
        if d != n:
            raise ValueError(f"random_phase_diagonal requires d=n, got d={d}, n={n}")
        if not dtype.is_complex:
            raise ValueError("random_phase_diagonal requires a complex dtype; use random_sign_diagonal for real experiments")
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        phase = 2.0 * math.pi * torch.rand(Q, n, dtype=real_dtype, generator=generator)
        R = torch.polar(torch.ones_like(phase), phase).to(dtype=dtype)
    elif strategy == "random_placements":
        R = torch.zeros(Q, n, d, dtype=dtype)
        rng_np = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        for q in range(Q):
            perm = torch.randperm(n, generator=rng_np)[:d]
            perm, _ = torch.sort(perm)
            R[q, perm, torch.arange(d)] = 1.0
    elif strategy == "explicit":
        if value is None:
            raise ValueError("explicit R initialisation requires value")
        R = torch.as_tensor(value, dtype=dtype).clone()
        if R.shape != (Q, n, d):
            raise ValueError(f"explicit R must have shape ({Q},{n},{d}), got {tuple(R.shape)}")
    elif strategy == "identity":
        if d != n or Q != 1:
            raise ValueError(f"identity R requires Q=1, n=d; got Q={Q}, n={n}, d={d}")
        R = torch.eye(n, dtype=dtype).unsqueeze(0)
    else:
        raise ValueError(f"unknown R_init '{strategy}'")
    return R


# --- C: local codebook, shape (d, V) ---------------------------------------


def init_C(strategy: str, d: int, V: int, dtype: torch.dtype,
           generator: torch.Generator | None = None,
           value: torch.Tensor | None = None) -> torch.Tensor:
    if d <= 0 or V <= 0:
        raise ValueError(f"C requires d,V > 0; got d={d}, V={V}")
    if strategy == "random_gaussian":
        C = nonzero_gaussian((d, V), dtype, generator)
        C = C / C.norm(dim=0, keepdim=True).clamp_min(1e-12)
    elif strategy == "explicit":
        if value is None:
            raise ValueError("explicit C initialisation requires value")
        C = torch.as_tensor(value, dtype=dtype).clone()
        if C.shape != (d, V):
            raise ValueError(f"explicit C must have shape ({d},{V}), got {tuple(C.shape)}")
    elif strategy == "identity":
        if d != V:
            raise ValueError(f"identity C requires d=V; got d={d}, V={V}")
        C = torch.eye(d, dtype=dtype)
    else:
        raise ValueError(f"unknown C_init '{strategy}'")
    return C


# --- U: validity, stored as (atom_q, atom_v) arrays of length N -----------


def init_U(strategy: str, Q: int, V: int, N: int | None,
           generator: torch.Generator | None = None,
           atom_q_value: torch.Tensor | None = None,
           atom_v_value: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy == "all_pairs":
        atom_q = torch.arange(Q).repeat_interleave(V)
        atom_v = torch.arange(V).repeat(Q)
    elif strategy == "random_subset":
        if N is None or N <= 0 or N > Q * V:
            raise ValueError(f"random_subset requires 0 < N <= Q*V={Q * V}; got N={N}")
        idx = torch.randperm(Q * V, generator=generator)[:N]
        atom_q = idx // V
        atom_v = idx % V
    elif strategy == "explicit":
        if atom_q_value is None or atom_v_value is None:
            raise ValueError("explicit U initialisation requires atom_q_value and atom_v_value")
        atom_q = torch.as_tensor(atom_q_value, dtype=torch.long)
        atom_v = torch.as_tensor(atom_v_value, dtype=torch.long)
        if atom_q.shape != atom_v.shape or atom_q.ndim != 1:
            raise ValueError("explicit U requires equal-length 1-D atom_q and atom_v")
        if int(atom_q.max()) >= Q or int(atom_q.min()) < 0:
            raise ValueError("atom_q out of range")
        if int(atom_v.max()) >= V or int(atom_v.min()) < 0:
            raise ValueError("atom_v out of range")
    else:
        raise ValueError(f"unknown U_init '{strategy}'")
    return atom_q.long(), atom_v.long()


# --- T: message-to-atom map, length M --------------------------------------


def init_T(strategy: str, N: int, M: int,
           generator: torch.Generator | None = None,
           value: torch.Tensor | None = None) -> torch.Tensor:
    if strategy == "identity":
        if N != M:
            raise ValueError(f"identity T requires N=M; got N={N}, M={M}")
        t = torch.arange(M)
    elif strategy == "round_robin":
        t = torch.arange(M) % N
    elif strategy == "random":
        t = torch.randint(0, N, (M,), generator=generator)
    elif strategy == "explicit":
        if value is None:
            raise ValueError("explicit T initialisation requires value")
        t = torch.as_tensor(value, dtype=torch.long).clone()
        if t.shape != (M,):
            raise ValueError(f"explicit T must have shape ({M},), got {tuple(t.shape)}")
        if int(t.max()) >= N or int(t.min()) < 0:
            raise ValueError("explicit T entries out of range")
    else:
        raise ValueError(f"unknown T_init '{strategy}'")
    return t.long()
