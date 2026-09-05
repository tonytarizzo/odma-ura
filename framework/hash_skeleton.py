"""Materialised certification controls for compact sparse support generators.

The scalable object in this module is the support rule, not the explicit
``n x 2^B`` matrix used by the current D0/D1 decoders.  At materialisable B we
construct that matrix so generated supports can be compared fairly with the
existing arbitrary sparse-global control.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .core import ComponentSpec, URASpec
from .initializers import nonzero_gaussian


HASH_SKELETON_FAMILIES = (
    "sparse_iid_fixed",
    "hash_table_random_fixed",
    "hash_linear_random_fixed",
    "hash_linear_selected_fixed",
)


def gf2_rank(matrix: torch.Tensor) -> int:
    """Return the row rank of a binary matrix over GF(2)."""
    work = torch.as_tensor(matrix, dtype=torch.uint8).clone().remainder_(2)
    rows, cols = work.shape
    rank = 0
    for col in range(cols):
        candidates = torch.nonzero(work[rank:, col], as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            saved = work[rank].clone(); work[rank] = work[pivot]; work[pivot] = saved
        eliminate = torch.nonzero(work[:, col], as_tuple=False).reshape(-1)
        eliminate = eliminate[eliminate != rank]
        if eliminate.numel():
            work[eliminate] ^= work[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def all_message_bits(payload_bits: int) -> torch.Tensor:
    """Return all binary messages as a ``(2^B,B)`` tensor, least-significant bit first."""
    B = int(payload_bits)
    if B <= 0 or B > 62:
        raise ValueError(f"payload_bits must lie in [1,62], got {B}")
    messages = torch.arange(1 << B, dtype=torch.int64).unsqueeze(1)
    return ((messages >> torch.arange(B, dtype=torch.int64)) & 1).to(torch.uint8)


def linear_hash_bins(A: torch.Tensor, b: torch.Tensor, message_bits: torch.Tensor) -> torch.Tensor:
    """Evaluate ``h_t(w)=A_t w+b_t`` and return integer bins with shape ``(T,num_messages)``."""
    A = torch.as_tensor(A, dtype=torch.uint8)
    b = torch.as_tensor(b, dtype=torch.uint8)
    message_bits = torch.as_tensor(message_bits, dtype=torch.uint8)
    if A.ndim != 3:
        raise ValueError(f"A must have shape (T,r,B), got {tuple(A.shape)}")
    T, r, B = A.shape
    if b.shape != (T, r) or message_bits.ndim != 2 or message_bits.shape[1] != B:
        raise ValueError(f"expected b=({T},{r}) and message_bits=(M,{B}); got {tuple(b.shape)}, {tuple(message_bits.shape)}")
    bits = (torch.einsum("trb,mb->trm", A.to(torch.int64), message_bits.to(torch.int64))
            + b.to(torch.int64).unsqueeze(-1)).remainder_(2)
    weights = (1 << torch.arange(r, dtype=torch.int64)).reshape(1, r, 1)
    return torch.sum(bits * weights, dim=1)


def random_linear_hash_bank(payload_bits: int, tables: int, bin_bits: int,
                            generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample balanced affine hashes whose complete tuple is injective, without enumerating messages."""
    B, T, r = int(payload_bits), int(tables), int(bin_bits)
    if r > B:
        raise ValueError(f"bin_bits r={r} cannot exceed payload bits B={B}")
    if T * r < B:
        raise ValueError(f"injective hash tuple requires T*r >= B, got {T}*{r} < {B}")
    for _ in range(10_000):
        blocks = []
        for _table in range(T):
            for _ in range(1_000):
                block = torch.randint(2, (r, B), generator=generator, dtype=torch.uint8)
                if gf2_rank(block) == r:
                    blocks.append(block); break
            else:
                raise RuntimeError("could not sample a full-row-rank binary hash block")
        A = torch.stack(blocks)
        if gf2_rank(A.reshape(T * r, B)) == B:
            b = torch.randint(2, (T, r), generator=generator, dtype=torch.uint8)
            return A, b
    raise RuntimeError("could not sample a jointly injective binary hash bank")


def linear_collision_diagnostics(A: torch.Tensor) -> dict:
    """Enumerate the XOR-difference collision spectrum; intended for certification at small B."""
    A = torch.as_tensor(A, dtype=torch.uint8)
    T, r, B = A.shape
    if B > 20:
        raise ValueError("exact collision enumeration is intentionally restricted to B <= 20")
    differences = all_message_bits(B)[1:]
    bins = linear_hash_bins(A, torch.zeros(T, r, dtype=torch.uint8), differences)
    collisions = (bins == 0).sum(dim=0).to(torch.float64)
    values = collisions.numpy()
    maximum = int(collisions.max())
    return {
        "num_nonzero_differences": int(values.size),
        "collision_tables_mean": float(values.mean()),
        "collision_tables_std": float(values.std()),
        "collision_tables_q95": float(np.quantile(values, 0.95)),
        "collision_tables_q99": float(np.quantile(values, 0.99)),
        "collision_tables_q999": float(np.quantile(values, 0.999)),
        "collision_tables_max": maximum,
        "num_differences_at_max": int((values == maximum).sum()),
        "minimum_table_distance": int(T - maximum),
        "maximum_support_overlap_fraction": float(maximum / T),
        "collision_sum_squares": float(np.square(values).sum()),
    }


def _collision_score(diagnostics: dict) -> tuple:
    return (diagnostics["collision_tables_max"], diagnostics["num_differences_at_max"],
            diagnostics["collision_tables_q999"], diagnostics["collision_tables_q99"],
            diagnostics["collision_sum_squares"])


def select_linear_hash_bank(payload_bits: int, tables: int, bin_bits: int, num_candidates: int,
                            generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Choose the best of random full-rank affine hashes by exact collision-spectrum geometry."""
    if num_candidates <= 0:
        raise ValueError(f"num_candidates must be positive, got {num_candidates}")
    best_A = best_b = None
    best_diagnostics = first_diagnostics = None
    best_score = None; best_index = -1
    for index in range(int(num_candidates)):
        A, b = random_linear_hash_bank(payload_bits, tables, bin_bits, generator)
        diagnostics = linear_collision_diagnostics(A)
        if first_diagnostics is None:
            first_diagnostics = diagnostics
        score = _collision_score(diagnostics)
        if best_score is None or score < best_score:
            best_A, best_b, best_diagnostics = A, b, diagnostics
            best_score, best_index = score, index
    assert best_A is not None and best_b is not None and best_diagnostics is not None and first_diagnostics is not None
    return best_A, best_b, {
        "search_candidates": int(num_candidates), "selected_candidate_index": int(best_index),
        "candidate_zero_collision_geometry": first_diagnostics,
        "selected_collision_geometry": best_diagnostics,
    }


def _iid_sparse_rows(n: int, tables: int, num_messages: int, generator: torch.Generator) -> torch.Tensor:
    rows = torch.empty(tables, num_messages, dtype=torch.long)
    for start in range(0, num_messages, 2048):
        stop = min(start + 2048, num_messages)
        keys = torch.rand(stop - start, n, generator=generator)
        rows[:, start:stop] = torch.topk(keys, tables, dim=1, largest=False, sorted=False).indices.transpose(0, 1)
    return rows


def _balanced_table_rows(tables: int, bins_per_table: int, num_messages: int,
                         generator: torch.Generator) -> torch.Tensor:
    if num_messages % bins_per_table:
        raise ValueError(f"balanced table control requires M divisible by R, got M={num_messages}, R={bins_per_table}")
    rows = torch.empty(tables, num_messages, dtype=torch.long)
    base = torch.arange(num_messages) % bins_per_table
    for table in range(tables):
        bins = base[torch.randperm(num_messages, generator=generator)]
        rows[table] = table * bins_per_table + bins
    return rows


def materialize_sparse_codebook(rows: torch.Tensor, amplitudes: torch.Tensor, n: int) -> torch.Tensor:
    """Materialise columns from generated row indices and already-normalised amplitudes."""
    rows = torch.as_tensor(rows, dtype=torch.long)
    amplitudes = torch.as_tensor(amplitudes)
    if rows.shape != amplitudes.shape or rows.ndim != 2:
        raise ValueError(f"rows and amplitudes must share shape (T,M), got {tuple(rows.shape)}, {tuple(amplitudes.shape)}")
    T, M = rows.shape
    if int(rows.min()) < 0 or int(rows.max()) >= n:
        raise ValueError(f"generated row outside [0,{n})")
    if T > 1:
        sorted_rows = torch.sort(rows, dim=0).values
        if bool((sorted_rows[1:] == sorted_rows[:-1]).any()):
            raise ValueError("each codeword must select distinct physical rows")
    codebook = torch.zeros(n, M, dtype=amplitudes.dtype)
    columns = torch.arange(M).repeat(T)
    codebook[rows.reshape(-1), columns] = amplitudes.reshape(-1)
    return codebook


def hash_skeleton_component_specs(spec: URASpec, family: str, support_size: int, seed: int,
                                  search_candidates: int = 128,
                                  learn_amplitudes: bool = False) -> tuple[list[ComponentSpec], dict]:
    """Build one B-small certification codebook while retaining its compact support-rule metadata."""
    if family not in HASH_SKELETON_FAMILIES:
        raise ValueError(f"unknown hash-skeleton family '{family}'")
    B, n, M, T = int(spec.payload_bits), int(spec.n), int(spec.num_codewords), int(support_size)
    if M != 1 << B:
        raise ValueError(f"hash-skeleton certification requires M=2^B, got M={M}, B={B}")
    if T <= 0 or T > n:
        raise ValueError(f"support size T must lie in [1,{n}], got {T}")
    structure_seed, amplitude_seed = int(seed) + 310_003, int(seed) + 410_009
    structure_generator = torch.Generator().manual_seed(structure_seed)
    amplitude_generator = torch.Generator().manual_seed(amplitude_seed)
    construction = {
        "family": family, "payload_bits": B, "n": n, "num_messages": M, "support_size": T,
        "structure_seed": structure_seed, "amplitude_seed": amplitude_seed,
        "amplitude_pairing_key": f"gaussian_seed_{amplitude_seed}_shape_{T}x{M}",
        "exact_column_energy": True, "materialised_for_small_B_certification": True,
        "scalable_claim_applies_to_support_rule_only": True,
        "learnable_amplitudes_on_fixed_support": bool(learn_amplitudes),
    }

    if family == "sparse_iid_fixed":
        rows = _iid_sparse_rows(n, T, M, structure_generator)
        construction.update({
            "support_rule": "iid_uniform_without_replacement", "one_position_per_predetermined_table": False,
            "log2_support_family_size": float((math.lgamma(n + 1) - math.lgamma(T + 1) - math.lgamma(n - T + 1)) / math.log(2.0)),
        })
    else:
        if n % T:
            raise ValueError(f"table construction requires support size T to divide n={n}, got T={T}")
        R = n // T
        if R <= 0 or R & (R - 1):
            raise ValueError(f"bins per table R=n/T must be a power of two, got R={R}")
        r = int(math.log2(R))
        construction.update({
            "num_tables": T, "bins_per_table": R, "bin_bits": r,
            "one_position_per_predetermined_table": True, "log2_support_family_size": float(T * r),
        })
        if family == "hash_table_random_fixed":
            rows = _balanced_table_rows(T, R, M, structure_generator)
            construction["support_rule"] = "balanced_random_table_assignment"
        else:
            candidates = 1 if family == "hash_linear_random_fixed" else int(search_candidates)
            A, b, search = select_linear_hash_bank(B, T, r, candidates, structure_generator)
            bins = linear_hash_bins(A, b, all_message_bits(B))
            rows = torch.arange(T, dtype=torch.long).unsqueeze(1) * R + bins
            construction.update({
                "support_rule": "affine_binary_linear_hash", "A_shape": list(A.shape), "b_shape": list(b.shape),
                "A": A.tolist(), "b": b.tolist(), "per_table_rank": [gf2_rank(block) for block in A],
                "stacked_rank": gf2_rank(A.reshape(T * r, B)), "support_rule_storage_bits": int(T * r * (B + 1)),
                **search,
            })

    amplitudes = nonzero_gaussian((T, M), torch.float32, amplitude_generator)
    amplitudes = amplitudes / amplitudes.norm(dim=0, keepdim=True).clamp_min(1e-12)
    codebook = materialize_sparse_codebook(rows, amplitudes, n)
    support_tuples = torch.unique(torch.sort(rows.transpose(0, 1), dim=1).values, dim=0).shape[0]
    row_load = (codebook != 0).sum(dim=1)
    construction.update({
        "distinct_support_tuples": int(support_tuples), "support_tuple_injective": bool(support_tuples == M),
        "row_load_min": int(row_load.min()), "row_load_max": int(row_load.max()),
    })
    components = [ComponentSpec(Q=1, d=n, V=M, N=M, R_init="identity", C_init="explicit", U_init="all_pairs",
                                T_init="identity", learn_R=False, learn_C=bool(learn_amplitudes), explicit_C=codebook,
                                fixed_C_support=(codebook != 0) if learn_amplitudes else None)]
    return components, construction
