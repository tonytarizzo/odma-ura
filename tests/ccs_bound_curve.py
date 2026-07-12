"""Paper-scale original NNLS/tree CCS validation against Polyanskiy.

Unlike ``tests/framework_ccs_test.py`` this script never materialises the global
Phi (n x 2^B) codebook, so it runs at B=100 where 2^B enumeration is impossible.
It is a direct, non-enumerative implementation of the Amalladinne-Vem-Soma-
Narayanan-Chamberland coupled compressed sensing scheme:

  * split B payload bits into L sections; each section carries some message bits
    plus random linear parity of the preceding message bits (the tree code),
  * transmit each coded section through one shared unit-norm Gaussian CS matrix
    of 2^J columns (J = coded bits per section),
  * per section, recover a K_a+10 candidate list by non-negative least squares,
  * stitch section lists and output a root only when its parity-consistent path is unique.

For each K we sweep Eb/N0, measure the count/multiset PUPE (fraction of active
users whose message is not in the recovered list), and report the smallest
Eb/N0 reaching a target PUPE. That empirical required-Eb/N0-vs-K curve is
overlaid on the Polyanskiy random-coding achievability bound (src/ura_bound.py).

The default dimensions match the original paper's finite experiment
(B=75, N=22517, L=11, J=14 for K_a<=125). The paper used a BCH-derived sensing
matrix and K-dependent optimised parity profiles; this script currently uses a
Gaussian matrix and either a supplied or uniform parity profile. Those remaining
differences are emitted in the validation report and must not be hidden when the
curve is compared with the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.equivalence_outputs import (  # noqa: E402
    default_section_bits,
    write_polyanskiy_outputs,
    write_required_ebn0_outputs,
    write_validation_report,
)
from tests.framework_ccs_test import (  # noqa: E402
    balanced_parity_bits,
    bits_to_int,
    int_to_bits,
    unit_columns,
)


PAPER_REFS = {
    "original_ccs": "https://avinashvem.github.io/unsourcedma_tree.pdf",
    "journal_ccs": "https://ieeexplore.ieee.org/document/9153051",
    "polyanskiy_isit2017": "https://people.lids.mit.edu/yp/homepage/data/isit17_mac.pdf",
}


@dataclass
class CCSCode:
    payload_bits: int
    n: int
    num_sections: int
    section_bits: int
    section_len: int
    message_bits: list[int]
    parity_bits: list[int]
    prefix_len: list[int]
    parity_masks: list[np.ndarray]
    section_codebooks: list[np.ndarray]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-B", "--payload-bits", type=int, default=75)
    p.add_argument("--n", type=int, default=22517, help="real channel uses; must be divisible by num-sections")
    p.add_argument("--num-sections", type=int, default=11)
    p.add_argument("--section-bits", type=int, default=14,
                   help="coded bits J per section (2^J columns); must cover message+parity")
    p.add_argument("--num-antennas", type=int, default=1)
    p.add_argument("--K-values", nargs="+", type=int, default=[25, 50, 75, 100, 125])
    p.add_argument("--ebn0-grid", nargs="+", type=float,
                   default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--list-extra", type=int, default=10,
                   help="per-section list size is K_a + list_extra; the paper uses 10")
    p.add_argument("--nnls-pool", type=int, default=0,
                   help="0 runs full NNLS as in the paper; positive values preselect that many columns by correlation")
    p.add_argument("--parity-bits", nargs="+", type=int, default=None,
                   help="L parity lengths including root zero; default distributes parity uniformly")
    p.add_argument("--max-tree-paths", type=int, default=200000)
    p.add_argument("--target-pupe", type=float, default=0.05)
    p.add_argument("--bound-grid", type=int, default=25)
    p.add_argument("--bound-num-pprime", type=int, default=25)
    p.add_argument("--skip-bounds", action="store_true")
    p.add_argument("--out-dir", default="results/ccs_original_paper_scale")
    return p.parse_args(argv)


def build_code(args: argparse.Namespace, rng: np.random.Generator) -> CCSCode:
    B, n, L = int(args.payload_bits), int(args.n), int(args.num_sections)
    if n % L != 0:
        raise ValueError(f"n={n} must be divisible by num_sections={L}")
    J = int(args.section_bits)
    parity_bits = list(args.parity_bits) if args.parity_bits is not None else balanced_parity_bits(B, L, J)
    if len(parity_bits) != L or parity_bits[0] != 0 or sum(parity_bits) != L * J - B:
        raise ValueError(f"invalid parity profile {parity_bits}; need L={L}, root zero, sum={L * J - B}")
    message_bits = [J - p for p in parity_bits]
    if sum(message_bits) != B:
        raise AssertionError("internal CCS bit allocation error")
    prefix_len, acc = [], 0
    for mb in message_bits:
        prefix_len.append(acc)
        acc += mb
    masks = []
    for ell in range(L):
        p_bits, prefix = parity_bits[ell], prefix_len[ell]
        mask = np.zeros((p_bits, B), dtype=np.uint8)
        for row in range(p_bits):
            coeffs = rng.integers(0, 2, size=prefix, dtype=np.uint8)
            if not np.any(coeffs):
                coeffs[int(rng.integers(0, prefix))] = 1
            mask[row, :prefix] = coeffs
        masks.append(mask)
    d = n // L
    V = 1 << J
    shared = unit_columns(rng.standard_normal((d, V)))
    codebooks = [shared for _ in range(L)]
    return CCSCode(B, n, L, J, d, message_bits, parity_bits, prefix_len, masks, codebooks)


def encode_user(code: CCSCode, payload: np.ndarray) -> list[int]:
    coded, pos = [], 0
    for ell in range(code.num_sections):
        mb, pb = code.message_bits[ell], code.parity_bits[ell]
        nxt = pos + mb
        msg_part = payload[pos:nxt]
        parity = (code.parity_masks[ell][:, :pos] @ payload[:pos]) % 2 if pb else np.zeros(0, dtype=np.uint8)
        coded.append(bits_to_int(np.concatenate([msg_part, parity.astype(np.uint8)])))
        pos = nxt
    return coded


def sample_signal(code: CCSCode, coded_per_user: list[list[int]], num_antennas: int,
                  ebn0_db: float, rng: np.random.Generator) -> np.ndarray:
    """Sum active users' section codewords, add AWGN over antennas, matched-filter.

    Unit-norm section columns give per-user energy L. For real AWGN with variance
    one, the paper uses P=2*B*(Eb/N0)/n, hence noise_var=L/(2*B*Eb/N0) when the
    signal amplitude is fixed to one. Independent antennas, when requested, are
    averaged and therefore add their explicit combining gain."""
    d, L = code.section_len, code.num_sections
    y = np.zeros(code.n, dtype=np.float64)
    for coded in coded_per_user:
        for ell in range(L):
            y[ell * d:(ell + 1) * d] += code.section_codebooks[ell][:, coded[ell]]
    energy = float(L)  # unit-norm columns, L sections per user
    noise_var = energy / (2.0 * code.payload_bits * (10.0 ** (float(ebn0_db) / 10.0)))
    Y = y[:, None] + rng.standard_normal((code.n, int(num_antennas))) * math.sqrt(noise_var)
    return np.mean(Y, axis=1)


def section_lists(code: CCSCode, y: np.ndarray, K: int, list_extra: int,
                  nnls_pool: int) -> list[np.ndarray]:
    d, V = code.section_len, 1 << code.section_bits
    list_size = max(1, min(V, int(K) + int(list_extra)))
    lists = []
    for ell in range(code.num_sections):
        C = code.section_codebooks[ell]
        yl = y[ell * d:(ell + 1) * d]
        if int(nnls_pool) > 0 and int(nnls_pool) < V:
            corr = C.T @ yl
            pool_size = max(list_size, int(nnls_pool))
            pool = np.argpartition(-corr, pool_size - 1)[:pool_size]
            x, _ = nnls(C[:, pool], yl)
            lists.append(pool[np.argsort(-x, kind="mergesort")[:list_size]].astype(np.int64))
        else:
            x, _ = nnls(C, yl)
            lists.append(np.argsort(-x, kind="mergesort")[:list_size].astype(np.int64))
    return lists


def parity_ok(code: CCSCode, ell: int, prefix_bits: np.ndarray, candidate_parity: np.ndarray) -> bool:
    if code.parity_bits[ell] == 0:
        return True
    expected = (code.parity_masks[ell][:, :prefix_bits.size] @ prefix_bits) % 2
    return bool(np.array_equal(expected.astype(np.uint8), candidate_parity.astype(np.uint8)))


def tree_survivors(code: CCSCode, lists: list[np.ndarray], K: int,
                   max_tree_paths: int) -> tuple[list[int], dict]:
    """Original root-wise tree decoder: output a root iff exactly one path survives."""
    survivors: set[int] = set()
    truncated_roots = ambiguous_roots = 0
    for root in lists[0]:
        truncated = False
        root_bits = int_to_bits(int(root), code.section_bits)
        paths = [root_bits[:code.message_bits[0]].copy()]
        for ell in range(1, code.num_sections):
            new_paths, mb = [], code.message_bits[ell]
            for prefix_bits in paths:
                for cand in lists[ell]:
                    coded = int_to_bits(int(cand), code.section_bits)
                    joined = np.concatenate([prefix_bits, coded[:mb]])
                    if parity_ok(code, ell, prefix_bits, coded[mb:]):
                        new_paths.append(joined)
                        if len(new_paths) >= max_tree_paths:
                            truncated = True
                            break
                if truncated:
                    break
            paths = new_paths
            if not paths or truncated:
                break
        if truncated:
            truncated_roots += 1
        elif len(paths) == 1:
            survivors.add(bits_to_int(paths[0]))
        elif len(paths) > 1:
            ambiguous_roots += 1
    return list(survivors), {"truncated_roots": truncated_roots, "ambiguous_roots": ambiguous_roots}


def run_point(code: CCSCode, args: argparse.Namespace, K: int, ebn0_db: float,
              rng: np.random.Generator) -> tuple[float, float, int, dict]:
    payloads = [rng.integers(0, 2, size=code.payload_bits, dtype=np.uint8) for _ in range(int(K))]
    true_msgs = [bits_to_int(p) for p in payloads]
    coded_per_user = [encode_user(code, p) for p in payloads]
    y = sample_signal(code, coded_per_user, args.num_antennas, ebn0_db, rng)
    lists = section_lists(code, y, int(K), args.list_extra, args.nnls_pool)
    survivors, tree_meta = tree_survivors(code, lists, int(K), args.max_tree_paths)
    decoded = set(survivors)
    missed = sum(1 for m in true_msgs if m not in decoded)  # count/multiset PUPE
    pupe = missed / float(K)
    false_alarms = len(decoded - set(true_msgs))
    return pupe, false_alarms / max(len(decoded), 1), len(survivors), tree_meta


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.num_antennas) != 1:
        raise SystemExit("paper-scale CCS validation currently requires --num-antennas 1")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    parity = list(args.parity_bits) if args.parity_bits is not None else balanced_parity_bits(
        args.payload_bits, args.num_sections, args.section_bits)
    print(f"CCS B={args.payload_bits}, n={args.n}, L={args.num_sections}, J={args.section_bits}, "
          f"V=2^J={1 << int(args.section_bits)}, antennas={args.num_antennas}")
    print(f"message_bits={[args.section_bits - p for p in parity]} parity_bits={parity}")
    print(f"K values={args.K_values}, Eb/N0 grid={args.ebn0_grid}, seeds={seeds}, target PUPE={args.target_pupe}")

    # Build each seed's (large) codebook once and reuse it across all K / Eb/N0 points;
    # trial randomness (users, noise) is drawn from a per-(seed,K,Eb/N0) generator.
    rows: list[dict] = []
    accum: dict[tuple[int, float], list[tuple[float, float, int]]] = {
        (int(K), float(e)): [] for K in args.K_values for e in args.ebn0_grid}
    for si, seed in enumerate(seeds):
        code = build_code(args, np.random.default_rng(int(seed)))
        for K in args.K_values:
            for ebn0_db in args.ebn0_grid:
                # Common random numbers across Eb/N0: messages and standard-normal
                # noise are identical, only the noise scale changes.
                trng = np.random.default_rng((int(seed), int(K)))
                pupe, far, surv, tree_meta = run_point(code, args, int(K), float(ebn0_db), trng)
                accum[(int(K), float(ebn0_db))].append((pupe, far, surv))
                rows.append({"family": "ccs", "construction": "coupled-CS", "K": int(K),
                             "ebn0_db": float(ebn0_db), "seed": int(seed), "pupe": pupe,
                             "false_alarm_rate": far, "num_survivors": surv, "tree_meta": tree_meta})
        del code
        print(f"seed {si + 1}/{len(seeds)} done", flush=True)

    points = []
    for K in args.K_values:
        for ebn0_db in args.ebn0_grid:
            vals = accum[(int(K), float(ebn0_db))]
            mean_pupe = float(np.mean([v[0] for v in vals]))
            points.append({"family": "ccs", "preset": "ccs", "construction": "coupled-CS",
                           "K": int(K), "ebn0_db": float(ebn0_db), "num_trials": len(vals),
                           "mean_pupe": mean_pupe, "mean_false_alarm_rate": float(np.mean([v[1] for v in vals]))})
            print(f"K={int(K):<4d} Eb/N0={float(ebn0_db):>5.2f} mean PUPE={mean_pupe:.4f}", flush=True)

    monotonicity = []
    for K in args.K_values:
        curve = sorted([p for p in points if p["K"] == int(K)], key=lambda p: p["ebn0_db"])
        increases = [curve[i + 1]["mean_pupe"] - curve[i]["mean_pupe"] for i in range(len(curve) - 1)]
        row = {"K": int(K), "violations": int(sum(v > 1e-12 for v in increases)),
               "max_pupe_increase": float(max([0.0, *increases]))}
        monotonicity.append(row)
        if row["violations"]:
            print(f"Warning: K={K} has {row['violations']} empirical PUPE monotonicity violation(s); "
                  f"max increase={row['max_pupe_increase']:.4f}")

    polyanskiy_rows = []
    if not args.skip_bounds:
        bound_args = SimpleNamespace(n=int(args.n), payload_bits=int(args.payload_bits),
                                     num_antennas=int(args.num_antennas), K_values=list(args.K_values))
        polyanskiy_rows = write_polyanskiy_outputs(
            bound_args, out_dir, target_pupe=args.target_pupe,
            grid=args.bound_grid, num_pprime=args.bound_num_pprime, axis="physical")
    required = write_required_ebn0_outputs(
        points, polyanskiy_rows, out_dir, target_pupe=args.target_pupe,
        preferred={"ccs": "coupled-CS"},
        title=f"CCS (coupled-CS) vs Polyanskiy, B={args.payload_bits}, n={args.n}, PUPE<={args.target_pupe:g}")
    notes = [
        f"Non-enumerative coupled-CS at B={args.payload_bits}: {args.num_sections} sections, "
        f"2^{args.section_bits} columns each; the 2^B global codebook is never formed.",
        "Parity checks use preceding message fragments only; the same sensing matrix is reused in every section.",
        f"Per-section detector uses {'full NNLS' if args.nnls_pool == 0 else f'correlation-preselected NNLS pool {args.nnls_pool}'} and a K_a+{args.list_extra} list.",
        "The root-wise tree decoder rejects ambiguous roots; there is no oracle global OMP refinement.",
        "The empirical and Polyanskiy curves use the physical real-AWGN Eb/N0 convention.",
        "Remaining paper mismatch: Gaussian sensing matrix instead of the paper's BCH-derived matrix.",
        "Remaining paper mismatch: the default uniform parity profile is not the paper's K-dependent CVX-optimised profile unless --parity-bits is supplied.",
    ]
    write_validation_report(
        out_dir, scheme=f"CCS coupled-CS required Eb/N0 (B={args.payload_bits})",
        max_phi_err=float("nan"), decoded_match=False, paper_refs=PAPER_REFS, notes=notes,
        polyanskiy_written=not args.skip_bounds, required_ebn0_written=True)
    payload = {"args": vars(args), "paper_refs": PAPER_REFS, "points": points, "rows": rows,
               "monotonicity": monotonicity,
               "polyanskiy_bounds": polyanskiy_rows, "required_ebn0": required}
    (out_dir / "ccs_bound_curve_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out_dir / 'ccs_bound_curve_summary.json'}")
    if not args.skip_bounds:
        print(f"Wrote {out_dir / 'polyanskiy_bounds.png'}")
    print(f"Wrote {out_dir / 'required_ebn0_with_bounds.png'}")
    print(f"Wrote {out_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
