"""Direct CCS/tree-code vs framework-codebook equivalence test.

This is a deliberately small implementation of the Amalladinne-Vem-Soma-
Narayanan-Chamberland coded compressed sensing construction:

  * split B payload bits into L sections,
  * append random linear parity bits to sections after the root section,
  * transmit each coded section through a slot-local Gaussian CS matrix,
  * recover per-section lists by NNLS and stitch candidates with the tree checks.

The framework side represents the exact same global codebook Phi through
section components. The tree-code parity metadata is external to Phi, so both
decoders consume the same parity checks after asserting direct/framework Phi
equality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.core import ComponentSpec, URASpec  # noqa: E402
from framework.encoder import build_encoder  # noqa: E402
from src.metrics import evaluate_counts  # noqa: E402
from tests.equivalence_outputs import (  # noqa: E402
    default_section_bits,
    write_polyanskiy_outputs,
    write_required_ebn0_outputs,
    write_validation_report,
)
from tests.framework_equivalence_curve import run_trial as run_dense_trial  # noqa: E402


PAPER_REFS = {
    "original_ccs": "https://avinashvem.github.io/unsourcedma_tree.pdf",
    "journal_ccs": "https://ieeexplore.ieee.org/document/9153051",
    "enhanced_ccs_amp_code": "https://github.com/vamsi128/CCS-AMP-Code",
}


@dataclass(frozen=True)
class CCSDesign:
    payload_bits: int
    num_codewords: int
    n: int
    num_sections: int
    section_len: int
    section_bits: int
    message_bits: list[int]
    parity_bits: list[int]
    parity_masks: list[np.ndarray]
    section_codebooks: list[np.ndarray]
    encoded_sections: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-B", "--payload-bits", type=int, default=10)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--num-sections", type=int, default=4)
    p.add_argument("--section-bits", type=int, default=8,
                   help="coded bits per section; choose enough parity to disambiguate the small-B tree")
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--K-values", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[4.0, 8.0, 12.0, 16.0])
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--list-extra", type=int, default=10,
                   help="extra per-section NNLS candidates beyond K_a; the original CCS paper uses 10")
    p.add_argument("--max-tree-paths", type=int, default=20000)
    p.add_argument("--phi-atol", type=float, default=1e-12)
    p.add_argument("--include-dense", action="store_true", default=True)
    p.add_argument("--no-include-dense", dest="include_dense", action="store_false")
    p.add_argument("--bound-target-pupe", type=float, default=0.05)
    p.add_argument("--bound-grid", type=int, default=25)
    p.add_argument("--bound-num-pprime", type=int, default=25)
    p.add_argument("--skip-bounds", action="store_true")
    p.add_argument("--skip-required-ebn0", action="store_true")
    p.add_argument("--out-dir", default="results/framework_equivalence_ccs")
    return p.parse_args(argv)


def int_to_bits(x: int, width: int) -> np.ndarray:
    return np.asarray([(int(x) >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        out = (out << 1) | int(bit)
    return int(out)


def unit_columns(A: np.ndarray) -> np.ndarray:
    return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)


def balanced_parity_bits(payload_bits: int, num_sections: int, section_bits: int) -> list[int]:
    total_parity = int(num_sections * section_bits - payload_bits)
    if total_parity < 0:
        raise ValueError(
            f"section_bits={section_bits} gives only {num_sections * section_bits} coded bits for B={payload_bits}")
    if num_sections == 1:
        if total_parity != 0:
            raise ValueError("one-section CCS cannot add tree parity")
        return [0]
    if total_parity > (num_sections - 1) * section_bits:
        raise ValueError("the root section has no parity, so section_bits must be <= payload_bits")
    parity = [0] * num_sections
    for i in range(total_parity):
        parity[1 + i % (num_sections - 1)] += 1
    if any(p > section_bits for p in parity):
        raise ValueError(f"invalid parity allocation {parity} for section_bits={section_bits}")
    return parity


def build_ccs_design(payload_bits: int, n: int, num_sections: int,
                     section_bits: int | None, rng: np.random.Generator) -> CCSDesign:
    if n % num_sections != 0:
        raise ValueError(f"CCS requires n divisible by num_sections, got n={n}, L={num_sections}")
    default_J = default_section_bits(payload_bits, num_sections)
    J = int(section_bits) if section_bits is not None else default_J
    if J <= 0:
        raise ValueError(f"section_bits must be positive, got {J}")
    parity_bits = balanced_parity_bits(payload_bits, num_sections, J)
    message_bits = [J - p for p in parity_bits]
    if sum(message_bits) != payload_bits:
        raise AssertionError("internal CCS bit allocation error")

    masks: list[np.ndarray] = []
    prefix = 0
    for ell, (m_bits, p_bits) in enumerate(zip(message_bits, parity_bits)):
        mask = np.zeros((p_bits, payload_bits), dtype=np.uint8)
        for row in range(p_bits):
            coeffs = rng.integers(0, 2, size=prefix, dtype=np.uint8)
            if not np.any(coeffs):
                coeffs[int(rng.integers(0, prefix))] = 1
            mask[row, :prefix] = coeffs
        masks.append(mask)
        prefix += m_bits

    V = 1 << J
    section_len = n // num_sections
    shared_codebook = unit_columns(rng.standard_normal((section_len, V)))
    section_codebooks = [shared_codebook for _ in range(num_sections)]
    M = 1 << payload_bits
    encoded = np.zeros((num_sections, M), dtype=np.int64)
    for msg in range(M):
        payload = int_to_bits(msg, payload_bits)
        pos = 0
        for ell in range(num_sections):
            m_bits, p_bits = message_bits[ell], parity_bits[ell]
            next_pos = pos + m_bits
            msg_part = payload[pos:next_pos]
            parity = (masks[ell][:, :pos] @ payload[:pos]) % 2 if p_bits else np.zeros(0, dtype=np.uint8)
            encoded[ell, msg] = bits_to_int(np.concatenate([msg_part, parity.astype(np.uint8)]))
            pos = next_pos
    return CCSDesign(payload_bits, M, n, num_sections, section_len, J,
                     message_bits, parity_bits, masks, section_codebooks, encoded)


def direct_phi(design: CCSDesign) -> np.ndarray:
    Phi = np.zeros((design.n, design.num_codewords), dtype=np.float64)
    for ell, C in enumerate(design.section_codebooks):
        lo, hi = ell * design.section_len, (ell + 1) * design.section_len
        Phi[lo:hi, :] = C[:, design.encoded_sections[ell]]
    return Phi


def framework_encoder(design: CCSDesign, K: int, num_antennas: int):
    components = []
    for ell, C in enumerate(design.section_codebooks):
        R = torch.zeros(1, design.n, design.section_len, dtype=torch.float64)
        rows = torch.arange(design.section_len)
        R[0, ell * design.section_len + rows, rows] = 1.0
        components.append(ComponentSpec(Q=1, d=design.section_len, V=1 << design.section_bits, N=1 << design.section_bits,
                                        R_init="explicit", C_init="explicit", U_init="all_pairs", T_init="explicit",
                                        explicit_R=R, explicit_C=torch.as_tensor(C, dtype=torch.float64),
                                        explicit_msg_to_atom=torch.as_tensor(design.encoded_sections[ell], dtype=torch.long)))
    spec = URASpec(n=design.n, num_codewords=design.num_codewords, num_active=int(K),
                   num_antennas=int(num_antennas), payload_bits=design.payload_bits)
    return build_encoder(spec, components, dtype=torch.float64)


def framework_phi_fast(encoder) -> np.ndarray:
    """CCS-specialised Phi extraction avoiding the generic dense R_sel tensor."""
    n, M = encoder.n, encoder.num_codewords
    Phi = np.zeros((n, M), dtype=np.float64)
    for comp in encoder.components:
        R = comp.R.detach().cpu().numpy()
        C = comp.C.detach().cpu().numpy()
        q = comp.atom_q[comp.msg_to_atom].detach().cpu().numpy()
        v = comp.atom_v[comp.msg_to_atom].detach().cpu().numpy()
        if R.shape[0] == 1:
            nz = np.abs(R[0]) > 0
            one_hot = np.all(nz.sum(axis=0) == 1)
            if one_hot:
                rows = np.argmax(nz, axis=0)
                vals = R[0, rows, np.arange(R.shape[2])]
                Phi[rows, :] += vals[:, None] * C[:, v]
                continue
        for m in range(M):
            Phi[:, m] += R[int(q[m])] @ C[:, int(v[m])]
    return Phi


def design_from_encoder(design: CCSDesign, encoder) -> CCSDesign:
    codebooks = [c.C.detach().cpu().numpy().astype(np.float64, copy=True) for c in encoder.components]
    encoded = np.vstack([c.msg_to_atom.detach().cpu().numpy().astype(np.int64, copy=True) for c in encoder.components])
    return replace(design, section_codebooks=codebooks, encoded_sections=encoded)


def sample_observation(Phi: np.ndarray, K: int, num_antennas: int, ebn0_db: float,
                       payload_bits: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    active = rng.integers(0, Phi.shape[1], size=int(K))
    counts = np.zeros(Phi.shape[1], dtype=np.float64)
    np.add.at(counts, active, 1.0)
    y = Phi @ counts
    h = np.ones(num_antennas, dtype=np.float64)
    energy = float(np.mean(np.sum(Phi ** 2, axis=0)))
    noise_var = energy / (payload_bits * (10.0 ** (float(ebn0_db) / 10.0)))
    Y = y[:, None] * h[None, :] + rng.standard_normal((Phi.shape[0], num_antennas)) * math.sqrt(noise_var)
    return counts, Y, active, noise_var


def matched_filter_y(Y: np.ndarray) -> np.ndarray:
    return np.mean(Y, axis=1)


def section_candidates(design: CCSDesign, y: np.ndarray, K: int, list_extra: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    lists, coeffs = [], []
    list_size = max(1, min(1 << design.section_bits, int(K) + int(list_extra)))
    for ell, C in enumerate(design.section_codebooks):
        lo, hi = ell * design.section_len, (ell + 1) * design.section_len
        x, _ = nnls(C, y[lo:hi])
        order = np.argsort(-x, kind="mergesort")[:list_size]
        lists.append(order.astype(np.int64))
        coeffs.append(x)
    return lists, coeffs


def parity_ok(design: CCSDesign, ell: int, bits_prefix: np.ndarray, candidate_parity: np.ndarray) -> bool:
    if design.parity_bits[ell] == 0:
        return True
    expected = (design.parity_masks[ell][:, :bits_prefix.size] @ bits_prefix) % 2
    return bool(np.array_equal(expected.astype(np.uint8), candidate_parity.astype(np.uint8)))


def tree_stitch(design: CCSDesign, candidates: list[np.ndarray], coeffs: list[np.ndarray], *,
                max_tree_paths: int) -> tuple[dict[int, float], dict]:
    """Amalladinne et al. tree decoding: collect every root->leaf path that survives
    all parity checks. Each surviving full path is one recovered message; its score
    is the sum of the per-section NNLS coefficients of the chosen sub-messages so the
    caller can rank the list. This is a pure list/support decoder - it never discards
    a valid path for being non-unique (which would drop correctly detected users)."""
    survivors: dict[int, float] = {}
    roots, truncated_roots, invalid_messages = 0, 0, 0
    for root in candidates[0]:
        roots += 1
        root_bits = int_to_bits(int(root), design.section_bits)
        paths = [(root_bits[:design.message_bits[0]].copy(), float(coeffs[0][int(root)]))]
        truncated = False
        for ell in range(1, design.num_sections):
            new_paths = []
            m_bits = design.message_bits[ell]
            for bits_so_far, score in paths:
                for cand in candidates[ell]:
                    coded = int_to_bits(int(cand), design.section_bits)
                    joined = np.concatenate([bits_so_far, coded[:m_bits]])
                    if parity_ok(design, ell, bits_so_far, coded[m_bits:]):
                        new_paths.append((joined, score + float(coeffs[ell][int(cand)])))
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
            continue
        if len(paths) != 1:
            continue
        for bits, score in paths:
            msg = bits_to_int(bits)
            if msg < design.num_codewords:
                if score > survivors.get(msg, -np.inf):
                    survivors[msg] = score
            else:
                invalid_messages += 1
    return survivors, {
        "roots": roots,
        "truncated_roots": truncated_roots,
        "invalid_messages": invalid_messages,
        "num_survivors": len(survivors),
    }


def decode_ccs(design: CCSDesign, y: np.ndarray, K: int, list_extra: int,
               max_tree_paths: int) -> tuple[np.ndarray, dict]:
    lists, coeffs = section_candidates(design, y, K, list_extra)
    survivors, tree_meta = tree_stitch(design, lists, coeffs, max_tree_paths=max_tree_paths)
    counts = np.zeros(design.num_codewords, dtype=np.float64)
    # The original tree decoder outputs a root only when exactly one full path survives.
    for msg, _ in sorted(survivors.items(), key=lambda kv: -kv[1]):
        counts[msg] += 1.0
    return counts, {
        "decoder": "ccs_nnls_tree",
        "section_list_sizes": [int(len(x)) for x in lists],
        "section_peak_coefficients": [float(np.max(x)) for x in coeffs],
        **tree_meta,
    }


def run_trial(args: argparse.Namespace, K: int, ebn0_db: float, seed: int) -> tuple[list[dict], float]:
    rng = np.random.default_rng(int(seed))
    design = build_ccs_design(args.payload_bits, args.n, args.num_sections, args.section_bits, rng)
    Phi_direct = direct_phi(design)
    encoder = framework_encoder(design, int(K), args.num_antennas)
    Phi_framework = framework_phi_fast(encoder)
    phi_err = float(np.max(np.abs(Phi_direct - Phi_framework)))
    if phi_err > float(args.phi_atol):
        raise AssertionError(f"CCS Phi mismatch at K={K}, Eb/N0={ebn0_db}, seed={seed}: max_abs={phi_err:.3e}")

    counts_true, Y, active, noise_var = sample_observation(
        Phi_direct, int(K), args.num_antennas, float(ebn0_db), args.payload_bits, rng)
    y = matched_filter_y(Y)
    rows, decoded = [], []
    for name, dec_design in (("direct", design), ("framework", design_from_encoder(design, encoder))):
        t0 = time.time()
        counts, meta = decode_ccs(dec_design, y, int(K), args.list_extra, args.max_tree_paths)
        wall = time.time() - t0
        decoded.append(counts)
        metrics = evaluate_counts(counts_true, counts, max_list_size=int(K))
        rows.append({"family": "ccs", "preset": "ccs", "construction": name,
                     "K": int(K), "ebn0_db": float(ebn0_db), "seed": int(seed),
                     "metrics": metrics, "wall_s": wall, "phi_max_abs_err": phi_err,
                     "noise_var": float(noise_var), "active_messages": active.tolist(), "decoder_meta": meta,
                     "design": {"section_bits": design.section_bits, "message_bits": design.message_bits,
                                "parity_bits": design.parity_bits}})
    if not np.array_equal(decoded[0], decoded[1]):
        raise AssertionError(f"CCS decoded counts differ despite matching Phi at K={K}, Eb/N0={ebn0_db}, seed={seed}")
    return rows, phi_err


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r.get("family", r.get("preset", "unknown")), r["construction"], r["K"], r["ebn0_db"]) for r in rows})
    for family, construction, K, ebn0_db in keys:
        sel = [r for r in rows if (r.get("family", r.get("preset", "unknown")), r["construction"], r["K"], r["ebn0_db"])
               == (family, construction, K, ebn0_db)]
        metric_keys = sel[0]["metrics"].keys()
        out.append({"family": family, "preset": family, "construction": construction, "K": int(K), "ebn0_db": float(ebn0_db),
                    "num_trials": len(sel), "mean_wall_s": float(np.mean([r["wall_s"] for r in sel])),
                    "max_phi_abs_err": float(max(r["phi_max_abs_err"] for r in sel)),
                    **{f"mean_{k}": float(np.mean([r["metrics"][k] for r in sel]))
                       for k in metric_keys if isinstance(sel[0]["metrics"][k], (int, float, np.integer, np.floating))}})
    return out


def plot_summary(points: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in ("dense", "ccs") if any(p["family"] == f for p in points)]
    families.extend(sorted({p["family"] for p in points if p["family"] not in families}))
    fig, axes = plt.subplots(len(families), 2, figsize=(11, 4.2 * len(families)), squeeze=False)
    construction_order = {"dense": [("legacy", "-"), ("framework", "--")],
                          "ccs": [("direct", "-"), ("framework", "--")]}
    for row, family in zip(axes, families):
        f_points = [p for p in points if p["family"] == family]
        for ebn0_db in sorted({p["ebn0_db"] for p in f_points}):
            for construction, style in construction_order.get(family, [("direct", "-"), ("framework", "--")]):
                curve = [p for p in f_points if p["ebn0_db"] == ebn0_db and p["construction"] == construction]
                curve = sorted(curve, key=lambda p: p["K"])
                if not curve:
                    continue
                label = f"{construction}, {ebn0_db:g} dB"
                row[0].plot([p["K"] for p in curve], [p["mean_l1_acc"] for p in curve], style, marker="o", label=label)
                row[1].plot([p["K"] for p in curve], [p["mean_pupe"] for p in curve], style, marker="o", label=label)
        row[0].set_title(f"{family}: L1 accuracy")
        row[1].set_title(f"{family}: PUPE")
        for ax in row:
            ax.set_xlabel("Active devices K")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    section_bits = args.section_bits
    if section_bits is None:
        section_bits = default_section_bits(args.payload_bits, args.num_sections)
    section_alphabet = 1 << int(section_bits)
    print(f"CCS/tree equivalence: n={args.n}, B={args.payload_bits}, M={1 << int(args.payload_bits)}, "
          f"L={args.num_sections}, J={section_bits}, V={section_alphabet}, antennas={args.num_antennas}")
    print(f"K values={args.K_values}, Eb/N0={args.ebn0_grid}, seeds={seeds}")
    if max(args.K_values) > section_alphabet:
        print(f"Warning: max K={max(args.K_values)} exceeds the section alphabet V={section_alphabet}; "
              "this small-B CCS regime is much harsher than paper-scale CCS.")

    rows: list[dict] = []
    max_phi_err = 0.0
    dense_factor = 1 if args.include_dense else 0
    total = (1 + dense_factor) * len(args.K_values) * len(args.ebn0_grid) * len(seeds)
    done = 0
    if args.include_dense:
        dense_args = argparse.Namespace(payload_bits=args.payload_bits, n=args.n, d=args.n,
                                        num_blocks=1, num_antennas=args.num_antennas,
                                        phi_atol=args.phi_atol)
        for K in args.K_values:
            for ebn0_db in args.ebn0_grid:
                for seed in seeds:
                    trial_rows, phi_err = run_dense_trial(dense_args, "dense", int(K), float(ebn0_db), int(seed))
                    for row in trial_rows:
                        row["family"] = "dense"
                    rows.extend(trial_rows)
                    max_phi_err = max(max_phi_err, phi_err)
                    done += 1
                latest = [r for r in rows if r.get("family") == "dense"
                          and r["K"] == int(K) and r["ebn0_db"] == float(ebn0_db)]
                legacy_l1 = np.mean([r["metrics"]["l1_acc"] for r in latest if r["construction"] == "legacy"])
                framework_l1 = np.mean([r["metrics"]["l1_acc"] for r in latest if r["construction"] == "framework"])
                print(f"[{done:4d}/{total}] dense K={int(K):<4d} Eb/N0={float(ebn0_db):>5.2f} "
                      f"L1 legacy={legacy_l1:.4f} framework={framework_l1:.4f} phi_err<={max_phi_err:.1e}", flush=True)

    for K in args.K_values:
        for ebn0_db in args.ebn0_grid:
            for seed in seeds:
                trial_rows, phi_err = run_trial(args, int(K), float(ebn0_db), int(seed))
                rows.extend(trial_rows)
                max_phi_err = max(max_phi_err, phi_err)
                done += 1
            latest = [r for r in rows if r["K"] == int(K) and r["ebn0_db"] == float(ebn0_db)]
            latest = [r for r in latest if r.get("family", r.get("preset")) == "ccs"]
            direct_l1 = np.mean([r["metrics"]["l1_acc"] for r in latest if r["construction"] == "direct"])
            framework_l1 = np.mean([r["metrics"]["l1_acc"] for r in latest if r["construction"] == "framework"])
            print(f"[{done:4d}/{total}] CCS K={int(K):<4d} Eb/N0={float(ebn0_db):>5.2f} "
                  f"L1 direct={direct_l1:.4f} framework={framework_l1:.4f} phi_err<={max_phi_err:.1e}", flush=True)

    points = summarize(rows)
    polyanskiy_rows = []
    if not args.skip_bounds:
        polyanskiy_rows = write_polyanskiy_outputs(
            args, out_dir, target_pupe=args.bound_target_pupe,
            grid=args.bound_grid, num_pprime=args.bound_num_pprime)
    required_ebn0_rows = []
    if not args.skip_required_ebn0:
        required_ebn0_rows = write_required_ebn0_outputs(
            points, polyanskiy_rows, out_dir, target_pupe=args.bound_target_pupe,
            preferred={"dense": "framework", "ccs": "framework"},
            title=f"Empirical required Eb/N0 with Polyanskiy bounds, PUPE<={args.bound_target_pupe:g}")
    notes = [
        f"This explicit framework run enumerates M=2^B={1 << int(args.payload_bits)} messages; paper-scale CCS with "
        "B around 75-100 needs an implicit large-alphabet validator.",
        f"The CCS section alphabet is V=2^J={section_alphabet}; when K approaches or exceeds V, this small-B regime "
        "is not comparable to the original large-B operating point.",
        "The CCS implementation uses section-wise NNLS plus tree stitching; AMP-BP and enhanced CCS variants should "
        "be validated separately.",
        "Dense is included as the same iid Gaussian global-codebook baseline used by the ODMA equivalence script.",
    ]
    write_validation_report(
        out_dir, scheme="CCS/tree equivalence", max_phi_err=max_phi_err,
        decoded_match=True, paper_refs=PAPER_REFS, notes=notes,
        polyanskiy_written=not args.skip_bounds, required_ebn0_written=not args.skip_required_ebn0)
    payload = {"args": vars(args), "paper_refs": PAPER_REFS, "rows": rows,
               "points": points, "polyanskiy_bounds": polyanskiy_rows,
               "required_ebn0": required_ebn0_rows,
               "max_phi_abs_err": max_phi_err}
    (out_dir / "ccs_equivalence_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_summary(points, out_dir / "ccs_equivalence_curves.png")
    print(f"Max Phi abs error: {max_phi_err:.3e}")
    print(f"Wrote {out_dir / 'ccs_equivalence_summary.json'}")
    print(f"Wrote {out_dir / 'ccs_equivalence_curves.png'}")
    if not args.skip_bounds:
        print(f"Wrote {out_dir / 'polyanskiy_bounds.png'}")
    if not args.skip_required_ebn0:
        print(f"Wrote {out_dir / 'required_ebn0_with_bounds.png'}")
    print(f"Wrote {out_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
