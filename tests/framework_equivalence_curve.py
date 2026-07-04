"""Matched legacy-vs-framework equivalence curve.

For each requested setup, this script builds a legacy `src.scenario` trial and
then rebuilds the same dictionary through the framework `(R, C, U, T)` factors.
It asserts exact dictionary equivalence before running a shared oracle-K NNOMP
decoder on both dictionaries and the same observation.
"""

from __future__ import annotations

import argparse
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
from src.scenario import Scenario, build_scenario  # noqa: E402
from src.signal import ebn0_db_to_esn0_db  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--presets", nargs="+", choices=["dense", "odma"], default=["dense", "odma"])
    p.add_argument("-B", "--payload-bits", type=int, default=10)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--d", type=int, default=64, help="ODMA local codeword length; dense uses d=n")
    p.add_argument("--num-blocks", type=int, default=4, help="ODMA blocks; dense uses one block")
    p.add_argument("--num-antennas", type=int, default=2)
    p.add_argument("--K-values", nargs="+", type=int, default=[2, 5, 10, 15, 20, 30, 40])
    p.add_argument("--ebn0-grid", nargs="+", type=float, default=[0.0, 4.0])
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--phi-atol", type=float, default=1e-12)
    p.add_argument("--out-dir", default="results/framework_equivalence_curve")
    return p.parse_args(argv)


def scenario_for(args: argparse.Namespace, preset: str, K: int, ebn0_db: float, seed: int) -> Scenario:
    M = 1 << int(args.payload_bits)
    d = int(args.n) if preset == "dense" else int(args.d)
    num_blocks = 1 if preset == "dense" else int(args.num_blocks)
    esn0_db = ebn0_db_to_esn0_db(float(ebn0_db), d, M)
    return build_scenario(n=int(args.n), d=d, num_blocks=num_blocks, num_codewords=M,
                          num_devices_active=int(K), num_antennas=int(args.num_antennas),
                          esn0_db=float(esn0_db), seed=int(seed))


def legacy_phi(scenario: Scenario) -> np.ndarray:
    Phi = np.zeros((scenario.n, scenario.num_codewords), dtype=np.float64)
    for m in range(scenario.num_codewords):
        b = scenario.msg_to_block[m]
        Phi[:, m] = scenario.P_mats[b] @ scenario.codebook[m]
    return Phi


def framework_phi(scenario: Scenario) -> np.ndarray:
    R = torch.stack([torch.as_tensor(scenario.P_mats[b], dtype=torch.float64) for b in range(scenario.num_blocks)])
    C = torch.as_tensor(scenario.codebook.T, dtype=torch.float64)
    msg = torch.arange(scenario.num_codewords)
    component = ComponentSpec(Q=scenario.num_blocks, d=scenario.d, V=scenario.num_codewords, N=scenario.num_codewords,
                              R_init="explicit", C_init="explicit", U_init="explicit", T_init="identity",
                              explicit_R=R, explicit_C=C,
                              explicit_atom_q=msg % scenario.num_blocks, explicit_atom_v=msg)
    spec = URASpec(n=scenario.n, num_codewords=scenario.num_codewords,
                   num_active=scenario.num_devices_active, num_antennas=scenario.num_antennas,
                   payload_bits=int(round(math.log2(scenario.num_codewords))))
    encoder = build_encoder(spec, [component], dtype=torch.float64)
    return encoder.explicit_matrix().detach().cpu().numpy()


def matched_filter_y(Y: np.ndarray) -> np.ndarray:
    h = np.ones(Y.shape[1], dtype=Y.dtype)
    return Y @ h.conj() / float(np.real(np.vdot(h, h)))


def project_nonneg_integer_total(x: np.ndarray, total: int) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    if total == 0:
        return np.zeros_like(x)
    if x.size == 0:
        raise ValueError("cannot assign a positive total count to an empty support")
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - float(total)
    idx = np.arange(1, x.size + 1, dtype=np.float64)
    active = u - cssv / idx > 0
    theta = cssv[np.nonzero(active)[0][-1]] / float(np.sum(active)) if np.any(active) else 0.0
    z = np.maximum(x - theta, 0.0)
    counts = np.floor(z)
    rem = int(total - np.sum(counts))
    if rem > 0:
        counts[np.argsort(-(z - counts), kind="mergesort")[:rem]] += 1.0
    elif rem < 0:
        for i in np.argsort(z - counts, kind="mergesort"):
            if rem == 0:
                break
            take = min(int(counts[i]), -rem)
            counts[i] -= take
            rem += take
    if int(np.sum(counts)) != int(total):
        raise RuntimeError("integer total projection failed")
    return counts


def oracle_k_nnomp(Phi: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    residual = y.copy()
    support: list[int] = []
    used = np.zeros(Phi.shape[1], dtype=bool)
    x_nn = np.zeros(0, dtype=np.float64)
    for _ in range(min(int(K), Phi.shape[0], Phi.shape[1])):
        corrs = np.real(Phi.conj().T @ residual)
        corrs[used] = -np.inf
        best = int(np.argmax(corrs))
        if not np.isfinite(corrs[best]):
            break
        support.append(best); used[best] = True
        A = Phi[:, support]
        x_nn, _ = nnls(A, y)
        residual = y - A @ x_nn
    counts = np.zeros(Phi.shape[1], dtype=np.float64)
    if support:
        counts[np.asarray(support, dtype=int)] = project_nonneg_integer_total(x_nn, int(K))
    return counts


def run_trial(args: argparse.Namespace, preset: str, K: int, ebn0_db: float, seed: int) -> tuple[list[dict], float]:
    scenario = scenario_for(args, preset, K, ebn0_db, seed)
    Phi_legacy = legacy_phi(scenario)
    Phi_framework = framework_phi(scenario)
    phi_err = float(np.max(np.abs(Phi_legacy - Phi_framework)))
    if phi_err > float(args.phi_atol):
        raise AssertionError(f"{preset} Phi mismatch at K={K}, Eb/N0={ebn0_db}, seed={seed}: max_abs={phi_err:.3e}")

    y = matched_filter_y(scenario.Y)
    rows = []
    decoded = []
    for name, Phi in (("legacy", Phi_legacy), ("framework", Phi_framework)):
        t0 = time.time()
        counts = oracle_k_nnomp(Phi, y, int(K))
        decoded.append(counts)
        wall = time.time() - t0
        metrics = evaluate_counts(scenario.message_counts, counts, max_list_size=int(K))
        rows.append({"preset": preset, "construction": name, "K": int(K), "ebn0_db": float(ebn0_db),
                     "seed": int(seed), "metrics": metrics, "wall_s": wall, "phi_max_abs_err": phi_err})

    if not np.array_equal(decoded[0], decoded[1]):
        raise AssertionError(f"{preset} decoded counts differ despite matching Phi at K={K}, Eb/N0={ebn0_db}, seed={seed}")
    return rows, phi_err


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["preset"], r["construction"], r["K"], r["ebn0_db"]) for r in rows})
    for preset, construction, K, ebn0_db in keys:
        sel = [r for r in rows if (r["preset"], r["construction"], r["K"], r["ebn0_db"]) == (preset, construction, K, ebn0_db)]
        metric_keys = sel[0]["metrics"].keys()
        out.append({"preset": preset, "construction": construction, "K": int(K), "ebn0_db": float(ebn0_db),
                    "num_trials": len(sel), "mean_wall_s": float(np.mean([r["wall_s"] for r in sel])),
                    "max_phi_abs_err": float(max(r["phi_max_abs_err"] for r in sel)),
                    **{f"mean_{k}": float(np.mean([r["metrics"][k] for r in sel])) for k in metric_keys
                       if isinstance(sel[0]["metrics"][k], (int, float, np.integer, np.floating))}})
    return out


def plot_summary(points: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presets = sorted({p["preset"] for p in points})
    fig, axes = plt.subplots(len(presets), 2, figsize=(11, 4.2 * len(presets)), squeeze=False)
    for row, preset in zip(axes, presets):
        p_points = [p for p in points if p["preset"] == preset]
        for ebn0_db in sorted({p["ebn0_db"] for p in p_points}):
            for construction, style in (("legacy", "-"), ("framework", "--")):
                curve = [p for p in p_points if p["ebn0_db"] == ebn0_db and p["construction"] == construction]
                curve = sorted(curve, key=lambda p: p["K"])
                label = f"{construction}, {ebn0_db:g} dB"
                row[0].plot([p["K"] for p in curve], [p["mean_l1_acc"] for p in curve], style, marker="o", label=label)
                row[1].plot([p["K"] for p in curve], [p["mean_pupe"] for p in curve], style, marker="o", label=label)
        row[0].set_title(f"{preset}: L1 accuracy")
        row[1].set_title(f"{preset}: PUPE")
        for ax in row:
            ax.set_xlabel("Active devices K")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        row[0].set_ylim(0.0, 1.02)
        row[1].set_ylim(0.0, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.num_antennas) < 2:
        raise SystemExit("--num-antennas must be >= 2 to match the legacy common-signature model")
    if int(args.d) <= 0 or int(args.d) > int(args.n):
        raise SystemExit(f"invalid ODMA geometry: d={args.d}, n={args.n}")

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Equivalence curve: presets={args.presets}, n={args.n}, B={args.payload_bits}, M={1 << int(args.payload_bits)}")
    print(f"K values={args.K_values}, Eb/N0={args.ebn0_grid}, seeds={seeds}")

    rows: list[dict] = []
    max_phi_err = 0.0
    total = len(args.presets) * len(args.K_values) * len(args.ebn0_grid) * len(seeds)
    done = 0
    for preset in args.presets:
        for K in args.K_values:
            for ebn0_db in args.ebn0_grid:
                for seed in seeds:
                    trial_rows, phi_err = run_trial(args, preset, int(K), float(ebn0_db), int(seed))
                    rows.extend(trial_rows)
                    max_phi_err = max(max_phi_err, phi_err)
                    done += 1
                latest = [r for r in rows if r["preset"] == preset and r["K"] == int(K) and r["ebn0_db"] == float(ebn0_db)]
                legacy_l1 = np.mean([r["metrics"]["l1_acc"] for r in latest if r["construction"] == "legacy"])
                framework_l1 = np.mean([r["metrics"]["l1_acc"] for r in latest if r["construction"] == "framework"])
                print(f"[{done:4d}/{total}] {preset:<5s} K={int(K):<4d} Eb/N0={float(ebn0_db):>5.2f} "
                      f"L1 legacy={legacy_l1:.4f} framework={framework_l1:.4f} phi_err<={max_phi_err:.1e}", flush=True)

    points = summarize(rows)
    payload = {"args": vars(args), "rows": rows, "points": points, "max_phi_abs_err": max_phi_err}
    (out_dir / "equivalence_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_summary(points, out_dir / "equivalence_curves.png")
    print(f"Max Phi abs error: {max_phi_err:.3e}")
    print(f"Wrote {out_dir / 'equivalence_summary.json'}")
    print(f"Wrote {out_dir / 'equivalence_curves.png'}")


if __name__ == "__main__":
    main()
