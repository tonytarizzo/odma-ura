"""Plotting helpers — sweep curves and convergence trajectories.

All figures use the project palette / linestyle from
src.decoders.registry, so adding a decoder there propagates everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .decoders.registry import LINESTYLE, MARKER, PALETTE


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _configure_matplotlib() -> None:
    """Keep Matplotlib/font caches inside the repo for sandboxed runs."""
    cache_dir = PROJECT_ROOT / ".cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "xdg" / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


METRIC_LABELS = {
    "f1":     "F1 score",
    "l1_acc": "L1 accuracy",
    "l1_err": "L1 error",
    "nmse":   "NMSE",
    "pupe":   "PUPE",
}


def _select_rows(rows: list[dict], scenario_filter: dict, decoder: str,
                  swept_param: str, value: float) -> list[dict]:
    out = []
    for r in rows:
        if r["decoder"] != decoder:
            continue
        sc = r["scenario"]
        if sc.get(swept_param) != value:
            continue
        ok = True
        for k, v in scenario_filter.items():
            if k == swept_param:
                continue
            if sc.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def plot_sweep_lines(rows: list[dict], swept_param: str, values: list,
                     decoders: list[str], scenario_filter: dict,
                     out_dir: Path, *, sweep_label: str = "",
                     metrics: list[str] = ("f1", "l1_acc")) -> None:
    """One figure per metric, plus one combined panel.

    `scenario_filter` is the base scenario dict; the swept_param value
    is replaced with each entry of `values`.
    """
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    xv = np.array(values, dtype=float)

    fixed_parts = []
    display = {"n": "n", "d": "d", "num_blocks": "B", "num_codewords": "M",
               "num_devices_active": "K", "num_antennas": "M_ant", "esn0_db": "SNR"}
    for k, v in scenario_filter.items():
        if k == swept_param:
            continue
        if k in display:
            fixed_parts.append(f"{display[k]}={v}")
    subtitle = ", ".join(fixed_parts)

    def gather(decoder: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        means = np.full(len(values), np.nan)
        stds  = np.full(len(values), np.nan)
        for i, val in enumerate(values):
            sel = _select_rows(rows, scenario_filter, decoder, swept_param, val)
            if not sel:
                continue
            vs = [r["metrics"].get(metric) for r in sel
                  if r["metrics"].get(metric) is not None]
            vs = [v for v in vs if isinstance(v, (int, float)) and np.isfinite(v)]
            if not vs:
                continue
            means[i] = float(np.mean(vs))
            stds[i] = float(np.std(vs)) if len(vs) > 1 else 0.0
        return means, stds

    def ordered_series(metric: str) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        series = []
        for dec in decoders:
            mean, std = gather(dec, metric)
            score = float(np.nanmean(mean)) if np.any(np.isfinite(mean)) else -np.inf
            series.append((dec, mean, std, score))
        return sorted(series, key=lambda item: item[3], reverse=True)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        any_data = False
        for dec, mean, std, _ in ordered_series(metric):
            if np.all(np.isnan(mean)):
                continue
            any_data = True
            color = PALETTE.get(dec, "#333333")
            ls = LINESTYLE.get(dec, "-")
            marker = MARKER.get(dec, "o")
            ax.plot(xv, mean, marker=marker, lw=2, ms=5, label=dec, color=color, ls=ls)
            if np.any(std > 0):
                ax.fill_between(xv, mean - std, mean + std, alpha=0.12, color=color)
        ax.set_xlabel(sweep_label or swept_param, fontsize=11)
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (higher = better)" if metric in ("f1", "l1_acc")
                      else METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_ylim(0.0, 1.02) if metric in ("f1", "l1_acc") else None
        ax.grid(True, alpha=0.3)
        if any_data:
            ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs {sweep_label or swept_param}\n({subtitle})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{swept_param}_{metric}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        any_data = False
        for dec, mean, std, _ in ordered_series(metric):
            if np.all(np.isnan(mean)):
                continue
            any_data = True
            color = PALETTE.get(dec, "#333333")
            ls = LINESTYLE.get(dec, "-")
            marker = MARKER.get(dec, "o")
            ax.plot(xv, mean, marker=marker, lw=2, ms=4, label=dec, color=color, ls=ls)
            if np.any(std > 0):
                ax.fill_between(xv, mean - std, mean + std, alpha=0.12, color=color)
        ax.set_xlabel(sweep_label or swept_param, fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
        if metric in ("f1", "l1_acc"):
            ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        if any_data:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Sweep: {sweep_label or swept_param}  ({subtitle})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{swept_param}_combined.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history: list[dict], out_path: Path,
                     decoder_name: str, *, true_values: dict | None = None) -> None:
    """Per-iteration trajectory plot for an iterative decoder.

    Auto-detects which keys are present in `history` items and plots up to
    a 2x2 grid of available numeric series. Common keys: delta, lambda /
    lam, noise_var / noise_var_est, k_est / K_hat, r_pri, r_dual, objective,
    rho.
    """
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iters = [h.get("iter", i + 1) for i, h in enumerate(history)]

    candidate_groups = [
        # Graph-BP style
        [("delta",     "Max site mean delta",  True),
         ("lambda",    "Poisson rate lambda",  False),
         ("noise_var", "Noise variance sigma^2", True),
         ("k_est",     "Total mean count",     False)],
        # ADMM style
        [("r_pri",     "Primal residual",      True),
         ("r_dual",    "Dual residual",        True),
         ("objective", "Objective",            True),
         ("lam",       "Poisson rate lambda",  False)],
    ]

    chosen = None
    for grp in candidate_groups:
        if any(k in history[0] for k, _, _ in grp):
            chosen = [(k, lab, log) for k, lab, log in grp if k in history[0]]
            break

    if not chosen:
        return

    n_panels = len(chosen)
    cols = 2 if n_panels > 1 else 1
    rows_n = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 3.5 * rows_n), squeeze=False)
    color = PALETTE.get(decoder_name, "#333333")

    for ax, (key, lab, log) in zip(axes.flat, chosen):
        ys = [float(h[key]) for h in history if key in h]
        xs = [iters[i] for i, h in enumerate(history) if key in h]
        if log:
            ax.semilogy(xs, ys, marker="o", ms=3, lw=1.5, color=color)
        else:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color=color)
        ax.set_xlabel("Iteration"); ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
        if true_values and key in true_values:
            ax.axhline(true_values[key], color="grey", lw=1.0, ls="--",
                       label=f"true={true_values[key]:.3g}")
            ax.legend(fontsize=8)

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"{decoder_name} convergence", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_bars(rows_for_one_scenario: list[dict],
                      out_path: Path, *, metrics: list[str] = ("f1", "l1_acc")) -> None:
    """Single-scenario decoder comparison: grouped bar chart."""
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows_for_one_scenario:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoders = [r["decoder"] for r in rows_for_one_scenario]
    x = np.arange(len(decoders))
    n_m = len(metrics)
    w = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(max(8, len(decoders) * 1.0 + 2), 4.5))
    for i, m in enumerate(metrics):
        vals = [r["metrics"].get(m, 0.0) for r in rows_for_one_scenario]
        ax.bar(x + (i - (n_m - 1) / 2) * w, vals, w,
               label=METRIC_LABELS.get(m, m))
    ax.set_xticks(x); ax.set_xticklabels(decoders, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    ax.set_title("Decoder comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_count_estimates(true_counts, decoder_counts: dict[str, np.ndarray],
                         out_path: Path) -> None:
    """Bar chart of estimated counts per active message vs truth."""
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_counts = np.asarray(true_counts)
    active_idx = np.nonzero(true_counts)[0]
    if active_idx.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoders = list(decoder_counts.keys())
    n_active = len(active_idx)
    n_dec = len(decoders)
    w = 0.75 / (n_dec + 1)
    offsets = np.linspace(-(n_dec / 2.0) * w, (n_dec / 2.0) * w, n_dec + 1)

    fig, ax = plt.subplots(figsize=(max(9, n_active * 1.2 + 2), 4.5))
    ax.bar(np.arange(n_active) + offsets[0], true_counts[active_idx], w,
           label="True", color="#888888", alpha=0.8)
    for i, dec in enumerate(decoders):
        vals = np.asarray(decoder_counts[dec])[active_idx]
        ax.bar(np.arange(n_active) + offsets[i + 1], vals, w,
               label=dec, color=PALETTE.get(dec, "#333"), alpha=0.85)
    ax.set_xticks(np.arange(n_active))
    ax.set_xticklabels([str(i) for i in active_idx], fontsize=8)
    ax.set_xlabel("Message index"); ax.set_ylabel("Count")
    ax.set_title("Estimated counts — active messages only")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_required_ebn0(summary_rows: list[dict], out_path: Path, *, title: str = "Required Eb/N0") -> None:
    """Plot threshold curves produced by tests.threshold_test."""
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summary_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, float], list[dict]] = {}
    for row in summary_rows:
        groups.setdefault((row["decoder"], float(row["target_pupe"])), []).append(row)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False
    for (decoder, target), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        rows = sorted(rows, key=lambda row: row["num_devices_active"])
        x = np.array([row["num_devices_active"] for row in rows], dtype=float)
        y = np.array([row["required_ebn0_db"] for row in rows], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        any_data = True
        color = PALETTE.get(decoder, "#333333")
        marker = MARKER.get(decoder, "o")
        ls = LINESTYLE.get(decoder, "-")
        ax.plot(x[finite], y[finite], marker=marker, lw=2, ms=5, color=color, ls=ls,
                label=f"{decoder}, PUPE<={target:g}")

    ax.set_xlabel("Active devices K")
    ax.set_ylabel("Required Eb/N0 (dB)")
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
