"""Required-Eb/N0-vs-K plotting and seed-bootstrap CIs for arrangement sweeps.

Consumed by ``tests.arrangement_sweep_threshold_test``; not a standalone CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import _configure_matplotlib  # noqa: E402


def bootstrap_required_ci(trials: list[dict], arrangement: str, K: int, target: float, *,
                          num_boot: int = 500, ci: float = 0.95, seed: int = 0) -> tuple[float, float] | None:
    """Percentile CI of required Eb/N0 for one (arrangement, K), by resampling seeds.

    Uses the per-seed PUPE values already stored at each evaluated Eb/N0. Each
    bootstrap resamples the seed set with replacement, recomputes the mean-PUPE
    curve, and takes the smallest evaluated Eb/N0 whose resampled mean is <=
    target. Returns None when the point never reaches the target (nothing to
    bracket) so the caller can simply omit the error bar.
    """
    by_ebn0: dict[float, dict[int, float]] = {}
    for r in trials:
        if r.get("arrangement") != arrangement or int(r["scenario"]["num_devices_active"]) != int(K):
            continue
        by_ebn0.setdefault(float(r["ebn0_db"]), {})[int(r["seed"])] = float(r["metrics"]["pupe"])
    if not by_ebn0:
        return None
    ebn0_grid = np.array(sorted(by_ebn0))
    seeds = sorted({s for d in by_ebn0.values() for s in d})
    pupe = np.array([[by_ebn0[e].get(s, np.nan) for s in seeds] for e in ebn0_grid])  # (n_ebn0, n_seed)

    rng = np.random.default_rng(seed)
    required = []
    for _ in range(num_boot):
        pick = rng.integers(0, len(seeds), size=len(seeds))
        mean_pupe = np.nanmean(pupe[:, pick], axis=1)
        ok = np.flatnonzero(mean_pupe <= target)
        if ok.size:
            required.append(float(ebn0_grid[ok[0]]))
    if not required:
        return None
    lo = float(np.percentile(required, 100.0 * (1.0 - ci) / 2.0))
    hi = float(np.percentile(required, 100.0 * (1.0 + ci) / 2.0))
    return lo, hi


BOUND_STYLES = {
    "canonical": {"color": "#6B7280", "ls": (0, (5, 2)), "marker": "*"},
    "count":     {"color": "#111827", "ls": (0, (1, 1)), "marker": "P"},
    "strict":    {"color": "#B91C1C", "ls": (0, (3, 1, 1, 1)), "marker": "X"},
}
BOUND_LABELS = {
    "canonical": "Polyanskiy RCU (canonical, collisions ignored)",
    "count":     "Polyanskiy RCU (count/multiset metric)",
    "strict":    "Polyanskiy RCU (strict, collision = error)",
}


def plot_series(series: list[tuple[str, list[dict]]], out_path: Path, *, title: str,
                yerr: dict[str, dict[int, tuple[float, float]]] | None = None,
                bounds: dict[str, list[tuple[int, float]]] | None = None) -> None:
    """Required-Eb/N0-vs-K plot.

    ``yerr`` optionally maps arrangement label -> {K: (lo, hi)} to draw CI bars.
    ``bounds`` optionally maps a Polyanskiy variant name ("canonical"/"count"/
    "strict") to (K, Eb/N0) points already on the experiment axis.
    """
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#0E7490", "#2F6B00", "#7C2D12", "#7C3AED", "#4B5563", "#C2410C"]
    markers = ["o", "s", "D", "^", "v", "P"]

    for i, (label, rows) in enumerate(series):
        rows = sorted(rows, key=lambda r: r["num_devices_active"])
        x = np.array([r["num_devices_active"] for r in rows], dtype=float)
        y = np.array([r["required_ebn0_db"] for r in rows], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        color = colors[i % len(colors)]
        err = None
        if yerr and label in yerr:
            lo = np.array([yerr[label].get(int(k), (np.nan, np.nan))[0] for k in x])
            hi = np.array([yerr[label].get(int(k), (np.nan, np.nan))[1] for k in x])
            err = np.vstack([y - lo, hi - y])
            err = np.where(np.isfinite(err), err, 0.0)
        if err is not None:
            ax.errorbar(x[finite], y[finite], yerr=err[:, finite], lw=2, ms=5,
                        marker=markers[i % len(markers)], color=color, label=label,
                        capsize=3, elinewidth=1)
        else:
            ax.plot(x[finite], y[finite], lw=2, ms=5, marker=markers[i % len(markers)],
                    color=color, label=label)

    for name, pts in (bounds or {}).items():
        if not pts:
            continue
        bx = np.array([k for k, _ in pts], dtype=float)
        by = np.array([v for _, v in pts], dtype=float)
        finite = np.isfinite(by)
        if not np.any(finite):
            continue
        st = BOUND_STYLES.get(name, {"color": "#111827", "ls": ":", "marker": "*"})
        ax.plot(bx[finite], by[finite], lw=1.8, ls=st["ls"], color=st["color"],
                marker=st["marker"], ms=6, label=BOUND_LABELS.get(name, name))

    ax.set_xlabel("Active devices K")
    ax.set_ylabel("Required Eb/N0 (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
