"""Combine paired arrangement-threshold summaries into one required-Eb/N0 plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402
from src.plotting import _configure_matplotlib  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+",
                   help="result directories under results/threshold, or direct arrangement_threshold_summary.json paths")
    p.add_argument("--target", type=float, default=0.05)
    p.add_argument("--include-dense", action="store_true",
                   help="also plot the dense baseline once, using the first summary that contains it")
    p.add_argument("--out-name", default="arrangement_sweep",
                   help="subdirectory under results/threshold for the combined plot")
    return p.parse_args(argv)


def summary_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "arrangement_threshold_summary.json"
    return RESULTS_DIR / "threshold" / arg / "arrangement_threshold_summary.json"


def load_rows(path: Path, target: float) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    rows = [r for r in data["summary"] if abs(float(r["target_pupe"]) - float(target)) < 1e-12]
    return data, rows


def arrangement_label(row: dict) -> str:
    base = row.get("base", {})
    if row["arrangement"] == "Dense":
        return f"dense d={base.get('d')}, blocks=1"
    return f"d={base.get('d')}, blocks={base.get('num_blocks')}"


def plot_series(series: list[tuple[str, list[dict]]], out_path: Path, *, title: str) -> None:
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
        ax.plot(x[finite], y[finite], lw=2, ms=5, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], label=label)

    ax.set_xlabel("Active devices K")
    ax.set_ylabel("Required Eb/N0 (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    series: list[tuple[str, list[dict]]] = []
    dense_series: tuple[str, list[dict]] | None = None
    metadata = None

    for arg in args.summaries:
        path = summary_path(arg)
        data, rows = load_rows(path, args.target)
        metadata = metadata or data
        odma_rows = [r for r in rows if r["arrangement"] == "ODMA"]
        if odma_rows:
            series.append((arrangement_label(odma_rows[0]), odma_rows))
        if dense_series is None:
            dense_rows = [r for r in rows if r["arrangement"] == "Dense"]
            if dense_rows:
                dense_series = (arrangement_label(dense_rows[0]), dense_rows)

    if args.include_dense and dense_series is not None:
        series.append(dense_series)
    if not series:
        raise SystemExit("no finite arrangement rows found")

    out_dir = RESULTS_DIR / "threshold" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Arrangement sweep required Eb/N0 (PUPE<={args.target:g})"
    if metadata:
        title += f"\n(n={metadata.get('arrangements', [['', {'n': '?'}]])[0][1].get('n')}, bits={metadata.get('payload_bits')})"
    out_path = out_dir / "arrangement_sweep_required_ebn0.png"
    plot_series(series, out_path, title=title)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
