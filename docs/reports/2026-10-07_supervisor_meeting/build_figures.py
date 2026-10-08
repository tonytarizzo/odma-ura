"""Rebuild the curated meeting figures from a frozen, auditable data snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = {
    "matched": "jobs/033_matched_d2/results/analysis/summary.json",
    "decoders": "jobs/032_published_baselines/results/comparison/analysis/summary.json",
    "bounds": "jobs/032_published_baselines/results/comparison/analysis/both_bounds_distinct/bound_overlay.json",
}
COLORS = {"dense": "#1867a5", "sparse": "#df8b24", "odma_polar": "#8155a5", "hash_full": "#484c55",
          "hash_coordinate_J10": "#a269af", "hash_coordinate_J8": "#128873", "hash_coordinate_J4": "#c84b51"}


def refresh():
    inputs = {k: json.loads((ROOT / p).read_text()) for k, p in SOURCES.items()}
    selected = lambda rows: [r for r in rows if r["sampling"] == "distinct" and r["K"] == 26]
    data = {"provenance": {k: {"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                           for k, p in SOURCES.items()}, "audit_date": "2026-10-06"}
    data["matched"] = selected(inputs["matched"]["points"])
    data["decoders"] = [r for r in selected(inputs["decoders"]["points"])
                        if r["family"] == "dense" and r["mode"] == "fixed" and r["decoder"] != "d4"]
    data["bounds"] = {k: [r for r in inputs["bounds"][k] if r["K"] == 26]
                      for k in ("curves", "converse_curves")}
    data["bound_metadata"] = {k: v for k, v in inputs["bounds"].items() if k not in ("curves", "converse_curves", "audit")}
    (HERE / "plot_data.json").write_text(json.dumps(data, indent=2) + "\n")


def points(data, b, family, mode="fixed", decoder="d2", group="matched"):
    rows = sorted([r for r in data[group] if r["B"] == b and r["family"] == family
                   and r["mode"] == mode and r["decoder"] == decoder], key=lambda r: r["ebn0_db"])
    assert len(rows) == 8 and all(len(r["seeds"]) == 3 for r in rows), (b, family, mode, decoder)
    return np.array([r["ebn0_db"] for r in rows]), np.array([r["pupe"] for r in rows])


def axes_pair(height=3.6, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(7, height), sharey=True)
    for ax in axes:
        ax.set(xlabel=r"$E_b/N_0$ (dB)", xticks=[-4, 0, 4, 8, 10], xlim=(-4.3, 10.3))
        ax.grid(alpha=.17, which="major")
        if log:
            ax.set(yscale="log", ylim=(1e-4, 1.05))
        else:
            ax.set(ylim=(-.02, 1.02))
    axes[0].set_ylabel("PUPE (lower is better)")
    return fig, axes


def finish(fig, axes, name, bottom=.24, ncol=3):
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=ncol, frameon=False, columnspacing=1.3, handlelength=2.2)
    fig.subplots_adjust(left=.09, right=.99, top=.91, bottom=bottom, wspace=.13)
    fig.savefig(HERE / "figures" / f"{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true", help="Refresh frozen data from local audited results")
    if parser.parse_args().refresh_data:
        refresh()
    data = json.loads((HERE / "plot_data.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "lines.linewidth": 1.6, "lines.markersize": 3.5})
    fig, axes = axes_pair(height=4.25, log=True)
    styles = [("dense", "fixed", "d2", "Dense + D2", COLORS["dense"], "-"),
              ("sparse", "fixed", "d2", "Sparse (32) + D2", COLORS["sparse"], "-"),
              ("odma_polar", "native", "native", "ODMA–polar: native", COLORS["odma_polar"], "--"),
              ("odma_polar", "fixed", "d2", "ODMA–polar + D2", COLORS["odma_polar"], "-"),
              ("ccs_amp", "native", "native", "CCS–AMP: native*", "#88562c", "--"),
              ("ccs_block", "native", "native", "Block CCS: native", "#717775", ":")]
    for ax, b in zip(axes, (12, 14)):
        for family, mode, decoder, label, color, ls in styles:
            ax.plot(*points(data, b, family, mode, decoder), color=color, ls=ls, marker="o", label=label)
        for key, label, color, ls in [("curves", "Achievability (upper)", "#16191e", "--"),
                                      ("converse_curves", "Converse (lower)", "#d04478", "-.")]:
            row = next(r for r in data["bounds"][key] if r["B"] == b)
            xy = np.array(row["snr_and_bound"])
            ax.plot(xy[:, 0], xy[:, 1], color=color, ls=ls, lw=1.5, label=label)
        ax.set_title(f"B = {b}, n = 256, K = 26")
    finish(fig, axes, "matched_systems_bounds", bottom=.29, ncol=2)

    fig, axes = axes_pair(height=3.45)
    labels = {"dense": "Dense", "sparse": "Sparse (32)", "hash_full": "Hash: unrestricted",
              "hash_coordinate_J10": "Hash: J=10", "hash_coordinate_J8": "Hash: J=8", "hash_coordinate_J4": "Hash: J=4"}
    for ax, mode in zip(axes, ("fixed", "joint")):
        for family, label in labels.items():
            ax.plot(*points(data, 14, family, mode), marker="o", label=label, color=COLORS[family])
        ax.set_title(f"{'Fixed' if mode == 'fixed' else 'Jointly learned'} encoder + D2")
    finish(fig, axes, "amplitude_frontier", bottom=.25)

    fig, axes = axes_pair(height=3.35, log=True)
    for ax, b in zip(axes, (12, 14)):
        for decoder, color in zip(("d0", "d1", "d2", "d3"), ("#8e979d", "#dc8a28", "#1867a5", "#128873")):
            ax.plot(*points(data, b, "dense", decoder=decoder, group="decoders"), marker="o",
                    color=color, label=decoder.upper())
        ax.set_title(f"B = {b}, fixed dense, K = 26")
    finish(fig, axes, "decoder_ladder", bottom=.22, ncol=4)

if __name__ == "__main__":
    main()
