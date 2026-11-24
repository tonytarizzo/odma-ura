"""Plot the 12-frame low-load thresholds and the energy measurements that determined them."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __package__:
    from .plot_load import DECODERS, LOADS, OUT, OZATES_PDF_SHA256, reference_contract, references
else:
    from plot_load import DECODERS, LOADS, OUT, OZATES_PDF_SHA256, reference_contract, references

STYLE = {
    "DEC-38": dict(color="#006d91", marker="o", linestyle="-", markerfacecolor="#006d91"),
    "DEC-39": dict(color="#b84822", marker="D", linestyle="--", markerfacecolor="white"),
}


def figure(loads, ref, frames):
    fig = plt.figure(figsize=(10.4, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.8, 1), left=.079, right=.985, bottom=.18, top=.82, hspace=.55, wspace=.30)
    main = fig.add_subplot(grid[:, 0])
    panels = [fig.add_subplot(grid[i, 1]) for i in range(2)]
    fig.suptitle("Energy at 5% per-user error", x=.079, y=.97, ha="left", fontsize=16, weight="bold")
    fig.text(.079, .885, f"100-bit messages   ·   30,000 real channel uses   ·   Exploratory · {frames} geometries per measured load",
             fontsize=10, color="#4b5457")
    states = set()
    for name in DECODERS:
        records = [loads[k]["arms"][name] for k in (50, 100)]
        points = [np.nan if r["snr_db"] is None else r["snr_db"] for r in records]
        style = STYLE[name]
        main.plot((50, 100), points, **style, ms=5.5, lw=1.7, label=f"{name} · balanced", zorder=4)
        for k, r in zip((50,100), records, strict=True):
            status = r["status"]
            if status in ("at_grid", "interpolated"):
                lo, hi = r["paired_bootstrap_95_db"]
                if lo is not None and hi is not None:
                    main.vlines(k, lo, hi, color=style["color"], alpha=.8, lw=1.)
                    main.hlines([lo, hi], k-2, k+2, color=style["color"], alpha=.8, lw=1.)
                else:
                    states.add("open")
            elif status in ("left_censored", "right_censored"):
                left = status == "left_censored"
                edge = r["censor_bounds_db"][1 if left else 0]
                main.plot(k, edge, marker="v" if left else "^", color=style["color"], linestyle="none", ms=6.5)
                states.add("limits")
            elif status == "ambiguous":
                states.add("ambiguous")
            else:
                raise ValueError(f"Unknown crossing status {status}")
    sources = [ref["loads"][str(k)] for k in LOADS]
    main.plot(LOADS, [s["published_odma_db"] for s in sources], "-.", color="#716382", marker="s", ms=4,
              markerfacecolor="white", lw=1.5, label="Özates et al. · published GMAC")
    main.plot(LOADS, [s["achievability"]["ebn0_db"] for s in sources], ":", color="#333d3d", lw=1.7,
              label="Polyanskiy · random-coding achievability")
    main.plot(LOADS, [s["genie_list_fano_db"] for s in sources], "--", color="#747c7c", lw=1.2,
              label="Genie/list-Fano · necessary energy")
    main.set(xlim=(40, 310), xticks=LOADS, xlabel=r"Active users $K_a$", ylabel=r"Required $E_b/N_0$ (dB)")
    main.legend(frameon=False, fontsize=8, loc="upper left")
    for ax, k in zip(panels, (50, 100), strict=True):
        row = loads[k]
        energies = np.asarray(row["energies_db"])
        fine = np.unique(np.r_[energies, np.linspace(energies[0], energies[-1], 600)])
        for name in DECODERS:
            values = 100*np.asarray(row["arms"][name]["pupe"])
            style = STYLE[name]
            # Interpolate in PUPE before displaying it on the nonlinear axis, as in the threshold calculation.
            ax.plot(fine, np.interp(fine, energies, values), color=style["color"], linestyle=style["linestyle"], lw=1.1)
            ax.plot(energies, values, **{**style, "linestyle": "none"}, ms=3.5)
        ax.axhline(5, color="#555e62", linestyle=":", lw=1., zorder=0)
        ax.set_title(f"{k} users", loc="left", fontsize=10, weight="bold", pad=4)
        ax.set_yscale("symlog", linthresh=5)
        ax.set(ylim=(-.5, 105), yticks=[0,5,20,80], yticklabels=["0","5","20","80"], ylabel="PUPE (%)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    panels[-1].set_xlabel(r"Tested $E_b/N_0$ (dB)")
    for ax in (main, *panels):
        ax.grid(axis="y", alpha=.21, linewidth=.65)
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.079, .066, "Whiskers show 95% frame-bootstrap intervals conditional on the tested grid. The dotted horizontal line marks 5% PUPE.",
             fontsize=8, color="#4b5457")
    foot = {"open": "Open interval endpoints are recorded in the summary.", "limits": "Triangles mark one-sided search limits.",
            "ambiguous": "Recrossings have no threshold estimate."}
    if states:
        fig.text(.079, .035, "  ".join(foot[s] for s in sorted(states)), fontsize=8, color="#4b5457")
    return fig


def render(summary, out):
    out = Path(out)
    loads = {int(k): v for k, v in summary["loads"].items()}
    if (summary["frames_per_load"] != 12 or not summary["exploratory"] or summary["target"] != .05 or set(loads) != {50, 100}
            or any(r["frames"] != 12 or r["target"] != .05 for r in loads.values())):
        raise ValueError("Expected the 12-geometry exploratory 5% panel at 50 and 100 active users")
    out.mkdir(parents=True, exist_ok=True)
    path = out.parent/"load_references.json"
    ref = json.loads(path.read_bytes()) if path.exists() else references(out.parent)
    if ref["contract"] != reference_contract() or ref["published_pdf_sha256"] != OZATES_PDF_SHA256:
        raise ValueError("Incompatible reference figure cache")
    fig = figure(loads, ref, frames=12)
    stem = out/"required_ebn0_vs_load"
    try:
        for suffix in (".pdf", ".png"):
            fig.savefig(stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    folder = OUT/"preview12"
    print(render(json.loads((folder/"summary.json").read_bytes()), folder))
