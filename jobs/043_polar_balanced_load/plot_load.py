"""Required Eb/N0 versus active users on the balanced B100 real GMAC."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.ura_bounds import list_fano_converse, required_ebn0

LOADS = (50, 100, 150, 200, 250, 300)
DECODERS = ("DEC-38", "DEC-39")
OUT = ROOT / "results/dec43_balanced_load"
POLYANSKIY = "https://people.lids.mit.edu/yp/homepage/data/isit17_mac.pdf"
OZATES = "https://spiral.imperial.ac.uk/server/api/core/bitstreams/284d4dc9-4e39-4086-a7eb-c50813dfe01a/content"
OZATES_PDF_SHA256 = "f62b36a2d99791e993a43217de1e2d3bc3b3b2da47583b112eac831ac25b014f"
# Ordinates on the first magenta stroke in the authors' Fig. 3, page 4, form /Im3.
OZATES_Y = (473.754, 537.504, 783.996, 1119.75, 1515., 1918.75)
OZATES_DASHED_DB = (.33420, .45320, .51150, .84581, 1.30331, 1.80409)


def reference_contract():
    names = ("benchmarks/ura_bounds.py", "src/ura_bound.py")
    return dict(B=100, n=30000, target=.05, sampling="iid", loads=list(LOADS), bracket_db=[-1., 5.],
                grid=31, power_grid=31, order=96,
                bound_source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names})


def bound_point(k):
    contract = reference_contract()
    result = required_ebn0(100, 30000, k, .05, distinct=False, bracket=tuple(contract["bracket_db"]),
                           grid=contract["grid"], power_grid=contract["power_grid"], order=contract["order"])
    if result["status"] != "crossing":
        raise ValueError(f"The 5% achievability crossing for K={k} is {result['status']}")
    return result


def references(out):
    contract = reference_contract()
    path = out / "load_references.json"
    cached = json.loads(path.read_bytes()) if path.is_file() else None
    if cached and cached["contract"] != contract:
        raise ValueError(f"Reference contract changed at {path}")
    rows = {}
    for k, y, paper_bound in zip(LOADS, OZATES_Y, OZATES_DASHED_DB, strict=True):
        achievable = cached["loads"][str(k)]["achievability"] if cached else bound_point(k)
        floor = brentq(lambda db: list_fano_converse(100, 30000, k, db, distinct=False) - .05, -6., 2.)
        rows[str(k)] = dict(achievability=achievable, genie_list_fano_db=floor,
                            published_odma_db=round((y - 367.5) / 425., 2), published_random_coding_trace_db=paper_bound)
    result = dict(contract=contract, loads=rows, achievability_source=POLYANSKIY, published_source=OZATES,
                  published_pdf_sha256=OZATES_PDF_SHA256, fig3_axis=dict(zero_y=367.5, units_per_db=425., series_y=OZATES_Y),
                  note="Theorem 1, all p_t and q_1, recomputed at 5%. The original article's illustrative Fig. 1 uses 10%. "
                       "The separate digitized Fig. 3 dashed trace cites the same theorem; its numerical recipe is unspecified. "
                       "The genie/list-Fano necessary energy uses the local norm-constrained single-user list converse.")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def render(summary, out, *, preview=False):
    """Draw the holdout figure, the eight-geometry pilot preview, or the theoretical and published references."""
    if preview and summary is None:
        raise ValueError("A preview requires measured pilot results")
    if summary is not None:
        by_k = {int(k): v for k, v in summary["loads"].items()}
        loads = sorted(by_k)
        expected = {50, 100} if preview else set(LOADS)
        if set(loads) != expected:
            raise ValueError("Expected two measured pilot loads or the complete six-load holdout")
        if any(by_k[k]["target"] != .05 or (preview and by_k[k]["frames"] != 8) for k in loads):
            raise ValueError("Expected the declared 5% target and eight pilot geometries in preview")
    out = Path(out)
    ref = references(out)
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    colors = {"DEC-38": "#006d91", "DEC-39": "#b84822"}
    markers = {"DEC-38": "o", "DEC-39": "D"}
    notes = []
    if summary is not None:
        notes.append("Whiskers show bounded 95% frame-bootstrap intervals.")
        limits, open_intervals, ambiguous = False, False, False
        for d in DECODERS:
            estimates = [by_k[k]["arms"][d] for k in loads]
            points = [np.nan if arm["snr_db"] is None else arm["snr_db"] for arm in estimates]
            ax.plot(loads, points, color=colors[d], marker=markers[d], ms=5.7, lw=1.8,
                    linestyle="-" if d == "DEC-38" else "--", markerfacecolor=colors[d] if d == "DEC-38" else "white",
                    label=d + " · balanced", zorder=4)
            for k, arm in zip(loads, estimates, strict=True):
                status, point = arm["status"], arm["snr_db"]
                if status in ("left_censored", "right_censored"):
                    left = status == "left_censored"
                    edge = arm["censor_bounds_db"][1 if left else 0]
                    if point is not None or edge is None:
                        raise ValueError("Incorrectly censored threshold")
                    ax.plot(k, edge, marker="v" if left else "^", color=colors[d], ms=6, linestyle="none")
                    limits = True
                elif status == "ambiguous" and point is None:
                    ambiguous = True
                elif status not in ("at_grid", "interpolated") or point is None:
                    raise ValueError(f"Unrecognized crossing at K={k} for {d}")
                else:
                    low, high = arm["paired_bootstrap_95_db"]
                    if low is not None and high is not None:
                        ax.vlines(k, low, high, color=colors[d], lw=1., alpha=.75, zorder=2)
                        ax.hlines([low, high], k - 1.75, k + 1.75, color=colors[d], lw=1., alpha=.75, zorder=2)
                    else:
                        open_intervals = True
        if limits:
            notes.append("Triangles mark one-sided search limits.")
        if ambiguous:
            notes.append("Recrossings have no point estimate.")
        if open_intervals:
            notes.append("Open interval endpoints appear in the numeric summary.")
    rows = [ref["loads"][str(k)] for k in LOADS]
    ax.plot(LOADS, [r["published_odma_db"] for r in rows], color="#716382", lw=1.55, linestyle="-.", marker="s",
            ms=4.6, markerfacecolor="white", label="Özates et al. · published GMAC", zorder=1)
    ax.plot(LOADS, [r["achievability"]["ebn0_db"] for r in rows], color="#333d3d", lw=1.7, linestyle=":",
            label="Polyanskiy · random-coding achievability")
    ax.plot(LOADS, [r["genie_list_fano_db"] for r in rows], color="#747c7c", lw=1.2, linestyle=(0, (5, 4)),
            label="Genie/list-Fano · necessary energy")
    ax.set(xlabel=r"Active users $K_a$", ylabel=r"Required $E_b/N_0$ (dB)", xticks=LOADS, xlim=(40, 310))
    ax.set_title("Energy at 5% per-user error", loc="left", weight="bold", fontsize=14, pad=19)
    subtitle = "100-bit messages   ·   30,000 real channel uses"
    if preview:
        subtitle += "   ·   Pilot: 8 geometries per measured load"
    ax.text(0., 1.012, subtitle, transform=ax.transAxes, fontsize=8.7, color="#4b5457")
    ax.grid(axis="y", alpha=.23, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    if notes:
        footer = notes[0] + ("\n" + "  ".join(notes[1:]) if len(notes) > 1 else "")
        fig.text(.1, .015, footer, fontsize=7.1, color="#4b5457", va="bottom", wrap=True)
    fig.subplots_adjust(top=.87, left=.105, right=.985, bottom=.19 if notes else .125)
    stem = "load_preview" if preview else "load_references" if summary is None else "required_ebn0_vs_load"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out / f"{stem}.pdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    report = None if args.references_only else json.loads((args.out / "summary.json").read_bytes())
    print(render(report, args.out))
