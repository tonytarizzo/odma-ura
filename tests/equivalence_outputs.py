"""Shared outputs for explicit-codebook equivalence experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.plotting import _configure_matplotlib
from src.ura_bound import required_ebn0_curve


BOUND_LABELS = {
    "canonical": "Polyanskiy canonical",
    "count": "Polyanskiy count/multiset",
    "strict": "Polyanskiy strict collisions",
}
BOUND_STYLES = {
    "canonical": {"color": "#6B7280", "ls": (0, (5, 2)), "marker": "*"},
    "count": {"color": "#111827", "ls": (0, (1, 1)), "marker": "P"},
    "strict": {"color": "#B91C1C", "ls": (0, (3, 1, 1, 1)), "marker": "X"},
}


def plot_polyanskiy_bounds(rows: list[dict], out_path: Path, *, title: str) -> None:
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for variant in ("canonical", "count", "strict"):
        pts = sorted([r for r in rows if r["variant"] == variant], key=lambda r: r["K"])
        if not pts:
            continue
        x = np.array([p["K"] for p in pts], dtype=float)
        y = np.array([p["ebn0_db_experiment"] for p in pts], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        style = BOUND_STYLES[variant]
        ax.plot(x[finite], y[finite], lw=1.8, ms=6, color=style["color"],
                ls=style["ls"], marker=style["marker"], label=BOUND_LABELS[variant])
    ax.set_xlabel("Active devices K")
    ax.set_ylabel("Required Eb/N0 (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_polyanskiy_outputs(args: SimpleNamespace, out_dir: Path, *, target_pupe: float,
                             grid: int, num_pprime: int, axis: str = "experiment") -> list[dict]:
    if axis not in ("experiment", "physical"):
        raise ValueError(f"axis must be 'experiment' or 'physical', got {axis!r}")
    curve = required_ebn0_curve(
        n=int(args.n),
        payload_bits=int(args.payload_bits),
        k_values=[int(k) for k in args.K_values],
        target=float(target_pupe),
        variants=("canonical", "count", "strict"),
        num_antennas=int(args.num_antennas),
        grid=int(grid),
        num_pprime=int(num_pprime),
    )
    rows = []
    for K, entry in curve.items():
        for variant in ("canonical", "count", "strict"):
            plotted = float(entry[variant]["ebn0_db_experiment" if axis == "experiment" else "ebn0_db_phys"])
            rows.append({
                "K": int(K),
                "variant": variant,
                "target_pupe": float(target_pupe),
                "ebn0_db_phys": float(entry[variant]["ebn0_db_phys"]),
                "ebn0_db_experiment": plotted,
                "collision_floor_strict": float(entry["collision_floor_strict"]),
                "collision_prob_union": float(entry["collision_prob_union"]),
                "distinct_count": float(entry["distinct_count"]),
            })
    payload = {
        "target_pupe": float(target_pupe),
        "plotted_axis": axis,
        "args": {"n": int(args.n), "payload_bits": int(args.payload_bits),
                 "num_antennas": int(args.num_antennas), "K_values": [int(k) for k in args.K_values],
                 "grid": int(grid), "num_pprime": int(num_pprime)},
        "rows": rows,
    }
    (out_dir / "polyanskiy_bounds.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_polyanskiy_bounds(
        rows, out_dir / "polyanskiy_bounds.png",
        title=f"Polyanskiy required Eb/N0, PUPE<={target_pupe:g} (B={int(args.payload_bits)}, n={int(args.n)})")
    return rows


def empirical_required_ebn0(points: list[dict], target_pupe: float, *,
                            preferred: dict[str, str] | None = None) -> list[dict]:
    """Smallest evaluated Eb/N0 whose mean PUPE is below target for each curve."""
    preferred = preferred or {}
    keys = sorted({(p.get("family", p.get("preset", "unknown")), p["construction"], int(p["K"])) for p in points})
    rows = []
    for family, construction, K in keys:
        if family in preferred and construction != preferred[family]:
            continue
        curve = sorted([p for p in points if p.get("family", p.get("preset", "unknown")) == family
                        and p["construction"] == construction and int(p["K"]) == K],
                       key=lambda p: float(p["ebn0_db"]))
        reached = [p for p in curve if float(p.get("mean_pupe", math.inf)) <= float(target_pupe)]
        req = float(reached[0]["ebn0_db"]) if reached else math.inf
        rows.append({
            "family": family,
            "construction": construction,
            "K": int(K),
            "target_pupe": float(target_pupe),
            "required_ebn0_db": req,
            "status": "grid_reached" if reached else "not_reached",
            "num_grid_points": len(curve),
        })
    return rows


def plot_required_ebn0_with_bounds(empirical_rows: list[dict], bound_rows: list[dict],
                                   out_path: Path, *, title: str) -> None:
    _configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"dense": "#0E7490", "odma": "#7C2D12", "ccs": "#7C3AED"}
    markers = {"dense": "o", "odma": "s", "ccs": "D"}
    for family, construction in sorted({(r["family"], r["construction"]) for r in empirical_rows}):
        rows = sorted([r for r in empirical_rows if r["family"] == family and r["construction"] == construction],
                      key=lambda r: r["K"])
        x = np.array([r["K"] for r in rows], dtype=float)
        y = np.array([r["required_ebn0_db"] for r in rows], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        label = family if construction == "framework" else f"{family} ({construction})"
        ax.plot(x[finite], y[finite], lw=2, ms=5, marker=markers.get(family, "o"),
                color=colors.get(family, "#333333"), label=label)

    for variant in ("canonical", "count", "strict"):
        rows = sorted([r for r in bound_rows if r["variant"] == variant], key=lambda r: r["K"])
        if not rows:
            continue
        x = np.array([r["K"] for r in rows], dtype=float)
        y = np.array([r["ebn0_db_experiment"] for r in rows], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        style = BOUND_STYLES[variant]
        ax.plot(x[finite], y[finite], lw=1.8, ms=6, color=style["color"],
                ls=style["ls"], marker=style["marker"], label=BOUND_LABELS[variant])

    ax.set_xlabel("Active devices K")
    ax.set_ylabel("Required Eb/N0 (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_required_ebn0_outputs(points: list[dict], bound_rows: list[dict], out_dir: Path, *,
                                target_pupe: float, preferred: dict[str, str] | None = None,
                                title: str) -> list[dict]:
    empirical = empirical_required_ebn0(points, target_pupe, preferred=preferred)
    payload = {"target_pupe": float(target_pupe), "threshold_rule": "smallest evaluated Eb/N0 with mean PUPE <= target",
               "empirical": empirical, "polyanskiy_bounds": bound_rows}
    (out_dir / "required_ebn0_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    plot_required_ebn0_with_bounds(empirical, bound_rows, out_dir / "required_ebn0_with_bounds.png", title=title)
    return empirical


def write_validation_report(out_dir: Path, *, scheme: str, max_phi_err: float, decoded_match: bool,
                            paper_refs: dict[str, str], notes: list[str],
                            polyanskiy_written: bool, required_ebn0_written: bool = False) -> None:
    lines = [
        f"# {scheme} validation report",
        "",
        "## Algebraic equivalence",
        "",
        f"- Max absolute `Phi` error: `{max_phi_err:.3e}`",
        f"- Implemented decoder and framework decoder counts matched exactly: `{bool(decoded_match)}`",
        "",
        "## Reference context",
        "",
    ]
    if paper_refs:
        for label, url in paper_refs.items():
            lines.append(f"- `{label}`: {url}")
    else:
        lines.append("- No external scheme-specific paper reference was configured for this script.")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Polyanskiy bounds", ""])
    if polyanskiy_written:
        lines.append(
            "- Wrote `polyanskiy_bounds.json` and `polyanskiy_bounds.png` for canonical/count/strict collision "
            "conventions.")
    else:
        lines.append("- Polyanskiy bound generation was skipped for this run.")
    lines.extend(["", "## Required Eb/N0", ""])
    if required_ebn0_written:
        lines.append("- Wrote `required_ebn0_summary.json` and `required_ebn0_with_bounds.png`.")
    else:
        lines.append("- Required-Eb/N0 curve generation was skipped for this run.")
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n")


def default_section_bits(payload_bits: int, num_sections: int) -> int:
    return payload_bits if int(num_sections) == 1 else int(math.ceil(int(payload_bits) / (int(num_sections) - 1)))
