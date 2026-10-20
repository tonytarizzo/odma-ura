"""Summarize existing, completed results for a supervisor discussion; runs no receiver or simulation."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEEDS = (3201, 3202, 3203)
BLUE, GREEN, ORANGE, GRAY, PURPLE = "#336c99", "#22846b", "#c57a28", "#687585", "#8260a2"
SPECS = {
    "dense": ("032_published_baselines", "comparison/B{b}_n256_dense_fixed_d2_s{seed}"),
    "sparse": ("032_published_baselines", "comparison/B{b}_n256_sparse_joint_d2_s{seed}"),
    "hash_full": ("033_matched_d2", "B{b}_n256_hash_full_joint_d2_s{seed}"),
    "hash_j8": ("033_matched_d2", "B{b}_n256_hash_coordinate_J8_joint_d2_s{seed}"),
    "polar": ("033_matched_d2", "B{b}_n256_odma_polar_d2_s{seed}"),
    "energy": ("034_polar_mapping", "B{b}_n256_polar_energy_joint_d2_s{seed}"),
    "mixer": ("034_polar_mapping", "B{b}_n256_polar_mixer_joint_d2_s{seed}"),
    "native": ("032_published_baselines", "comparison/B{b}_n256_odma_polar_native_s{seed}"),
}


def read_data():
    data, provenance = [], []
    for b in (12, 14):
        for key, (job, name) in SPECS.items():
            for seed in SEEDS:
                path = ROOT / "jobs" / job / "results" / name.format(b=b, seed=seed) / "summary.json"
                raw = path.read_bytes()
                summary = json.loads(raw)
                assert summary["status"] == "complete", path
                config = summary["config"]
                assert config["B"] == b and config["n"] == 256 and config["seed"] == seed
                evaluation = [r for r in summary["evaluation"] if r["sampling"] == "distinct"]
                assert {(r["K"], r["ebn0_db"]) for r in evaluation} == {
                    (k, snr) for k in (7, 15, 22, 26) for snr in range(-4, 11, 2)}
                assert len(evaluation) == 32 and all(r["frames"] == 256 for r in evaluation)
                for r in evaluation:
                    assert len(r["frame_metrics"]) == 256
                    metric = float(np.mean([f["pupe"] for f in r["frame_metrics"]]))
                    np.testing.assert_allclose(metric, r["means"]["pupe"], atol=1e-12)
                    assert all(f["duplicate_users"] == 0 for f in r["frame_metrics"])
                    assert all(f["list_size"] <= r["K"] for f in r["frame_metrics"])
                audit = summary.get("energy_audit_final") or summary.get("energy_audit_initial")
                provenance.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "family": key, "B": b, "seed": seed, "encoder_mode": config["mode"],
                                   "energy_audit": audit})
                data.append({"B": b, "family": key, "seed": seed,
                             "full_grid": float(np.mean([r["means"]["pupe"] for r in evaluation])),
                             "K26_10dB": next(r["means"]["pupe"] for r in evaluation if r["K"] == 26 and r["ebn0_db"] == 10),
                             "K26_curve": [{"snr": r["ebn0_db"], "pupe": r["means"]["pupe"]}
                                           for r in evaluation if r["K"] == 26]})
    posthoc_path = ROOT / "jobs/034_polar_mapping/results/analysis/gap_investigation_2026-10-10/aggregate.json"
    posthoc = [r for r in json.loads(posthoc_path.read_text()) if r["B"] == 14 and r["snr"] == 10 and
               r["family"] in ("dense", "energy", "balanced") and r["method"] in ("own_d2", "polished_d2")]
    assert len(posthoc) == 6 and all(r["frames"] == 1536 for r in posthoc)
    provenance.append({"path": str(posthoc_path.relative_to(ROOT)), "sha256": hashlib.sha256(posthoc_path.read_bytes()).hexdigest()})
    payload = {"created": "2026-10-18", "contract": "Historical distinct-message results; real AWGN, known K/noise, n=256",
               "production": data, "diagnostic": posthoc, "source_files": provenance}
    (OUT / "plot_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    return data, posthoc


def values(data, family, b=14, field="full_grid"):
    rows = sorted((r for r in data if r["family"] == family and r["B"] == b), key=lambda r: r["seed"])
    assert len(rows) == 3
    return np.array([r[field] for r in rows])


def make_overview(data, posthoc):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), gridspec_kw={"width_ratios": (1.0, 1.25, 1.05)})
    ax = axes[0]
    families = ("native", "polar", "dense")
    labels = ("Polar: native receiver", "Same polar: D2", "Dense: D2")
    for y, key, label, color in zip(range(3), families, labels, (GRAY, GREEN, BLUE)):
        v = values(data, key, field="K26_10dB")
        ax.barh(y, v.mean(), height=.5, color=color)
        ax.text(v.mean()+.012, y, f"{v.mean():.3f}", va="center", fontsize=11)
    ax.set_yticks(range(3), labels); ax.set_ylim(2.65, -.65); ax.set_xlim(0, .65)
    ax.set_title("A. Changing the receiver", loc="left", pad=31, fontsize=13)
    ax.text(0, 1.05, "B=14, K=26, 10 dB | original test grid", transform=ax.transAxes, fontsize=10, color=GRAY)
    ax.set_xlabel("PUPE"); ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)

    ax = axes[1]
    families = ("dense", "sparse", "hash_full", "hash_j8", "polar", "energy", "mixer")
    labels = ("Dense (fixed)", "Random sparse (learned)", "Hash + stored amplitudes",
              "Hash + generated J8", "Polar (uniform energy)", "Polar + learned energy", "Polar + shared mixer")
    colors = (BLUE, GRAY, PURPLE, PURPLE, GREEN, GREEN, GRAY)
    for y, key, color in zip(range(7), families, colors):
        v = values(data, key)
        ax.plot([v.min(), v.max()], [y, y], color=color, alpha=.45, lw=2)
        ax.scatter(v, np.full(3, y), color=color, s=13, alpha=.45)
        ax.scatter(v.mean(), y, color=color, s=58, marker="D", zorder=3)
        ax.text(.323, y, f"{v.mean():.5f}", va="center", ha="right", fontsize=10)
    ax.set_yticks(range(7), labels); ax.set_ylim(6.65, -.65); ax.set_xlim(.275, .325)
    ax.set_xticks((.28, .30, .32)); ax.grid(axis="x", alpha=.15)
    ax.set_title("B. Encoder families stay close", loc="left", pad=31, fontsize=13)
    ax.text(0, 1.05, "B=14 | whole-grid mean | zoomed axis", transform=ax.transAxes, fontsize=10, color=GRAY)
    ax.set_xlabel("Mean PUPE; line = range across 3 seeds")

    ax = axes[2]
    for y, family in enumerate(("dense", "energy", "balanced")):
        first, second = [next(r["pupe"] for r in posthoc if r["family"] == family and r["method"] == method)
                         for method in ("own_d2", "polished_d2")]
        ax.plot([first, second], [y, y], color=GRAY, lw=1.5, alpha=.5)
        ax.scatter(first, y, color=BLUE, s=62, label="D2" if y == 0 else None, zorder=3)
        ax.scatter(second, y, color=ORANGE, s=65, marker="D", label="D2 + refinement" if y == 0 else None, zorder=3)
        ax.text(first+.004, y, f"{first:.3f}", va="center", fontsize=10)
        ax.text(second-.004, y, f"{second:.3f}", va="center", ha="right", fontsize=10)
    ax.set_yticks(range(3), ("Dense", "Polar: learned energy", "Polar: balanced"))
    ax.set_ylim(2.65, -.65); ax.set_xlim(.015, .125); ax.set_xticks((.02, .06, .10))
    ax.set_title("C. More inference headroom", loc="left", pad=31, fontsize=13)
    ax.text(0, 1.05, "B=14, K=26, 10 dB | fresh diagnostic", transform=ax.transAxes, fontsize=10, color=GRAY)
    ax.set_xlabel("PUPE; fixed codebook, no new training"); ax.grid(axis="x", alpha=.15)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.text(.012, .085, "Real AWGN, n=256, known K/noise. Native scheme is a small-B adaptation, not a paper-scale reproduction.", fontsize=10, color=GRAY)
    fig.text(.012, .045, "A/B: 3 seeds; 256 frames/cell/seed. B: K=7,15,22,26; -4:2:10 dB; all use trained D2. Only dense/uniform polar have fixed encoders.", fontsize=10, color=GRAY)
    fig.text(.012, .006, "C: 512 fresh frames/seed; balanced polar uses transferred D2. D2 and likelihood refinement still scan the full alphabet.", fontsize=10, color=GRAY)
    fig.tight_layout(rect=(0, .13, 1, .96), w_pad=2.7)
    fig.savefig(OUT / "research_direction.png", dpi=210)
    plt.close(fig)


def make_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    specifications = (("native", "Polar: native receiver", GRAY, ":"), ("dense", "Dense / D2", BLUE, "-"),
                      ("sparse", "Learned sparse / D2", "#9b8a49", "--"),
                      ("hash_j8", "Learned hash J8 / D2", PURPLE, "--"),
                      ("polar", "Uniform polar / D2", GREEN, ":"),
                      ("energy", "Learned-energy polar / D2", GREEN, "-"))
    for ax, b in zip(axes, (12, 14)):
        for family, label, color, linestyle in specifications:
            rows = [r for r in data if r["family"] == family and r["B"] == b]
            x = np.array([p["snr"] for p in sorted(rows[0]["K26_curve"], key=lambda v: v["snr"])])
            all_y = np.array([[p["pupe"] for p in sorted(r["K26_curve"], key=lambda v: v["snr"])] for r in rows])
            ax.plot(x, all_y.mean(0), color=color, ls=linestyle, lw=1.9, marker="o", ms=3, label=label)
        ax.set_title(f"B={b}, n=256, K=26"); ax.set_xlabel(r"$E_b/N_0$ (dB)")
        ax.set_ylim(-.025, 1.025); ax.set_xticks(range(-4, 11, 2)); ax.grid(alpha=.2)
    axes[0].set_ylabel("PUPE (lower is better)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Historical matched comparisons: average of three seeds", fontsize=13)
    fig.tight_layout(rect=(0, .13, 1, .95)); fig.savefig(OUT / "performance_curves.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    data, diagnostic = read_data()
    make_overview(data, diagnostic)
    make_curves(data)
    for b in (12, 14):
        print("B", b, {key: round(values(data, key, b).mean(), 6) for key in SPECS})
    print("Verified 48 completed summaries, distinct sampling, full grids, budgets, and stored frame means.")
