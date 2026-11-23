"""Render the paired energy curves and export their measured performance and cost."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np

NAMES = dict(native="Native SIC", rollback="Progressive rollback", growth="DEC-35")
COLORS = ("#405568", "#b56721", "#18735e")


def write_tables_and_figure(result, out):
    out = Path(out)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.6), sharey=True, layout="constrained")
    for axis, k in zip(axes, (50, 100), strict=True):
        load = result["loads"][str(k)]
        db = np.asarray(load["snr_cdb"]) / 100
        for (name, label), color in zip(NAMES.items(), COLORS, strict=True):
            points = [load["by_snr"][str(snr)]["endpoints"][name] for snr in load["snr_cdb"]]
            rate = np.asarray([p["list_pupe"] for p in points]) * 100
            lo, hi = 100 * np.asarray([p["list_pupe_pointwise_95"] for p in points]).T
            axis.fill_between(db, lo, hi, color=color, alpha=.11)
            # Display the same raw-PUPE linear interpolation used by the crossing estimator.
            dense_db = np.unique(np.concatenate([np.linspace(a,b,33) for a,b in zip(db[:-1],db[1:],strict=True)]))
            axis.plot(dense_db, np.interp(dense_db,db,rate), color=color, label=label, linewidth=1.5)
            axis.plot(db, rate, ".", color=color, markersize=4)
            crossing = load["crossings"]["list_pupe"]["thresholds"][name]["snr_db"]
            if crossing is not None:
                axis.plot(crossing, 5, "o", color=color, markeredgecolor="white", markeredgewidth=.7, markersize=5, zorder=3)
            for level, point in zip(db, points, strict=True):
                r = dict(active_users=k, ebn0_db=level, receiver=name, frames=load["frames_per_snr"], users=load["users_per_snr"])
                for metric in ("list_pupe", "count_pupe"):
                    r[metric] = point[metric]
                    r[metric+"_lo95"], r[metric+"_hi95"] = point[metric+"_pointwise_95"]
                for m in ("list_misses","count_misses","false_copies","list_size","output_copies"):
                    r[m] = point[m]
                for m in ("seconds_with_geometry","receiver_seconds","native_scl128","rollback_scl512","continuation_scl128"):
                    for stat in ("mean","p50","p95","maximum"):
                        r[m+"_"+stat] = point[m][stat]
                records.append(r)
        axis.set_yscale("symlog", linthresh=1, linscale=.6)
        axis.yaxis.set_major_locator(FixedLocator([0, 1, 2, 5, 10, 25, 50, 100]))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.set_ylim(-.05, 104)
        axis.axhline(5, color=".45", linestyle="--", linewidth=.9, label="5% target")
        axis.set_title(f"{k} active users")
        axis.set_xlabel(r"$E_b/N_0$ (dB)")
        axis.grid(alpha=.16)
    axes[0].set_ylabel("Missed transmitted messages (%)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.savefig(out / "list_pupe.pdf")
    plt.close(fig)
    with (out / "performance_cost.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_crossings(curves, sensitivity, out):
    with (out / "required_energy.csv").open("w", newline="") as thresholds, (out / "energy_gain.csv").open("w", newline="") as gains:
        t, g = csv.writer(thresholds), csv.writer(gains)
        t.writerow(["panel","K","metric","receiver","status","crossing_dB","bracket_lo","bracket_hi","censor_lo","censor_hi","ci95_lo","ci95_hi"])
        g.writerow(["panel","K","metric","first_minus_second","dB_saved","ci95_lo","ci95_hi","two_load_ci_lo","two_load_ci_hi"])
        for k in (50,100):
            main = curves["loads"][str(k)]
            extra = sensitivity[str(k)]
            for panel,metric,table,contrasts in [
                ("main",m,e["thresholds"],e["required_energy_contrasts"]) for m,e in main["crossings"].items()
            ] + [("detector_sensitivity","list_pupe",extra["thresholds"],extra["contrasts"])]:
                for name,row in table.items():
                    bracket = row["observed_grid_bracket_db"] or [None,None]
                    limits = row["paired_bootstrap_95_db"] if panel=="main" else row["bootstrap_95_db"]
                    t.writerow([panel,k,metric,name,row["status"],row["snr_db"],*bracket,*row["censor_bounds_db"],*limits])
                for pair,row in contrasts.items():
                    adjusted = row.get("bonferroni_two_loads_97_5_db",[None,None])
                    g.writerow([panel,k,metric,pair,row["snr_db_saved_by_second"],*row["paired_bootstrap_95_db"],*adjusted])
