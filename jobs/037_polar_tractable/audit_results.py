"""Audit returned 035--037 frames and compare only complete, explicitly matched seed/cell cohorts."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import interval, normalized_config, source_hashes, write_json
from framework.polar_decoders import REGISTRY

JOB = Path(__file__).resolve().parent
CONTROL_NAMES = {"D2", "ODMA", "DEC-02", "DEC-04", "DEC-05", "DEC-06", "DEC-07"}
DIAGNOSTICS = ("candidate_recall", "true_configuration_in_candidates", "initial_pupe", "effective_sample_size",
               "accepted", "changed", "attempts", "terminal_changed_blocks", "terminal_attempts",
               "expected_excess_risk_bound", "optimizer_converged", "cached_waveform_bytes")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def plots(out):
    import matplotlib.pyplot as plt
    rows = json.loads((out / "aggregates.json").read_text())
    selected = ["D2", "DEC-14", "DEC-25", "DEC-26", "DEC-29", "DEC-31", "DEC-32"]
    for b in (12, 14):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, k in zip(axes.ravel(), (7, 15, 22, 26)):
            for name in selected:
                values = sorted((r for r in rows if r["B"] == b and r["K"] == k and r["decoder"] == name
                                 and r["seeds"] == [3201, 3202, 3203]), key=lambda r: r["ebn0_db"])
                ax.plot([r["ebn0_db"] for r in values], [r["pupe"] for r in values], "o-", markersize=3, label=name)
            ax.set(title=f"K={k}", xlabel="Eb/N0 (dB)", ylabel="Count PUPE", ylim=(0, 1))
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7)
        fig.suptitle(f"B{b}: complete paired runs, 3 seeds × 256 frames/cell; exploratory")
        fig.tight_layout(rect=(0, .06, 1, .95))
        fig.savefig(out / f"B{b}_decoder_selection.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, b in zip(axes, (12, 14)):
        for r in rows:
            if r["B"] == b and r["K"] == 26 and r["ebn0_db"] == 4 and len(r["seeds"]) == 3:
                ax.scatter(r["seconds"], r["pupe"])
                ax.annotate(r["decoder"], (r["seconds"], r["pupe"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set(xscale="log", xlabel="Recorded seconds/frame (different HPC executions)", ylabel="Count PUPE",
               title=f"B{b}, K26, 4 dB")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "cost_accuracy.png", dpi=160)
    plt.close(fig)


def audit(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    contract = json.loads((JOB / "reference_contract.json").read_text())
    if source_hashes() != contract["current_source_sha256"]:
        raise ValueError("Shared runtime differs from the reviewed Job037 snapshot")
    allowed = [contract["current_source_sha256"], *[x["source_sha256"] for x in contract["historical_snapshots"]]]
    roots = {"035": ROOT / "jobs/035_polar_decoders/results_all_decoders",
             "036": ROOT / "jobs/036_polar_combinations/results_combinations", "037": JOB / "results_tractable"}
    inventory, cells, arrays, failures = [], [], {}, []
    for campaign, root in roots.items():
        manifest = [r for line in (root.parent / "manifest.jsonl").read_text().splitlines()
                    for r in json.loads(line)["runs"]]
        for row in manifest:
            name = row["decoders"][0]
            if row["mapping"] != "balanced" or (campaign == "035" and name not in CONTROL_NAMES):
                continue
            directory = root / row["name"]
            retry = root.parent / "results_balanced_is_retry" / row["name"]
            if campaign == "035" and (retry / "summary.json").exists():
                directory = retry
            sf, ff = directory / "summary.json", directory / "frames.jsonl"
            status = "complete" if sf.exists() else "partial_uncertified" if ff.exists() else "absent"
            item = dict(campaign=campaign, run=row["name"], path=str(directory.relative_to(ROOT)), B=row["B"],
                        seed=row["seed"], decoder=name, loads=row["loads"], status=status, frames=0)
            inventory.append(item)
            if status == "absent":
                continue
            c = json.loads((directory / "config.json").read_text())
            if c != normalized_config(row):
                raise ValueError(f"Manifest/config mismatch: {directory}")
            summary = json.loads(sf.read_text()) if sf.exists() else None
            params = {**dict(REGISTRY[name].defaults), **c["decoder_params"].get(name, {})}
            if summary:
                if (summary["status"] != "complete" or summary["config"] != c
                        or summary["source_sha256"] not in allowed or summary["decoder_params"] != {name: params}
                        or summary["pupe_definition"] != "multiset_matching_v2"):
                    raise ValueError(f"Unreviewed source/config/parameters/metric: {directory}")
                item.update(summary_sha256=digest(sf.read_bytes()), revision=summary["revision"],
                            source_sha256=summary["source_sha256"])
            reference = roots["035"] / f"B{c['B']}_n256_balanced_iid_D2_s{c['seed']}"
            reference_summary = json.loads((reference / "summary.json").read_text())
            with np.load(directory / "encoder.npz") as f:
                matrix_hash = digest(f["phi"].tobytes())
            if matrix_hash != reference_summary["matrix_sha256"] or (summary and matrix_hash != summary["matrix_sha256"]):
                raise ValueError(f"Unpaired matrix: {directory}")
            item["matrix_sha256"] = matrix_hash
            inputs, groups, seen = {}, defaultdict(list), set()
            initial = ff.stat()
            file_hash = hashlib.sha256()
            with ff.open("rb") as stream:
                for line in stream:
                    file_hash.update(line)
                    r = json.loads(line)
                    k, snr, frame = r["K"], r["ebn0_db"], r["frame"]
                    key = k, snr
                    if ((r["decoder"], r["seed"], r["sampling"]) != (name, c["seed"], "iid") or k not in c["loads"]
                            or snr not in c["eval_ebn0"] or frame not in range(c["eval_frames"]) or (k, snr, frame) in seen):
                        raise ValueError(f"Unexpected/duplicate frame: {directory}")
                    seen.add((k, snr, frame))
                    if key not in inputs:
                        filename = f"inputs/iid_K{k}_EbN0{snr:g}.npz"
                        with np.load(directory / filename) as f:
                            data = dict(f)
                        with np.load(reference / filename) as f:
                            if data.keys() != set(f.files) or any(not np.array_equal(data[x], f[x]) for x in data):
                                raise ValueError(f"Unpaired physical inputs: {directory}/{filename}")
                        inputs[key] = data
                    data = inputs[key]
                    truth = Counter(data["message_ids"][frame])
                    counts = dict(r["counts"])
                    if (len(counts) != len(r["counts"]) or sum(counts.values()) > k
                            or any(type(w) is not int or type(v) is not int or not 0 <= w < 2**c["B"] or v < 1
                                   for w, v in counts.items())):
                        raise ValueError(f"Illegal counts: {directory}")
                    pupe = 1 - sum(min(v, truth[w]) for w, v in counts.items()) / k
                    if (abs(pupe - r["metrics"]["pupe"]) > 1e-12 or r["metrics"]["pupe_metric_version"] != 2
                            or r["input_sha256"] != digest(data["y"][frame].tobytes()) or not np.isfinite(r["seconds"])):
                        raise ValueError(f"Metric/input/timing mismatch: {directory}")
                    meta = r["meta"]
                    diag = {x: float(meta[x]) for x in DIAGNOSTICS if isinstance(meta.get(x), (bool, int, float))}
                    for field, value in meta.items():
                        if any(x in field for x in ("accept", "change", "initial_", "attempt")) and isinstance(value, (bool, int, float)):
                            diag[field] = float(value)
                    diag.update({"work_" + field: float(value) for field, value in meta.get("work", {}).items()
                                 if isinstance(value, (int, float))})
                    if "candidate_ids" in meta:
                        candidates = set(meta["candidate_ids"])
                        recall = sum(v for w, v in truth.items() if w in candidates) / k
                        if abs(recall - meta["candidate_recall"]) > 1e-12:
                            raise ValueError(f"Candidate recall mismatch: {directory}")
                        diag["scored_words"] = len(candidates)
                        diag["candidate_missing_copy_floor"] = 1 - recall
                    if "tail_probabilities" in meta:
                        capacity = {w: max((j for j, value in enumerate(tails, 1) if value > 0), default=0)
                                    for w, tails in meta["tail_probabilities"]}
                        floor = 1 - sum(min(v, capacity.get(w, 0)) for w, v in truth.items()) / k
                        diag["positive_tail_capacity_floor"] = floor
                        diag["pupe_above_tail_capacity_floor"] = pupe - floor
                    groups[key].append((frame, pupe, r["seconds"], digest(json.dumps(r["counts"]).encode()), diag))
            if (ff.stat().st_size, ff.stat().st_mtime_ns) != (initial.st_size, initial.st_mtime_ns):
                raise ValueError(f"Result changed during audit: {directory}")
            item.update(frames=len(seen), frames_sha256=file_hash.hexdigest())
            if summary and len(seen) != len(c["loads"]) * len(c["eval_ebn0"]) * c["eval_frames"]:
                raise ValueError(f"Complete run has missing frames: {directory}")
            for (k, snr), records in groups.items():
                if len(records) != c["eval_frames"]:
                    continue
                records.sort()
                values = np.array([r[1] for r in records])
                seconds = np.array([r[2] for r in records])
                if summary:
                    e = next(e for e in summary["evaluation"] if (e["K"], e["ebn0_db"]) == (k, snr))
                    if abs(e["pupe"]["mean"] - values.mean()) > 1e-12:
                        raise ValueError(f"Summary mean mismatch: {directory}")
                diag_keys = set.intersection(*(set(r[4]) for r in records))
                cell = dict(B=c["B"], decoder=name, seed=c["seed"], K=k, ebn0_db=snr, status=status,
                            frames=len(records), pupe=float(values.mean()), seconds=float(seconds.mean()),
                            diagnostics={x: float(np.mean([r[4][x] for r in records])) for x in sorted(diag_keys)})
                cells.append(cell)
                akey = c["B"], name, c["seed"], k, snr
                if akey in arrays:
                    raise ValueError(f"Duplicate cell: {akey}")
                arrays[akey] = values, [r[3] for r in records], status
            print(campaign, row["name"], status, len(seen), flush=True)
        for log in sorted(root.parent.glob("logs*/*.log")):
            content = log.read_text(errors="replace")
            if "Traceback" in content:
                failures.append(dict(path=str(log.relative_to(ROOT)), tail=content[-1600:]))
                names = re.findall(r"^Running [^\n]+: (\S+)\s*$", content, re.M)
                if "Source changed during" in content and names:
                    for item in inventory:
                        if (item["campaign"], item["run"], item["status"]) == (campaign, names[-1], "partial_uncertified"):
                            item["status"] = "source_guard_failed"
                            for cell in cells:
                                if (cell["B"], cell["seed"], cell["decoder"]) == (item["B"], item["seed"], item["decoder"]):
                                    if cell["K"] in item["loads"]:
                                        cell["status"] = "source_guard_failed"
    aggregates, contrasts = [], []
    for b in (12, 14):
        for seeds in ([3201], [3201, 3202], [3201, 3202, 3203]):
            for k in (7, 15, 22, 26):
                for snr in (-4, -2, 0, 2, 4, 6, 8, 10):
                    complete = {}
                    for name in sorted({r["decoder"] for r in cells}):
                        keys = [(b, name, seed, k, snr) for seed in seeds]
                        if any(key not in arrays or arrays[key][2] != "complete" for key in keys):
                            continue
                        complete[name] = keys
                        cs = [r for r in cells if r["B"] == b and r["decoder"] == name and r["seed"] in seeds
                              and r["K"] == k and r["ebn0_db"] == snr]
                        diagnostics = set.intersection(*(set(r["diagnostics"]) for r in cs))
                        aggregates.append(dict(B=b, seeds=seeds, decoder=name, K=k, ebn0_db=snr,
                                               pupe=float(np.mean([r["pupe"] for r in cs])),
                                               seconds=float(np.mean([r["seconds"] for r in cs])),
                                               diagnostics={x: float(np.mean([r["diagnostics"][x] for r in cs]))
                                                            for x in diagnostics}))
                    for name, keys in complete.items():
                        for other in ("D2", "DEC-14", "DEC-16", "DEC-17", "DEC-20", "DEC-26", "DEC-32"):
                            if other not in complete or other == name:
                                continue
                            differences = [arrays[a][0] - arrays[z][0] for a, z in zip(keys, complete[other])]
                            delta = np.concatenate(differences)
                            contrasts.append(dict(B=b, seeds=seeds, decoder=name, reference=other, K=k, ebn0_db=snr,
                                                  delta=float(delta.mean()), paired_frame_interval=interval(delta, bounded=False),
                                                  paired_seed_interval=interval([v.mean() for v in differences], bounded=False),
                                                  better_frames=int(np.sum(delta < -1e-12)),
                                                  worse_frames=int(np.sum(delta > 1e-12)),
                                                  identical_count_fraction=float(np.mean([x == y for a, z in zip(keys, complete[other])
                                                      for x, y in zip(arrays[a][1], arrays[z][1])]))))
    out.mkdir(parents=True)
    for name, value in (("inventory", inventory), ("cells", cells), ("aggregates", aggregates),
                        ("contrasts", contrasts), ("log_failures", failures)):
        write_json(out / f"{name}.json", value)
    write_json(out / "audit.json", dict(scope="local snapshot; partial runs excluded from contrasts; exploratory seeds",
                                        verified_frames=sum(x["frames"] for x in inventory),
                                        status={campaign: dict(Counter(x["status"] for x in inventory if x["campaign"] == campaign))
                                                for campaign in roots}, source_contract=contract,
                                        analysis_sha256=digest(Path(__file__).read_bytes())))
    print(out)
    plots(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="Render an existing audited snapshot without rereading runs")
    args = parser.parse_args()
    plots(args.out) if args.plot_only else audit(args.out)
