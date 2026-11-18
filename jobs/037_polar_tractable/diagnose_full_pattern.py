"""Frozen full-pattern continuation, all-bank count refit, and predetermined fresh panels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.polar_full_pattern import continue_full_patterns
from rollback_frames import geometry, make_frame, native
from rollback_progressive import progressive

PANELS = {"development": list(range(5701, 5705)), "reserved": list(range(5705, 5709)),
          "expanded_validation": list(range(5901, 5909))}
DECLARATION = ROOT / "results/polar_continued_largeB_2026-11-11/full_pattern_design.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    with Path(path).open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def evaluate(action, bank, labels, k):
    started = time.perf_counter()
    truth, action, bank = Counter(labels), Counter(action), set(bank)
    matched = (truth & action).total()
    absent_ids = sorted(set(truth).difference(bank))
    return {"matched_copies": matched, "misses": k - matched, "count_pupe": (k - matched) / k,
            "false_copies": action.total() - matched, "output_copies": action.total(),
            "bank_size": len(bank), "bank_absent_true_ids": absent_ids,
            "multiplicity_capable_oracle_bank_miss_floor": sum(truth[word] for word in absent_ids),
            "evaluation_seconds": time.perf_counter() - started}


def freeze(out, panel):
    out.mkdir(parents=True, exist_ok=True)
    names = ["design.json", "source_freeze.json", "summary.json", "completion.json"]
    names += [f"frame_{seed}.json" for seed in PANELS[panel]]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Existing frozen evidence will not be overwritten")
    design = json.loads(DECLARATION.read_text())
    if design["discovery"]["list_size"] != 128 or design["discovery"]["passes"] != 3:
        raise ValueError("Frozen declaration differs from this driver")
    with (out / "design.json").open("xb") as handle:
        handle.write(DECLARATION.read_bytes())
    sources = ["baselines/polar.py", "baselines/odma_polar.py", "benchmarks/polar_energy_geometry.py",
               "framework/polar_tractable.py", "framework/polar_full_pattern.py",
               "jobs/037_polar_tractable/rollback_frames.py", "jobs/037_polar_tractable/rollback_progressive.py",
               "jobs/037_polar_tractable/rollback_rescue.py", "jobs/037_polar_tractable/diagnose_full_pattern.py",
               "tests/polar_full_pattern_test.py"]
    for relative in sources:
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((ROOT / relative).read_bytes())
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "panel": panel, "seeds": PANELS[panel],
                "sources": {path: digest(ROOT / path) for path in sources}, "design_sha256": digest(out / "design.json")}
    write_json(out / "source_freeze.json", manifest)
    return manifest


def summarize(rows, panel, expected_seeds=None):
    expected_seeds = PANELS.get(panel) if expected_seeds is None else list(expected_seeds)
    if expected_seeds is None or [row["seed"] for row in rows] != expected_seeds:
        raise ValueError("Summary requires every predetermined frame in seed order")
    totals = {}
    for arm in ("native", "rollback", "refit_only", "full"):
        totals[arm] = {field: sum(row["metrics"][arm][field] for row in rows)
                       for field in ("matched_copies", "misses", "false_copies", "output_copies",
                                     "multiplicity_capable_oracle_bank_miss_floor")}
    novelty = sum(len(row["new_true_ids_absent_initial_bank"]) for row in rows)
    retained = sum(len(row["new_true_ids_retained_final"]) for row in rows)
    initial_absent = sum(len(row["metrics"]["rollback"]["bank_absent_true_ids"]) for row in rows)
    gates = {"at_least_two_fewer_misses_than_rollback": totals["full"]["misses"] <= totals["rollback"]["misses"] - 2,
             "at_most_two_extra_false_copies": totals["full"]["false_copies"] <= totals["rollback"]["false_copies"] + 2,
             "no_frame_more_than_two_extra_misses": all(r["metrics"]["full"]["misses"] <= r["metrics"]["rollback"]["misses"] + 2 for r in rows),
             "at_least_two_novel_true_ids": novelty >= 2, "novel_true_id_retained_final": retained >= 1,
             "fewer_misses_than_refit_only": totals["full"]["misses"] < totals["refit_only"]["misses"],
             "all_declared_executions_complete": all(r["continuation"]["complete"] for r in rows)}
    insufficient = panel == "development" and (totals["rollback"]["misses"] < 4 or initial_absent < 2)
    if insufficient:
        decision = "insufficient_opportunity_extend_development_with_all5705_to5708"
    elif all(gates.values()):
        decision = "capability_gate_passed_requires_predetermined_fresh_validation"
    else:
        decision = "declared_capability_gate_failed_for_this_panel"
    return {"totals": totals, "gate": gates, "gate_passed": all(gates.values()), "decision": decision,
            "initial_bank_absent_true_ids": initial_absent, "new_true_ids": novelty, "new_true_ids_retained": retained,
            "insufficient_opportunity": insufficient,
            "panel_boundary": "Reserved panel role and combined-development gate follow the frozen opportunity rule; algorithm unchanged."}


def main(out, panel):
    started = time.perf_counter()
    t0 = time.perf_counter()
    manifest = freeze(out, panel)
    freeze_seconds = time.perf_counter() - t0
    rows, serialization = [], []
    for seed in PANELS[panel]:
        t0 = time.perf_counter()
        model, balance = geometry(seed)
        geometry_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        scenario, labels = make_frame(model, seed, 0)
        source_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        accepted, native_meta = native(scenario)
        native_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        initial, fresh, batches, rollback_inner_seconds = progressive(scenario, list(accepted), native_meta["valid_ids"], "ungated")
        rollback_seconds = time.perf_counter() - t0
        base_bank = set(native_meta["valid_ids"]).union(fresh, initial)
        print(json.dumps({"seed": seed, "stage": "baseline_complete", "native_outputs": len(accepted),
                          "rollback_output_copies": initial.total(), "initial_bank_size": len(base_bank)}), flush=True)

        def progress(record):
            print(json.dumps({"seed": seed, "stage": "full_scan", **record}), flush=True)

        t0 = time.perf_counter()
        action, bank, continuation = continue_full_patterns(scenario, initial, base_bank, progress=progress)
        continuation_wall_seconds = time.perf_counter() - t0
        refit_only = Counter(dict(continuation["initial_refit_action"]))
        metrics = {"native": evaluate(Counter(accepted.keys()), native_meta["valid_ids"], labels, scenario.k),
                   "rollback": evaluate(initial, base_bank, labels, scenario.k),
                   "refit_only": evaluate(refit_only, base_bank, labels, scenario.k),
                   "full": evaluate(action, bank, labels, scenario.k)}
        new_true = set(labels).intersection(bank).difference(base_bank)
        retained = sorted(new_true.intersection(action))
        row = {"seed": seed, "frame": 0, "truth": labels, "truth_counts": sorted(Counter(labels).items()),
               "y_sha256": hashlib.sha256(scenario.y.tobytes()).hexdigest(),
               "positions_sha256": hashlib.sha256(model.positions.tobytes()).hexdigest(),
               "amplitudes_sha256": hashlib.sha256(model.amplitudes.tobytes()).hexdigest(), "balance": balance,
               "geometry_seconds": geometry_seconds, "frame_source_seconds": source_seconds,
               "native_wall_seconds": native_seconds, "rollback_wall_seconds": rollback_seconds,
               "rollback_inner_seconds": rollback_inner_seconds, "native_meta": native_meta, "rollback_batches": batches,
               "continuation_wall_seconds": continuation_wall_seconds,
               "rollback_action": sorted(initial.items()), "initial_bank": sorted(base_bank), "final_action": sorted(action.items()),
               "final_bank": sorted(bank), "continuation": continuation, "metrics": metrics,
               "new_true_ids_absent_initial_bank": sorted(new_true), "new_true_ids_retained_final": retained,
               "new_true_copies_retained_final": sum(min(action[word], Counter(labels)[word]) for word in retained),
               "receiver_seconds": {"native": native_seconds, "rollback": native_seconds + rollback_seconds,
                    "refit_only": native_seconds + rollback_seconds + continuation["initial_bank_seconds"]
                    + continuation["initial_refit"]["seconds"],
                    "full": native_seconds + rollback_seconds + continuation_wall_seconds}}
        row["receiver_with_geometry_seconds"] = {arm: seconds + geometry_seconds for arm, seconds in row["receiver_seconds"].items()}
        t0 = time.perf_counter()
        write_json(out / f"frame_{seed}.json", row)
        serialization.append({"seed": seed, "seconds": time.perf_counter() - t0})
        rows.append(row)
        print(json.dumps({"seed": seed, "stage": "frame_complete", "misses": {a: m["misses"] for a, m in metrics.items()},
                          "novel_true_ids": len(new_true), "retained_novel_true_ids": len(retained),
                          "stop": continuation["stop"], "elapsed_seconds": time.perf_counter() - started}), flush=True)
    scientific = summarize(rows, panel)
    unchanged = {path: digest(ROOT / path) == sha for path, sha in manifest["sources"].items()}
    if not all(unchanged.values()):
        raise AssertionError("A frozen source changed during the panel")
    costs = {arm: sum(row["receiver_seconds"][arm] for row in rows)
             for arm in ("native", "rollback", "refit_only", "full")}
    summary = {**scientific, "panel": panel, "seeds": PANELS[panel], "receiver_seconds": costs,
               "receiver_with_geometry_seconds": {arm: sum(row["receiver_with_geometry_seconds"][arm] for row in rows) for arm in costs},
               "source_freeze_seconds": freeze_seconds, "geometry_seconds": sum(row["geometry_seconds"] for row in rows),
               "frame_source_seconds": sum(row["frame_source_seconds"] for row in rows),
               "continuation_seconds_by_stage": {field: sum(row["continuation"][field] for row in rows)
                    for field in ("scl_seconds", "llr_seconds", "ranking_seconds", "query_code_setup_seconds",
                                  "bank_construction_seconds", "count_refit_seconds")},
               "continuation_queries": sum(len(row["continuation"]["queries"]) for row in rows),
               "count_moves": sum(len(row["continuation"]["initial_refit"]["moves"])
                    + sum(len(refit["meta"]["moves"]) for refit in row["continuation"]["refits"]) for row in rows),
               "evaluation_seconds": sum(metric["evaluation_seconds"] for row in rows for metric in row["metrics"].values()),
               "artifact_serialization_by_frame": serialization, "artifact_serialization_seconds": sum(r["seconds"] for r in serialization),
               "elapsed_seconds_before_summary_serialization": time.perf_counter() - started,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
               "source_hashes_unchanged": unchanged, "source_freeze_sha256": digest(out / "source_freeze.json"),
               "scope": "Complete bounded unknown-pattern continuation and count actions; approximate outside law and local optimization."}
    t0 = time.perf_counter()
    write_json(out / "summary.json", summary)
    summary_seconds = time.perf_counter() - t0
    write_json(out / "completion.json", {"complete": True, "panel": panel, "seeds": PANELS[panel],
                                         "summary_sha256": digest(out / "summary.json"),
                                         "summary_serialization_seconds": summary_seconds,
                                         "elapsed_seconds_before_completion_serialization": time.perf_counter() - started})
    print(json.dumps({"decision": scientific["decision"], "gate": scientific["gate"], "totals": scientific["totals"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", choices=tuple(PANELS), default="development")
    args = parser.parse_args()
    main(args.out, args.panel)
