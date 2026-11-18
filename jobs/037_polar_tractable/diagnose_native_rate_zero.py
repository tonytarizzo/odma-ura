"""Fresh paired complete-receiver cost test of native-stage-only rate-zero SCL collapse."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baselines.polar import PolarCode
from framework.polar_count_feedback import run as reference
from framework.polar_fast_scl import RateZeroPolarCode
from framework.polar_native_rate_zero import run as native_only
from framework.polar_tractable import PolarScenario, integer
from rollback_frames import geometry, make_frame, native
from diagnose_count_feedback import trace_signature
from diagnose_full_pattern import digest, evaluate, write_json

BASE = ROOT / "results/polar_dec35_exploration_2026-11-11/rate_zero/receiver"
SEEDS = list(range(6701, 6717))
NATIVE_KEYS = ("calls", "rounds", "valid_ids", "residual_sha256")
SOURCES = ["baselines/polar.py", "baselines/odma_polar.py", "benchmarks/polar_energy_geometry.py",
    "framework/polar_tractable.py", "framework/polar_full_pattern.py", "framework/polar_count_cavity.py",
    "framework/polar_count_feedback.py", "framework/polar_fast_scl.py", "framework/polar_native_rate_zero.py",
    "tests/polar_fast_scl_test.py", "tests/polar_native_rate_zero_test.py",
    "jobs/037_polar_tractable/rollback_frames.py", "jobs/037_polar_tractable/rollback_progressive.py",
    "jobs/037_polar_tractable/rollback_rescue.py", "jobs/037_polar_tractable/diagnose_full_pattern.py",
    "jobs/037_polar_tractable/diagnose_count_feedback.py", "jobs/037_polar_tractable/diagnose_native_rate_zero.py",
    "pyproject.toml", "uv.lock"]


class RecordingMixin:
    def __init__(self, *args):
        super().__init__(*args)
        self.trace = []

    def decode_list(self, llr):
        payloads, metrics = super().decode_list(llr)
        self.trace.append((np.asarray(llr, dtype=np.float64).tobytes(), payloads.copy(), metrics.copy()))
        return payloads, metrics


class RecordedOriginal(RecordingMixin, PolarCode):
    pass


class RecordedRateZero(RecordingMixin, RateZeroPolarCode):
    pass


def native_replay(scenario, provider):
    started = time.perf_counter()
    old = scenario.model.code
    code = provider(old.payload_bits, old.n, old.crc_bits, old.list_size)
    model = copy.copy(scenario.model)
    model.code = code
    copied = PolarScenario(model, scenario.y, scenario.k, scenario.noise_var, scenario.sampling)
    accepted, meta = native(copied)
    return accepted, meta, code.trace, time.perf_counter() - started


def compare_native(scenario, arms):
    recorded, traces = {}, {}
    for name, provider in (("reference", RecordedOriginal), ("native_only", RecordedRateZero)):
        accepted, meta, trace, seconds = native_replay(scenario, provider)
        traces[name] = trace
        recorded[name] = {"accepted": sorted(accepted.items()), "meta": meta, "seconds": seconds,
                          "matches_timed_aggregate": all(meta[key] == arms[name]["meta"]["native"][key] for key in NATIVE_KEYS)}
    started = time.perf_counter()
    old, fast = traces["reference"], traces["native_only"]
    same_inputs = len(old) == len(fast)
    same_payloads, metrics_ok, maximum_error = same_inputs, True, 0.
    differences = []
    for index, (left, right) in enumerate(zip(old, fast)):
        inputs_equal, payloads_equal = left[0] == right[0], np.array_equal(left[1], right[1])
        same_inputs = same_inputs and inputs_equal
        same_payloads = same_payloads and payloads_equal
        if inputs_equal and payloads_equal and len(left[2]):
            error = float(np.max(np.abs(left[2] - right[2])))
            maximum_error = max(maximum_error, error)
            metrics_ok = metrics_ok and error <= 5e-12 * (1 + np.max(np.abs(left[2])))
        if not inputs_equal or not payloads_equal:
            differences.append({"query": index, "same_llr_bytes": inputs_equal, "same_ordered_payloads": payloads_equal})
    for name, trace in traces.items():
        recorded[name]["queries"] = [{"query": index, "llr_sha256": hashlib.sha256(llr).hexdigest(),
            "payload_ids": [integer(payload) for payload in payloads], "metrics": metrics.tolist()}
            for index, (llr, payloads, metrics) in enumerate(trace)]
    checks = {"native_aggregates_match_timed": all(row["matches_timed_aggregate"] for row in recorded.values()),
              "same_native_accepted_map": recorded["reference"]["accepted"] == recorded["native_only"]["accepted"],
              "same_native_query_count": len(old) == len(fast), "same_native_llr_bytes": same_inputs,
              "same_native_ordered_payloads": same_payloads, "metric_tolerance": bool(metrics_ok)}
    return {"arms": recorded, "checks": checks, "differences": differences, "maximum_metric_error": maximum_error,
            "comparison_seconds": time.perf_counter() - started,
            "scope": "Separate instrumented native replay, excluded from receiver timing; byte comparisons use actual inputs."}


def semantic_signature(meta):
    result = {key: meta[key] for key in ("rollback_action", "rollback_batches", "base_bank", "bank", "cancellation_action",
                                       "complete", "complete_support")}
    result["native"] = {key: meta["native"][key] for key in NATIVE_KEYS}
    result["continuation"] = trace_signature(meta["continuation"])
    result["cache"] = {key: meta["continuation"][key] for key in ("actual_scl_calls", "cache_hits", "cache_entries", "cache_bytes")}
    result["completion"] = {key: meta["support_completion"][key] for key in
        ("method", "additions", "steps", "input_count", "output_count", "bank_size", "absent_bank_ids")}
    return result


def input_snapshot(scenario):
    model, code = scenario.model, scenario.model.code
    arrays = {"y": scenario.y, "positions": model.positions, "amplitudes": model.amplitudes,
              "mean": model.mean, "information": code.information, "frozen": code.frozen}
    return {"arrays": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()},
            "code_configuration": [code.payload_bits, code.n, code.crc_bits, code.list_size],
            "scenario_configuration": [scenario.k, scenario.noise_var, scenario.sampling, model.n, model.p, model.B]}


def freeze(out):
    if out.exists():
        raise FileExistsError("Choose a fresh output directory")
    design = json.loads((BASE / "design.json").read_text())
    checked = json.loads((BASE / "pre_main.json").read_text())
    sources = {name: digest(ROOT / name) for name in SOURCES}
    if not checked["passed"] or checked["design_sha256"] != digest(BASE / "design.json"):
        raise ValueError("Require independent preflight of this frozen design")
    if checked["source_sha256"] != sources or design["panel"]["seeds"] != SEEDS:
        raise ValueError("Sources or panel differ from preflight")
    if design["gates"]["pooled_geometry_inclusive_receiver_ratio_at_most"] != .95:
        raise ValueError("Receiver saving gate differs from frozen declaration")
    out.mkdir(parents=True)
    (out / "design.json").write_bytes((BASE / "design.json").read_bytes())
    for name in SOURCES:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                "design_sha256": digest(out / "design.json"), "pre_main_sha256": digest(BASE / "pre_main.json")}
    write_json(out / "source_freeze.json", manifest)
    return manifest


def summarize(rows):
    if [(row["seed"], row["frame"]) for row in rows] != [(seed, 0) for seed in SEEDS]:
        raise ValueError("Require all16 predetermined frames in seed order")
    names = ("reference", "native_only")
    costs = {name: sum(row["receiver_with_geometry_seconds"][name] for row in rows) for name in names}
    totals = {name: {key: sum(row["metrics"][name][key] for row in rows) for key in
        ("matched_copies", "misses", "false_copies", "output_copies", "multiplicity_capable_oracle_bank_miss_floor")}
        for name in names}
    gates = {"all_complete": all(row["arms"][name]["meta"]["complete"] for row in rows for name in names),
             "identical_receiver_semantics": all(row["identical_receiver_semantics"] for row in rows),
             "original_inputs_unchanged": all(all(row["original_inputs_unchanged"].values()) for row in rows),
             "native_replay_and_metric_checks": all(all(row["native_audit"]["checks"].values()) for row in rows),
             "five_percent_complete_receiver_saving": costs["native_only"] <= .95 * costs["reference"]}
    return {"totals": totals, "receiver_with_geometry_seconds": costs, "ratio": costs["native_only"] / costs["reference"],
            "gates": gates, "passed": all(gates.values()),
            "native_replay_seconds": sum(row["native_audit"]["arms"][name]["seconds"] for row in rows for name in names),
            "native_trace_comparison_seconds": sum(row["native_audit"]["comparison_seconds"] for row in rows),
            "maximum_native_metric_error": max(row["native_audit"]["maximum_metric_error"] for row in rows),
            "scope": "Fresh native-only computational comparison of two complete receivers; no new discovery method. "
                     "Optional terminal-completion false copies are present in both arms; no universal float-order proof."}


def main(out):
    started = time.perf_counter()
    manifest = freeze(out)
    rows = []
    for seed in SEEDS:
        tick = time.perf_counter()
        model, balance = geometry(seed)
        geometry_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        scenario, labels = make_frame(model, seed, 0)
        source_seconds = time.perf_counter() - tick
        code, original_model = scenario.model.code, scenario.model
        tick = time.perf_counter()
        before = input_snapshot(scenario)
        verification_seconds = time.perf_counter() - tick
        arms, unchanged = {}, {}
        order = ("reference", "native_only") if seed % 2 else ("native_only", "reference")
        for name in order:
            tick = time.perf_counter()
            counts, meta = reference(scenario, memoize=True, complete_support=True) if name == "reference" else native_only(scenario)
            wall = time.perf_counter() - tick
            arms[name] = {"action": sorted(counts.items()), "meta": meta, "wall_seconds": wall}
            tick = time.perf_counter()
            unchanged[name] = (scenario.model is original_model and scenario.model.code is code and input_snapshot(scenario) == before)
            verification_seconds += time.perf_counter() - tick
            print(json.dumps({"seed": seed, "stage": "timed_arm_complete", "arm": name, "seconds": wall}), flush=True)
        identical = (arms["reference"]["action"] == arms["native_only"]["action"] and
                     semantic_signature(arms["reference"]["meta"]) == semantic_signature(arms["native_only"]["meta"]))
        audit = compare_native(scenario, arms)
        metrics = {name: evaluate(Counter(dict(arm["action"])), arm["meta"]["bank"], labels, scenario.k)
                   for name, arm in arms.items()}
        row = {"seed": seed, "frame": 0, "labels": labels, "balance": balance, "order": order,
               "geometry_seconds": geometry_seconds, "source_seconds": source_seconds,
               "input_snapshot": before, "original_inputs_unchanged": unchanged, "verification_seconds": verification_seconds,
               "arms": arms, "metrics": metrics, "native_audit": audit, "identical_receiver_semantics": identical,
               "receiver_with_geometry_seconds": {name: geometry_seconds + arm["wall_seconds"] for name, arm in arms.items()}}
        write_json(out / f"frame_{seed}.json", row)
        rows.append(row)
        print(json.dumps({"seed": seed, "stage": "frame_complete", "identical": identical,
            "audit_checks": audit["checks"], "elapsed_seconds": time.perf_counter() - started}), flush=True)
    result = summarize(rows)
    if any(digest(ROOT / name) != value for name, value in manifest["sources"].items()):
        raise AssertionError("Executed source changed after freeze")
    result.update({"all_sources_unchanged": True, "elapsed_seconds": time.perf_counter() - started,
        "source_freeze_sha256": digest(out / "source_freeze.json"),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
