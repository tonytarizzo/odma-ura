"""Independently reconcile complete trial decisions, changed queues and realized work."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def load(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def packed_action_hash(action):
    entries = sorted((int(w),int(n)) for w,n in action if n)
    return hashlib.sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()


def audit(data):
    if not __debug__:
        raise RuntimeError("Audit requires assertions")
    frozen = load(data/"freeze.json")
    checks = frozen["source_sha256"]
    assert frozen["seeds"] == list(range(88001,88033))
    assert all(sha(ROOT/p) == v for p,v in checks.items())
    summary = load(data/"summary.json")
    binds, metrics, timing, growth = {}, {"base":Counter(),"direction":Counter()}, {}, []
    copy_delta, changed = [], []
    enabled, capture_total, call_diff = [], Counter(), []
    changed_banks, changed_actions, safety = [], [], Counter()
    for seed in frozen["seeds"]:
        path = data/f"frame_{seed}.json"
        row = load(path)
        binds[path.name] = sha(path)
        assert row["seed"] == seed and row["enabled"] and row["source"]["scenario_configuration"] == [100,1/(200*10**.05),"iid",30000,8192,100]
        truth = Counter(row["labels"])
        actions = {}
        for name in ("base","direction"):
            arm, queries = row["arms"][name], row["traces"][name]
            count, bank = Counter(dict(arm["action"])), set(arm["bank"])
            fitted = Counter(dict(arm["cancellation"]))
            assert count.total() <= 100 and all(isinstance(w,int) and w in bank and int(c) == c > 0 for w,c in count.items())
            assert all(fitted[w] <= count[w] for w in fitted)
            assert set(arm["base_bank"]) | {w for _,_,_,ids,_ in queries for w in ids} == bank
            hits = sum(q[-1] for q in queries)
            assert len(queries)-hits == arm["actual_scl_calls"] and hits == arm["cache_hits"]
            assert len(queries) == 105*arm["sweeps"]
            for start in range(0,len(queries),105):
                assert len({q[0] for q in queries[start:start+105]}) == 105
            matched, false = (count & truth).total(), (count - truth).total()
            misses = (truth-count).total()
            floor = sum(n for w,n in truth.items() if w not in bank)
            assert (matched,false,misses,floor) == (arm["matched"],arm["false_copies"],arm["misses"],arm["bank_floor"])
            metrics[name].update(matched=matched,misses=misses,false_copies=false,bank_floor=floor,actual_scl_calls=arm["actual_scl_calls"],
                cache_hits=hits,wall=arm["wall"],with_geometry=arm["wall"]+row["geometry_seconds"])
            actions[name]=count
        d = row["direction"]
        before, after = row["traces"]["base"], row["traces"]["direction"]
        distinct_state = [i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
        if d["selection"]:
            v = d["selection"]
            enabled.append(seed)
            ix = v["at_query"]
            prefix = before[:ix]
            pseen = {q[0] for q in prefix}
            assert ix % 105 == 0 and len(v["original"]) == len(set(v["revised"])) == len(v["revised"]) == 105
            assert [q[0] for q in before[ix:ix+105]] == v["original"] and [q[0] for q in after[ix:ix+105]] == v["revised"]
            assert [p for p in v["original"] if p not in pseen] == [p for p in v["revised"] if p not in pseen] == v["selected_exploration"]
        if distinct_state:
            v = d["selection"]
            first = distinct_state[0]
            assert v["changed"] and before[:first] == after[:first] and before[first][0] != after[first][0]
            assert before[first][1] == after[first][1] and row["first_divergence"]["first_different_query"] == first
            changed.append(seed)
            start = v["at_query"]
            left, right = before[start:start+105], after[start:start+105]
            by_pattern = {q[0]:q[1:] for q in left}
            same = sum(by_pattern.get(q[0]) == q[1:] for q in right)
            safety["identical_pattern_queries_in_changed_blocks"] += same
            safety["constant_fit_and_identical_changed_blocks"] += int(same==105 and len({q[1] for q in left+right})==1)
        else:
            assert before == after and actions["base"] == actions["direction"]
        true_ids = set(truth)
        b, f = (set(row["arms"][name]["bank"]) for name in ("base","direction"))
        gained, lost = sorted((f-b) & true_ids), sorted((b-f) & true_ids)
        diff = row["arms"]["base"]["misses"]-row["arms"]["direction"]["misses"]
        if f != b:
            changed_banks.append(seed)
        if actions["base"] != actions["direction"]:
            changed_actions.append(seed)
        if d["selection"]:
            selection = d["selection"]
            safety["displaced_105_patterns"] += len(set(selection["original"])-set(selection["revised"]))
            safety["reordered_105_slots"] += sum(x != y for x,y in zip(selection["original"],selection["revised"]))
            safety["eligible_pending_sum"] += selection["h"]
        if gained or lost or diff:
            growth.append(dict(seed=seed,misses_saved=diff,gained=gained,lost=lost))
        copy_delta.append(diff)
        call_diff.append(row["arms"]["direction"]["actual_scl_calls"]-row["arms"]["base"]["actual_scl_calls"])
        timing[seed] = row["arms"]["direction"]["wall"]-row["arms"]["base"]["wall"]
        capture_total.update({k:d[k] for k in ("captured_queries","history_updates","selection_seconds","capture_readout_seconds","capture_ops_seconds")})
    for name,nums in metrics.items():
        for met,num in nums.items():
            assert abs(summary["totals"][name][met]-num) < 1e-7
    assert sum(copy_delta) == summary["net"] and sum(copy_delta)-max([0]+copy_delta) == summary["leave_best"]
    assert changed == summary["changed_queue"] and enabled == summary["eligible"]
    assert len(binds) == 32 and sum(a<0 for a in timing.values()) == summary["faster_frames"]
    assert all(abs(capture_total[k]-v)<1e-7 for k,v in summary["counters"].items())
    smoke=load(data/"archived_smoke.json")
    assert smoke["seed"] == 6506 and not smoke["enabled"] and smoke["traces"]["base"] == smoke["traces"]["direction"]
    report=dict(passed=True, source_verified=len(checks), bound_frames=binds, summary_sha256=sha(data/"summary.json"),
                smoke_sha256=sha(data/"archived_smoke.json"), source_sha256=sha(Path(__file__)),
                smoke_logical_calls=len(smoke["traces"]["base"]), metrics=metrics,growth=growth,
                changed_queue=changed,eligible=enabled,history=capture_total,net_physical_continuation_calls=sum(call_diff),
                changed_banks=changed_banks, changed_actions=changed_actions, allocation=safety,
                median_direction_minus_ordinary_by_first_arm={k:statistics.median(v for seed,v in timing.items() if seed%2==parity)
                    for k,parity in (("ordinary_first",1),("direction_first",0))},
                time_median_arm_difference=statistics.median(timing.values()), elapsed_arm_difference_by_seed=timing)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path,help="Complete output directory from retry_receiver_trial.py --out")
    args = parser.parse_args()
    print(json.dumps(audit(args.directory),indent=2,allow_nan=False))
