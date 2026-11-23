"""Directional transport over a previous polar list, with explicit recovery and attribution costs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import retry_allocation as base
from baselines.polar import PolarCode, append_crc


DIRECTORY = base.BASE.parent/"progress_continuation"
MODES = ("magnitude", "folded", "legal", "inner")


def dots(signs, field):
    return np.einsum("ij,j->i", signs, field, optimize=False)


def transport(signs, old, difference, anchor):
    columns = np.flatnonzero(difference)
    old_score = dots(signs, old)
    updated = old_score + dots(signs[:, columns], difference[columns])
    return float(anchor + logsumexp(updated)-logsumexp(old_score))


def finite():
    rng = np.random.default_rng(74901)
    n = 8
    code = PolarCode(3, n, 4, 8)
    messages = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.uint8)
    atoms = 1-2*code.encode(messages).astype(np.int8)
    all_signs = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=np.int8)
    for i in range(40):
        old = rng.normal(size=n)
        difference = rng.normal(size=n)
        difference[rng.random(n)<.5] = 0
        end = old+difference
        amp = rng.uniform(.1, .75, size=n)
        offset = float(sum(amp**2)/2)
        for signs in (atoms, all_signs):
            old_bf = float(logsumexp(dots(signs, old))-math.log(len(signs))-offset)
            conditional = np.exp(dots(signs, old)-logsumexp(dots(signs, old)))
            independent = old_bf+np.log(conditional @ np.exp(dots(signs, difference)))
            expected = float(logsumexp(dots(signs, end))-math.log(len(signs))-offset)
            np.testing.assert_allclose([transport(signs, old, difference, old_bf), independent], expected, atol=1e-12)
        old_folded = base.logcosh(old).sum()-offset
        np.testing.assert_allclose(transport(all_signs, old, difference, old_folded), base.logcosh(end).sum()-offset, atol=1e-12)
    return 120


def choose(s, raw, h, history, first):
    model = s.model
    code = model.code
    rawcode = PolarCode(model.b+code.crc_bits, code.n, 0, 128)
    assert np.array_equal(code.information, rawcode.information) and np.array_equal(code.frozen, rawcode.frozen)
    var = s.noise_var+max(0, h-1)/model.n
    costs = Counter()
    legal, inner, observer = {}, {}, {}
    for pstr in first["displacements"]:
        p = int(pstr)
        pos, amp = model.positions[p], model.amplitudes[p]
        previous, oldvar = history[p]["last"]
        original = 2*amp*previous/oldvar
        begin = time.perf_counter()
        paths, _ = rawcode.decode_list(original)
        costs["old_full_scl_replay_seconds"] += time.perf_counter()-begin
        begin = time.perf_counter()
        payload = paths[:, :model.b]
        valid = np.all(append_crc(payload, code.crc_bits) == paths, axis=1)
        assert [model.word(p, word) for word in payload[valid]] == history[p]["last_ids"]
        assert len(paths) == len(np.unique(paths, axis=0)) == 128
        costs["old_valid_subsets_verified"] += 1
        costs["validation_seconds"] += time.perf_counter()-begin
        begin = time.perf_counter()
        field = amp*previous/var
        changed = amp*(raw[pos]-previous)/var
        anchor = float((base.logcosh(field)-amp**2/(2*var)).sum())
        costs["common_scores_seconds"] += time.perf_counter()-begin
        begin = time.perf_counter()
        distinct = np.unique(payload, axis=0)
        signs = 1-2*code.encode(distinct).astype(np.int8)
        legal[p] = transport(signs, field, changed, anchor)
        costs["legal_processing_seconds"] += time.perf_counter()-begin
        begin = time.perf_counter()
        relaxed_signs = 1-2*rawcode.encode(paths).astype(np.int8)
        inner[p] = transport(relaxed_signs, field, changed, anchor)
        costs["inner_processing_seconds"] += time.perf_counter()-begin
        costs["corrected_waveforms"] += len(distinct)
        costs["inner_waveforms"] += len(paths)
        costs["changed_output_coordinates"] += int(np.count_nonzero(changed))
        # Packed payload bytes allow observer membership without a truth-dependent shortlist.
        observer[p] = set(map(bytes, np.packbits(distinct, axis=1)))
    begin = time.perf_counter()
    choice = {k:v for k,v in first.items() if k not in ("arms", "displacements")}
    choice["arms"] = {k:first["arms"][k] for k in ("magnitude", "folded")}
    for name, scores in (("legal", legal), ("inner", inner)):
        choice["arms"][name] = first["explored"]+sorted(scores, key=lambda p:(-scores[p],p))[:first["paid_retries"]]
    costs["ordering_seconds"] = time.perf_counter()-begin
    costs["prior_packed_all_leaf_information_bytes_for_seen_patterns"] = len(history)*128*((rawcode.k+7)//8)
    return choice, observer, costs


def trajectory(frame, source):
    meta = frame["arms"]["conditional_fourth"]["meta"]
    if not base.receiver_possible(meta):
        return dict(reason="no_positive_h_boundary_after_any_change")
    begin = time.perf_counter()
    seed = frame["seed"]
    model, _ = base.geometry(seed)
    s, labels = base.make_frame(model, seed, 0)
    assert labels == source["labels"]
    checks = base.input_snapshot(s)["arrays"]
    assert all(checks[k] == v for k,v in frame["input_hashes"].items())
    setup = time.perf_counter()-begin
    action = Counter(dict(meta["initial_refit_action"]))
    ids = set(frame["initial_bank"])
    changes = {q["query"]:q for q in meta["count_action_path"]}
    raw = base.raw_observation(s, action)
    history = defaultdict(lambda: {"seen":set()})
    for i,q in enumerate(meta["queries"]):
        h = s.k-action.total()
        assert base.action_hash(action) == q["pre_action_sha256"] and q["pending"] == h
        if i and q["sweep"] != meta["queries"][i-1]["sweep"] and h:
            first = base.pick(s, raw, h, history)
            if first is not None:
                assert first["rank"] == meta["sweeps"][q["sweep"]]["ranked_patterns"]
                decision, pools, costs = choose(s, raw, h, history, first)
                outcome = base.evaluate_state((s, labels, action, ids, raw, decision))
                observer = {}
                fullbank = set(frame["arms"]["conditional_fourth"]["bank"])
                all_pool_missing = []
                for w in sorted(set(labels)-ids):
                    p, payload = model.split(w)
                    packed = (payload << (-model.b % 8)).to_bytes((model.b+7)//8, "big")
                    if p in pools and packed in pools[p]:
                        all_pool_missing.append(w)
                for mode in MODES:
                    new = set(outcome["evaluations"][mode]["new_true"])
                    observer[mode] = dict(old_reencoded=list(sorted(new & set(all_pool_missing))),
                                         eventual_bank_absent=sorted(new-fullbank),
                                         old_pool_unobserved_pattern=sorted(w for w in new if model.split(w)[0] not in pools))
                return dict(eligible=True, h=h, sweep=q["sweep"], prefix_queries_verified=i, setup_seconds=setup,
                            costs=costs, pool_missing_true_ids=all_pool_missing, observer=observer, **outcome)
        p = q["pattern"]
        local = raw[model.positions[p]]
        var = s.noise_var+max(0, h-1)/model.n
        key = (2*model.amplitudes[p]*local/var).tobytes()
        assert var == q["variance"] and hashlib.sha256(key).hexdigest() == q["llr_sha256"]
        history[p]["seen"].add(key)
        history[p].update(last=(local,var), last_ids=q["ids"])
        assert set(q["new_ids"]) == set(q["ids"])-ids
        ids.update(q["ids"])
        if i in changes:
            assert dict(changes[i]["before"]) == action
            action = Counter(dict(changes[i]["after"]))
            raw = base.raw_observation(s, action)
    return dict(reason="no_eligible_boundary", prefix_queries_verified=len(meta["queries"]), setup_seconds=setup)


def comparison(cases, candidate, original):
    differences = []
    novel, pool_external_gain_frames = [], []
    for r in cases:
        arm, control = (r["evaluations"][m] for m in (candidate, original))
        gained = set(arm["new_true"])-set(control["new_true"])
        diff = len(gained)-len(set(control["new_true"])-set(arm["new_true"]))
        differences.append(diff)
        if gained.intersection(r["observer"][candidate]["eventual_bank_absent"]):
            novel.append(r["seed"])
        if diff > 0 and gained-set(r["pool_missing_true_ids"])-set(r["observer"][candidate]["old_pool_unobserved_pattern"]):
            pool_external_gain_frames.append(r["seed"])
    return dict(net=sum(differences), wins=sum(x>0 for x in differences), losses=sum(x<0 for x in differences),
                ties=differences.count(0), leave_best=sum(differences)-max([0]+differences),
                found_outside_eventual=novel, gained_on_old_pool_miss=pool_external_gain_frames,
                action_matched_gain=sum(r["evaluations"][original]["misses"] - r["evaluations"][candidate]["misses"] for r in cases))


def main(out):
    started = time.perf_counter()
    out.mkdir(exist_ok=False, parents=True)
    audit = read(DIRECTORY/"audit/outcomes.json")
    assert audit["passed"]
    inputs = sorted((DIRECTORY/"development").glob("frame_*.json"))
    assert [int(p.stem.split("_")[1]) for p in inputs] == [s for first, n in ((6501,32),(6601,32),(6701,16),(6901,16)) for s in range(first, first+n)]
    bindings = {}
    for p in inputs:
        rp = str(p.relative_to(base.ROOT))
        bindings[rp] = base.digest(p)
        assert bindings[rp] == audit["input_sha256"][rp]
        old = read(p)["saved_source"]
        bindings[old] = base.digest(base.ROOT/old)
    for p in (Path(__file__), Path(__file__).with_suffix(".md"), Path(base.__file__), DIRECTORY/"audit/outcomes.json",
              base.ROOT/"baselines/polar.py", base.JOB37/"rollback_frames.py", base.ROOT/"framework/polar_full_pattern.py"):
        bindings[str(p.resolve().relative_to(base.ROOT))] = base.digest(p)
    base.write_json(out/"freeze.json", bindings)
    checks = finite()
    records = []
    for p in inputs:
        f = read(p)
        data = trajectory(f, read(base.ROOT/f["saved_source"]))
        data["seed"] = f["seed"]
        base.write_json(out/p.name, data)
        records.append(data)
        if len(records)%16 == 0:
            print(len(records),flush=True)
    cases = [r for r in records if r.get("eligible")]
    totals, contrasts = {}, {}
    for m in MODES:
        arms = [r["evaluations"][m] for r in cases]
        totals[m] = dict(new_true=sum(len(a["new_true"]) for a in arms), new_false=sum(len(a["new_false"]) for a in arms),
            misses=sum(a["misses"] for a in arms), false_copies=sum(a["false_copies"] for a in arms),
            query_seconds=sum(r["calls"][p]["seconds"] for r in cases for p in r["arms"][m]))
    costs = sum((Counter(r["costs"]) for r in cases), Counter())
    for m in ("legal","inner"):
        contrasts[m] = {control:comparison(cases, m, control) for control in MODES if control != m}
        d = contrasts[m]
        cost = costs[f"{m}_processing_seconds"]+costs["common_scores_seconds"]+costs["ordering_seconds"]
        contrasts[m]["gate"] = bool(all(d[c]["net"] >= 2 and d[c]["wins"] >= 2 and d[c]["leave_best"] > 0
            for c in ("folded","magnitude")) and d["folded"]["action_matched_gain"] >= 1 and d["magnitude"]["action_matched_gain"] >= 0
            and (d["folded"]["found_outside_eventual"] or len(d["folded"]["gained_on_old_pool_miss"]) >= 2)
            and cost <= totals[m]["query_seconds"])
        contrasts[m]["incremental_processing_seconds"] = cost
    assert all(base.digest(base.ROOT/k) == v for k,v in bindings.items())
    result = dict(finite=checks, frames=len(records), eligible=len(cases), totals=totals, costs=costs, contrasts=contrasts,
        prior_llrs_verified=sum(r.get("prefix_queries_verified",0) for r in records), retries=sum(r["paid_retries"] for r in cases),
        new_pattern_slots=sum(8-r["paid_retries"] for r in cases), distinct_new_scl_calls=sum(len(r["calls"]) for r in cases),
        old_pool_unbanked_truth=sum(len(r["pool_missing_true_ids"]) for r in cases),
        max_raw_leaf_storage=max((r["costs"]["prior_packed_all_leaf_information_bytes_for_seen_patterns"] for r in cases),default=0),
        wall_seconds=time.perf_counter()-started)
    base.write_json(out/"summary.json", result)
    print(result,flush=True)


def read(p):
    return json.loads(p.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        print(finite())
    else:
        main(args.out)
