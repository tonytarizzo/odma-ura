"""Common-state matched SCL retry allocation after changed physical cancellations."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
JOB37 = ROOT / "jobs/037_polar_tractable"
sys.path[:0] = [str(ROOT), str(JOB37)]
from framework.polar_count_cavity import action_hash
from framework.polar_count_feedback import support_completion
from framework.polar_full_pattern import CountBank
from diagnose_full_pattern import digest, evaluate, write_json
from diagnose_native_rate_zero import input_snapshot
from rollback_frames import geometry, make_frame

BASE = ROOT / "results/polar_dec35_exploration_2026-11-11/count_growth"
DESIGN = Path(__file__).with_name("retry_design.md")
ARMS = ("magnitude", "folded", "physical", "variance")


def logcosh(x):
    return np.logaddexp(x, -x) - math.log(2.)


def log_displacement(a, b):
    if np.array_equal(a, b):
        return -np.inf
    log_rho = (logcosh((a+b)/4) - (logcosh(a/2)+logcosh(b/2))/2).sum()
    if log_rho > 1e-10:
        raise ArithmeticError("Positive log affinity")
    distance_squared = -np.expm1(2*min(float(log_rho), 0.))
    return .5*math.log(distance_squared) if distance_squared else -np.inf


def finite():
    rng = np.random.default_rng(67123)
    cases = 0
    for n in range(1, 9):
        signs = np.array(list(itertools.product((-1., 1.), repeat=n)))
        for _ in range(8):
            amp, old, now = np.exp(rng.normal(size=n)), rng.normal(size=n), rng.normal(size=n)
            v, w = np.exp(rng.normal(size=2))
            a, b = 2*amp*now/v, 2*amp*old/w
            np.testing.assert_allclose(a-b, 2*amp*(now-old)/v+2*amp*old*(1/v-1/w), atol=2e-13)
            def product(z):
                prob = np.exp(signs @ (z/2) - (logcosh(z/2)+math.log(2.)).sum())
                np.testing.assert_allclose(prob.sum(), 1., atol=2e-13)
                return prob
            aa, bb = product(a), product(b)
            rho = np.sqrt(aa*bb).sum()
            d = math.exp(log_displacement(a, b))
            np.testing.assert_allclose(d*d, 1-rho*rho, atol=2e-13)
            assert abs(aa-bb).sum()/2 <= d+1e-13
            assert log_displacement(a, a) == -np.inf
            cases += 1
    return cases


def raw_observation(s, action):
    canceled = np.zeros(s.model.n)
    for word, count in sorted(action.items()):
        p, amp = s.model.symbols(word)
        canceled[s.model.positions[p]] += count*amp
    return s.y-canceled


def candidates(s, raw, h, history, magnitude):
    model = s.model
    var = s.noise_var + max(0, h-1)/model.n
    retries, unchanged, log_distances = {}, set(), {}
    for p, records in history.items():
        pos, amp = model.positions[p], model.amplitudes[p]
        local = raw[pos]
        current = 2*amp*local/var
        if current.tobytes() in records["seen"]:
            unchanged.add(p)
            continue
        prev, oldvar = records["last"]
        last_matched_var = 2*amp*prev/var
        folded = float((logcosh(current/2)-amp**2/(2*var)).sum())
        change = log_displacement(current[1:], last_matched_var[1:])
        scale = log_displacement(last_matched_var[1:], (2*amp*prev/oldvar)[1:])
        retries[p] = (float(magnitude[p]), folded, folded+change, folded+scale)
        log_distances[p] = [float(math.exp(change)), float(math.exp(scale))]
    return retries, unchanged, log_distances


def pick(s, raw, h, history):
    started = time.perf_counter()
    mag = np.abs(raw[s.model.positions]).sum(axis=1)
    order = np.argsort(-mag, kind="stable").tolist()
    magnitude_seconds = time.perf_counter()-started
    started = time.perf_counter()
    scores, unchanged, distances = candidates(s, raw, h, history, mag)
    serial = [p for p in order if p not in unchanged][:8]
    new = [p for p in serial if p not in history]
    q = len(serial)-len(new)
    if not q or len(scores) <= q:
        return None
    arms = {}
    for j, name in enumerate(ARMS):
        # Only zero-displacement ties get the folded fallback. Other ties use pattern ID.
        ranks = sorted(scores, key=lambda p: (-scores[p][j], -scores[p][1] if scores[p][j] == -np.inf else 0., p))
        arms[name] = new + ranks[:q]
        assert len(set(arms[name])) == 8
    assert set(arms["magnitude"]) == set(serial)
    prev_bytes = sum(rows["last"][0].nbytes+8 for rows in history.values())
    return dict(arms=arms, rank=order[:105], paid_retries=q, ranks_checked=len(scores), exact_hits=len(unchanged),
                selector_seconds=time.perf_counter()-started, ranking_seconds=magnitude_seconds, previous_local_bytes=prev_bytes,
                displacements=distances, explored=new)


def receiver_possible(continuation):
    seen = set()
    for sweep in continuation["sweeps"]:
        q = [r for r in continuation["queries"] if r["sweep"] == sweep["sweep"]]
        if q[0]["pending"] and any(a != q[0]["pre_action_sha256"] for a in seen):
            return True
        seen.update(r["pre_action_sha256"] for r in q)
    return False


def shared_state(row):
    meta = row["arms"]["count_growth"]["meta"]
    if not receiver_possible(meta):
        return None, dict(regenerated=False, reason="no_positive_h_boundary_after_any_change")
    start = time.perf_counter()
    model, _ = geometry(row["seed"])
    scenario, labels = make_frame(model, row["seed"], 0)
    assert input_snapshot(scenario) == row["input_snapshot"] and labels == row["labels"]
    assert model.keys[0] == 0 and 0 not in model.keys[1:]
    setup_seconds = time.perf_counter()-start
    tick = time.perf_counter()
    action = Counter(dict(meta["initial_refit_action"]))
    ids = set(row["initial_bank"])
    transitions = {r["query"]: r for r in meta["count_action_path"]}
    raw = raw_observation(scenario, action)
    history = defaultdict(lambda: {"seen": set()})
    verified = 0
    for i, r in enumerate(meta["queries"]):
        h = scenario.k-action.total()
        assert i == r["query"] and r["pending"] == h and action_hash(action) == r["pre_action_sha256"]
        if i and r["sweep"] != meta["queries"][i-1]["sweep"] and h:
            choice = pick(scenario, raw, h, history)
            if choice is not None:
                assert choice["rank"] == meta["sweeps"][r["sweep"]]["ranked_patterns"]
                return (scenario, labels, action, ids, raw, choice), dict(regenerated=True, eligible=True, sweep=r["sweep"],
                    h=h, pre_action_sha256=r["pre_action_sha256"], prefix_queries_verified=verified, setup_seconds=setup_seconds,
                    prefix_reconstruction_seconds=time.perf_counter()-tick)
        p = r["pattern"]
        local = raw[model.positions[p]]
        var = scenario.noise_var + max(0, h-1)/model.n
        llr = 2*model.amplitudes[p]*local/var
        key = llr.tobytes()
        assert var == r["variance"] and hashlib.sha256(key).hexdigest() == r["llr_sha256"]
        history[p]["seen"].add(key)
        history[p]["last"] = (local, var)
        assert set(r["new_ids"]) == set(r["ids"])-ids
        ids.update(r["ids"])
        if i in transitions:
            move = transitions[i]
            assert dict(move["before"]) == action
            action = Counter(dict(move["after"]))
            raw = raw_observation(scenario, action)
        verified += 1
    return None, dict(regenerated=True, reason="no_eligible_boundary", prefix_queries_verified=verified,
                      setup_seconds=setup_seconds, prefix_reconstruction_seconds=time.perf_counter()-tick)


def evaluate_state(state):
    scenario, labels, action, ids, raw, choice = state
    model = scenario.model
    var = scenario.noise_var+max(0, scenario.k-action.total()-1)/model.n
    queries = {}
    for p in sorted(set().union(*choice["arms"].values())):
        start = time.perf_counter()
        llr = 2*model.amplitudes[p]*raw[model.positions[p]]/var
        payloads, _ = model.code.decode_list(llr)
        words = [model.word(p, payload) for payload in payloads]
        assert all(model.split(w)[0] == p for w in words)
        queries[p] = dict(llr_sha256=hashlib.sha256(llr.tobytes()).hexdigest(), ids=words, seconds=time.perf_counter()-start)
    arms = {}
    for name, patterns in choice["arms"].items():
        start = time.perf_counter()
        obtained = {w for p in patterns for w in queries[p]["ids"]} - ids
        bank = CountBank(scenario, ids | obtained)
        fit, log = bank.refit(action)
        assert log["converged"]
        output, _ = support_completion(scenario, fit, bank.words)
        assert sum(output.values()) <= scenario.k and all(c > 0 for c in output.values())
        arms[name] = dict(patterns=patterns, new_true=sorted(obtained.intersection(labels)),
                          new_false=sorted(obtained.difference(labels)), action=sorted(output.items()),
                          fit_completion_seconds=time.perf_counter()-start, **evaluate(output, bank.words, labels, scenario.k))
    return dict(**choice, evaluations=arms, calls=queries, bank_before=sorted(ids), cancellation=sorted(action.items()))


def summarize(records):
    eligible = [r for r in records if r.get("eligible")]
    totals = {}
    for mode in ARMS:
        selected = [r["evaluations"][mode] for r in eligible]
        totals[mode] = dict(true=sum(len(a["new_true"]) for a in selected), false_ids=sum(len(a["new_false"]) for a in selected),
                            misses=sum(a["misses"] for a in selected), false_copies=sum(a["false_copies"] for a in selected),
                            physical_scl_calls=8*len(selected))
    deltas = {}
    for mode in ("magnitude", "folded", "variance"):
        diff = [len(r["evaluations"]["physical"]["new_true"])-len(r["evaluations"][mode]["new_true"]) for r in eligible]
        differences = sum(set(r["arms"]["physical"]) != set(r["arms"][mode]) for r in eligible)
        win, lose = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
        deltas[mode] = dict(net=sum(diff), wins=win, losses=lose, ties=len(diff)-win-lose,
            leave_best=sum(diff)-max([0]+diff), different_frames=differences,
            one_sided_sign_p=binomtest(win, win+lose, .5, alternative="greater").pvalue if win+lose else 1.)
    gate = (all(deltas[m]["net"] >= 2 and deltas[m]["wins"] >= 2 and deltas[m]["leave_best"] > 0
                for m in ("magnitude", "folded")) and deltas["variance"]["net"] > 0 and
            totals["physical"]["misses"] < totals["magnitude"]["misses"] and totals["physical"]["misses"] <= totals["folded"]["misses"])
    return dict(frames=len(records), eligible=len(eligible), qualifying_seeds=[r["seed"] for r in eligible],
                screening=Counter(r.get("reason", "eligible") for r in records), totals=totals, paired_discovery=deltas, gate=gate,
                budget_per_arm=8*len(eligible), retries_per_arm=sum(r["paid_retries"] for r in eligible),
                exploration_per_arm=sum(8-r["paid_retries"] for r in eligible), union_scl_calls=sum(len(r["calls"]) for r in eligible),
                prior_llrs_verified=sum(r.get("prefix_queries_verified", 0) for r in records),
                setup_seconds=sum(r.get("setup_seconds", 0.) for r in records),
                prefix_reconstruction_seconds=sum(r.get("prefix_reconstruction_seconds", 0.) for r in records),
                selectors_seconds=sum(r["selector_seconds"] for r in eligible), magnitude_seconds=sum(r["ranking_seconds"] for r in eligible),
                maximum_extra_local_bytes=max((r["previous_local_bytes"] for r in eligible), default=0),
                union_query_seconds=sum(q["seconds"] for r in eligible for q in r["calls"].values()))


def main(out):
    start = time.perf_counter()
    out.mkdir(exist_ok=False, parents=True)
    paths = [BASE / "main" / f"frame_{seed}.json" for seed in range(7101, 7229)]
    audit = json.loads((BASE / "audit/outcomes.json").read_bytes())
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert audit["passed"] and all(audit["input_sha256"][k] == v for k,v in hashes.items())
    for p in (Path(__file__), DESIGN, JOB37 / "rollback_frames.py", ROOT / "framework/polar_full_pattern.py",
              ROOT / "framework/polar_count_feedback.py", ROOT / "baselines/polar.py", BASE / "audit/outcomes.json"):
        hashes[str(p.resolve().relative_to(ROOT))] = digest(p)
    write_json(out / "freeze.json", hashes)
    validation = finite()
    records = []
    for p in paths:
        row = json.loads(p.read_bytes())
        state, data = shared_state(row)
        if state is not None:
            data.update(evaluate_state(state))
        data["seed"] = row["seed"]
        write_json(out / p.name, data)
        records.append(data)
        if state is not None or len(records)%16 == 0:
            print(len(records), row["seed"], data.get("h"), data.get("paid_retries"), flush=True)
    result = summarize(records)
    result.update(finite=validation, wall_seconds=time.perf_counter()-start)
    assert all(digest(ROOT/k) == v for k,v in hashes.items())
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.out is None:
        print(finite())
    else:
        main(args.out)
