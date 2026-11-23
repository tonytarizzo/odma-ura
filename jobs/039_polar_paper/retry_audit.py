"""Read back the two frozen matched-query panels and reconcile with the incumbent end banks."""

from collections import Counter
import json
from pathlib import Path
import statistics

from retry_allocation import ROOT, BASE, ARMS, digest, write_json

RESULT = ROOT / "results/polar_dec35_exploration_2026-11-11/paper_position/retry_policy"


def read(path):
    return json.loads(path.read_bytes())


def audit(name, first):
    folder = RESULT/name
    summary = read(folder/"summary.json")
    bound = read(folder/"freeze.json")
    assert all(digest(ROOT/p) == h for p,h in bound.items())
    sums = {mode: Counter() for mode in ARMS}
    folded_wins, old_calls = [], 0
    totals = Counter()
    physical, scale, digests = [], [], {}
    for seed in range(first, first+128):
        p = folder/f"frame_{seed}.json"
        data = read(p)
        archive = (BASE/"main" if first == 7101 else BASE/"confirmation/complete_128")/p.name
        frame = read(archive)
        assert data["seed"] == frame["seed"] == seed
        digests[str(p.relative_to(ROOT))] = digest(p)
        if data.get("eligible") is not True:
            continue
        if first == 7101:
            arm = frame["arms"]["count_growth"]
            meta, initial, final = arm["meta"], frame["initial_bank"], set(arm["bank"])
        else:
            chain = frame["arms"]["B"]["meta"]
            meta, initial, final = chain["continuation"], chain["base_bank"], set(chain["bank"])
        truth = Counter(frame["labels"])
        before = [q for q in meta["queries"] if q["sweep"] < data["sweep"]]
        prior = set(initial)
        for q in before:
            prior.update(q["ids"])
        assert prior == set(data["bank_before"]) and len(before) == data["prefix_queries_verified"]
        current = [(q["pattern"], q["llr_sha256"]) for q in before]
        matched_archived = {(q["pattern"], q["llr_sha256"]):q["ids"] for q in meta["queries"]}
        seen = {p for p,_ in current}
        reference_ids = set()
        for pstr, q in data["calls"].items():
            p = int(pstr)
            pair = (p, q["llr_sha256"])
            assert pair not in current and all(w >> 87 == p for w in q["ids"])
            if pair in matched_archived:
                old_calls += 1
                assert q["ids"] == matched_archived[pair]
            reference_ids.update(q["ids"])
        totals["all_true_new_opportunities"] += len((reference_ids-prior) & truth.keys())
        for m in ARMS:
            arm = data["evaluations"][m]
            patterns = arm["patterns"]
            assert len(patterns) == len(set(patterns)) == 8
            assert [p for p in patterns if p not in seen] == data["explored"]
            assert sum(p in seen for p in patterns) == data["paid_retries"]
            ids = {w for p in patterns for w in data["calls"][str(p)]["ids"]} - prior
            assert set(arm["new_true"]) == (ids & truth.keys()) and set(arm["new_false"]) == (ids - truth.keys())
            output = Counter(dict(arm["action"]))
            assert output.keys() <= ids | prior and output.total() <= 100
            matched = (output & truth).total()
            assert arm["misses"] == 100-matched and arm["false_copies"] == output.total()-matched
            sums[m].update(true=len(arm["new_true"]), false_ids=len(arm["new_false"]), misses=arm["misses"],
                false_copies=arm["false_copies"], physical_scl_calls=8, acquired_against_eventual_bank=len(set(arm["new_true"])-final))
        a = set(data["evaluations"]["magnitude"]["new_true"])
        b = set(data["evaluations"]["folded"]["new_true"])
        if a != b:
            folded_wins.append(dict(seed=seed, h=data["h"], paid_retries=data["paid_retries"],
                                    gained=sorted(b-a), lost=sorted(a-b), gained_outside_final=sorted(b-a-final),
                                    lost_outside_final=sorted(a-b-final)))
        for d,n in data["displacements"].values():
            physical.append(d)
            scale.append(n)
        totals["different_folded_physical_ids"] += int(set(data["evaluations"]["physical"]["new_true"]) != b)
        totals["different_folded_variance_ids"] += int(set(data["evaluations"]["variance"]["new_true"]) != b)
        totals["different_folded_magnitude_selection"] += int(set(data["arms"]["folded"]) != set(data["arms"]["magnitude"]))
    for m in ARMS:
        for k,v in summary["totals"][m].items():
            assert sums[m][k] == v
    assert len(digests) == 128
    return dict(eligible=summary["eligible"], union_scl_calls=summary["union_scl_calls"],
                selected_calls_reproducing_archived_ordered_ids=old_calls, totals=sums, differences=folded_wins, opportunity=totals,
                median_candidate_physical_tv_upper=statistics.median(physical), median_candidate_variance_tv_upper=statistics.median(scale),
                physical_at_least_099=sum(x>=.99 for x in physical), changed_candidates=len(physical),
                hashes={**digests, str((folder/"summary.json").relative_to(ROOT)): digest(folder/"summary.json")})


if __name__ == "__main__":
    result = {name:audit(name, seed) for name, seed in (("development", 7101), ("replication", 7301))}
    result["script_sha256"] = digest(Path(__file__))
    write_json(RESULT/"audit.json", result)
    for name in ("development", "replication"):
        print(name,{k:v for k,v in result[name].items() if k!="hashes"})
