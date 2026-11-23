"""Finite, archived smoke and prospective paired complete tests for one directional retry block."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import binomtest
from retry_receiver import ROOT, CaptureCode, priority, requeue, run
from baselines.polar import PolarCode
from framework.polar_decoders import run as registered
from diagnose_full_pattern import digest, write_json
from diagnose_native_rate_zero import input_snapshot
from rollback_frames import geometry, make_frame

JOB = Path(__file__).resolve().parent
OPTIONS = dict(native_repetition=True,memoize=True,complete_support=True,progress_extra_sweep=True,count_growth=True)


def finite():
    rng = np.random.default_rng(181198)
    cases = 0
    for b,n,crc,width in ((3,8,4,8),(7,16,4,16),(87,512,16,128)):
        original = PolarCode(b,n,crc,width)
        alt = CaptureCode(original)
        for llr in (np.zeros(n),np.ones(n),rng.normal(size=n),rng.normal(size=n)*4):
            obs = original.decode_list(llr)
            data = alt.decode_list(llr)
            np.testing.assert_array_equal(data[0],obs[0])
            np.testing.assert_array_equal(data[1],obs[1])
            assert np.unpackbits(alt.packed,axis=1,count=b).shape == (width,b)
            amp, z = rng.uniform(.2,.7,n), rng.normal(size=n)
            anchor = (np.logaddexp(amp*z,-amp*z)-np.log(2.)-amp**2/2).sum()
            np.testing.assert_allclose(priority(alt,alt.packed,amp,z,z,1.),anchor,atol=3e-13)
            cases += 1
    original = list(range(20))
    known = set((1,2,4,6,7,10,12,14,18,21,22,23))
    unchanged = {2,6}
    queue = requeue(original, known, unchanged, sorted(known-unchanged,reverse=True))
    assert [p for p in original if p not in known or p in unchanged] == [p for p in queue if p not in known or p in unchanged]
    assert set(queue)-set(original) == {21,22,23} and len(queue) == len(set(queue)) == 20
    assert queue[:10] == [0,23,2,3,22,5,6,21,8,9] and requeue(list(range(5)),known,unchanged,None) is None
    return cases+1


def normal(meta):
    return [[q["pattern"],q["pre_action_sha256"],q["llr_sha256"],q["ids"],q["cache_hit"]] for q in meta["continuation"]["queries"]]


def value(s,labels,action,meta,wall):
    truth, count = Counter(labels), Counter(action)
    assert meta["complete"] and count.total() <= s.k and all(c > 0 and type(c) is int for c in count.values())
    bank = set(meta["bank"])
    assert not set(count)-bank
    matched = (truth & count).total()
    return dict(action=sorted(count.items()), matched=matched, misses=s.k-matched, false_copies=count.total()-matched,
        bank=sorted(bank), bank_floor=sum(n for w,n in truth.items() if w not in bank), wall=wall,
        native_calls=meta["native"]["calls"], native_rounds=meta["native"]["rounds"], rollback_action=meta["rollback_action"],
        base_bank=meta["base_bank"], cancellation=meta["cancellation_action"], actual_scl_calls=meta["continuation"]["actual_scl_calls"],
        cache_hits=meta["continuation"]["cache_hits"], stop=meta["continuation"]["stop"], sweeps=len(meta["continuation"]["sweeps"]))


def frame(seed, enabled):
    tick = time.perf_counter()
    m,_ = geometry(seed)
    geometry_seconds = time.perf_counter()-tick
    s,labels = make_frame(m,seed,0)
    source = input_snapshot(s)
    observations, trails, data = {}, {}, None
    for mode in (("base","direction") if seed%2 else ("direction","base")):
        tick = time.perf_counter()
        if mode == "base":
            output, meta = registered("DEC-35",s,**OPTIONS)
            wall = time.perf_counter()-tick
            trails[mode] = normal(meta)
        else:
            output, meta = run(s,policy=enabled)
            wall = time.perf_counter()-tick
            trails[mode] = meta["continuation"]["trace"]
            data = {key:v for key,v in meta["continuation"].items() if key not in ("trace","sweeps")}
        observations[mode] = value(s,labels,output,meta,wall)
        assert input_snapshot(s) == source
    a,b = observations["base"],observations["direction"]
    for k in ("native_calls","native_rounds","rollback_action","base_bank"):
        assert a[k] == b[k]
    audit = {}
    changed = data["selection"] and data["selection"]["changed"]
    if changed:
        decision = data["selection"]
        where = decision["at_query"]+next(i for i,(a,b) in enumerate(zip(decision["original"],decision["revised"])) if a!=b)
        assert trails["base"][:where] == trails["direction"][:where]
        assert trails["base"][where][0] != trails["direction"][where][0]
        assert trails["base"][where][1] == trails["direction"][where][1]
        assert [q[0] for q in trails["base"][decision["at_query"]:decision["at_query"]+len(decision["original"])]] == decision["original"]
        audit["first_different_query"] = where
    else:
        assert trails["direction"] == trails["base"]
        for k in ("action","bank","cancellation","stop","sweeps","actual_scl_calls","cache_hits"):
            assert a[k] == b[k]
    return dict(seed=seed, source=source, labels=labels, geometry_seconds=geometry_seconds,arms=observations,
                direction=data, first_divergence=audit, traces=trails, enabled=enabled)


def aggregate(rows):
    a = [r["arms"]["base"] for r in rows]
    b = [r["arms"]["direction"] for r in rows]
    def counts(data):
        totals={k:sum(v[k] for v in data) for k in ("misses","false_copies","matched","bank_floor","actual_scl_calls","cache_hits","wall")}
        totals["with_geometry"] = totals["wall"]+sum(r["geometry_seconds"] for r in rows)
        return totals
    totals = {"base":counts(a),"direction":counts(b)}
    d = [u["misses"]-v["misses"] for u,v in zip(a,b)]
    changed = [r["seed"] for r in rows if r["first_divergence"]]
    win, lose = sum(i>0 for i in d), sum(i<0 for i in d)
    factor = totals["direction"]["with_geometry"]/totals["base"]["with_geometry"]
    false_diff = totals["direction"]["false_copies"]-totals["base"]["false_copies"]
    same = all(u["action"]==v["action"] for u,v in zip(a,b))
    faster = sum(v["wall"]<u["wall"] for u,v in zip(a,b))
    return dict(frames=len(rows),total_transmitted=100*len(rows), totals=totals, changed_queue=changed,
        eligible=[r["seed"] for r in rows if r["direction"]["selection"]], wins=win, losses=lose, ties=d.count(0),
        net=sum(d), leave_best=sum(d)-max([0]+d), one_sided_sign_p=binomtest(win,win+lose,.5,alternative="greater").pvalue if win+lose else 1.,
        same_outputs=same, time_ratio=factor, faster_frames=faster,
        accuracy_gate=sum(d)>=3 and win>=2 and sum(d)-max([0]+d)>0 and false_diff<=sum(d) and factor<=1.25,
        efficiency_gate=same and factor<=.97 and faster>=20,
        counters={k:sum(r["direction"][k] for r in rows) for k in ("captured_queries","history_updates","capture_readout_seconds",
             "capture_ops_seconds","selection_seconds")}, maximum_extra_stored_data_bytes=max(r["direction"]["extra_stored_data_bytes"] for r in rows))


def main(out):
    started = time.perf_counter()
    out.mkdir(exist_ok=False,parents=True)
    source_paths = [JOB/"retry_receiver_design.md",JOB/"retry_receiver.py",Path(__file__).resolve(),
            ROOT/"framework/polar_decoders.py",ROOT/"framework/polar_count_feedback.py",ROOT/"framework/polar_count_growth.py",
            ROOT/"framework/polar_repetition_scl.py",ROOT/"framework/polar_count_cavity.py",ROOT/"framework/polar_full_pattern.py",
            ROOT/"baselines/polar.py",ROOT/"jobs/037_polar_tractable/rollback_frames.py",ROOT/"jobs/037_polar_tractable/rollback_progressive.py"]
    bindings={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    write_json(out/"freeze.json",dict(source_sha256=bindings,seeds=list(range(88001,88033)), finite=finite()))
    smoke=frame(6506,False)
    assert not smoke["first_divergence"]
    write_json(out/"archived_smoke.json",smoke)
    print("cold 6506 identity passed",flush=True)
    rows=[]
    for seed in range(88001,88033):
        r=frame(seed,True)
        write_json(out/f"frame_{seed}.json",r)
        rows.append(r)
        print(len(rows),seed,r["arms"]["base"]["misses"],r["arms"]["direction"]["misses"],bool(r["first_divergence"]),
              round(time.perf_counter()-started,1),flush=True)
    result=aggregate(rows)
    result["wall_seconds_including_smoke"]=time.perf_counter()-started
    assert all(digest(ROOT/p) == val for p,val in bindings.items())
    write_json(out/"summary.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.out is None:
        print(finite())
    else:
        main(args.out)
