"""Independent archived replication of the simpler folded retry selector."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import retry_allocation as study


def main(out):
    started = time.perf_counter()
    out.mkdir(exist_ok=False, parents=True)
    paths = [study.BASE / "confirmation/complete_128" / f"frame_{s}.json" for s in range(7301, 7429)]
    audited = study.BASE / "confirmation/audit/outcomes.json"
    checks = json.loads(audited.read_bytes())
    assert checks["passed"]
    bind = {str(p.relative_to(study.ROOT)): study.digest(p) for p in paths}
    assert all(checks["input_sha256"][k] == v for k,v in bind.items())
    source = Path(__file__).resolve()
    for p in [source, source.with_suffix(".md"), Path(study.__file__), study.DESIGN, audited,
              study.ROOT/"framework/polar_full_pattern.py", study.ROOT/"framework/polar_count_feedback.py",
              study.ROOT/"baselines/polar.py", study.JOB37/"rollback_frames.py"]:
        bind[str(p.relative_to(study.ROOT))] = study.digest(p)
    study.write_json(out/"freeze.json", bind)
    assert study.finite() == 64
    records = []
    for p in paths:
        frame = json.loads(p.read_bytes())
        meta = frame["arms"]["B"]["meta"]
        row = dict(seed=frame["seed"], input_snapshot=frame["input_snapshot"], labels=frame["labels"],
                   initial_bank=meta["base_bank"], arms={"count_growth": {"meta": meta["continuation"]}})
        state, data = study.shared_state(row)
        if state is not None:
            data.update(study.evaluate_state(state))
        data["seed"] = row["seed"]
        study.write_json(out/p.name, data)
        records.append(data)
        if len(records)%16 == 0:
            print(len(records), flush=True)
    summary = study.summarize(records)
    summary["selection_note"] = "The discovery deltas and gate inside paired_discovery belong to the first phase physical statistic."
    cases = [r for r in records if r.get("eligible")]
    diff = [len(r["evaluations"]["folded"]["new_true"])-len(r["evaluations"]["magnitude"]["new_true"]) for r in cases]
    better = sum(d > 0 for d in diff)
    copies = summary["totals"]["magnitude"]["misses"] - summary["totals"]["folded"]["misses"]
    false_delta = summary["totals"]["folded"]["false_copies"] - summary["totals"]["magnitude"]["false_copies"]
    net = sum(diff)
    leave = net-max([0]+diff)
    report = dict(net=net, wins=better, losses=sum(d < 0 for d in diff), ties=diff.count(0), leave_best=leave,
                  copies_improved=copies, false_copy_increase=false_delta, seed_differences={str(r["seed"]):d for r,d in zip(cases,diff) if d})
    summary["primary_folded_vs_magnitude"] = report
    summary["primary_gate"] = net >= 2 and better >= 2 and leave > 0 and copies >= 1 and false_delta <= copies
    summary["wall_seconds"] = time.perf_counter()-started
    assert all(study.digest(study.ROOT/k) == v for k,v in bind.items())
    study.write_json(out/"summary.json", summary)
    print(report, summary["primary_gate"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
