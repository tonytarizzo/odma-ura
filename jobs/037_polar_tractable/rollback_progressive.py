"""Equal-work wide polar-list rescue with and without reconsidering SIC cancellations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rollback_frames import Work, geometry, make_frame, native
from baselines.polar import PolarCode
from rollback_rescue import count_action, polar_list, prepare


def progressive(s, accepted, valid_ids, mode, *, code_factory=PolarCode):
    if mode not in ("static", "progressive", "gated", "ungated"):
        raise ValueError(mode)
    start = time.perf_counter()
    model, initial = s.model, Counter(accepted)
    prep = prepare(s, accepted)
    cache, action, found, tested = dict(prep[1]), initial.copy(), set(), {}
    code = code_factory(model.b, model.code.n, model.code.crc_bits, 512)
    if (not isinstance(code, PolarCode) or (code.payload_bits, code.n, code.crc_bits, code.list_size)
            != (model.b, model.code.n, model.code.crc_bits, 512)
            or not np.array_equal(code.information, model.code.information)
            or not np.array_equal(code.frozen, model.code.frozen)):
        raise ValueError("Query code does not match the scenario encoder and fixed SCL512 budget")
    batches, stalled = [], True
    for step in range(4):
        current = initial if mode == "static" else action
        residual = prep[0].copy()
        for w in (current.keys() | initial.keys()):
            difference = current[w] - initial[w]
            if difference:
                if w not in cache:
                    p, values = model.symbols(w)
                    cache[w] = model.positions[p], values
                idx, values = cache[w]
                residual[idx] -= difference * values
        released = []
        pending = s.k - current.total()
        if mode in ("gated", "ungated") and stalled and step in (1, 3):
            corr = {w: float(residual[cache[w][0]] @ cache[w][1]) for w in current}
            cut = -3 * np.sqrt(s.noise_var + pending/model.n)
            eligible = [w for w in current if mode == "ungated" or corr[w] < cut]
            released = sorted(eligible, key=lambda w: (corr[w], w))[:1 if step == 1 else 2]
            for w in released:
                idx, values = cache[w]
                residual[idx] += values
        effective = current - Counter(released)
        key = tuple(sorted(effective.items()))
        visited = tested.setdefault(key, set())
        query_rank = Work(model).patterns(residual, min(model.p, len(visited) + 8))
        selected = [int(p) for p in query_rank if p not in visited][:8]
        if len(selected) != 8:
            raise ValueError("This diagnostic requires a bank with at least 32 patterns")
        visited.update(selected)
        variance = s.noise_var + max(0, s.k-effective.total()-1)/model.n
        new = {w for p in selected for w in polar_list(model, code, p, residual, variance)}
        fresh = new - found
        found.update(new)
        previous = action
        action, readout = count_action(s, prep, valid_ids, found)
        stalled = action == previous
        batches.append(dict(released=released, patterns=selected, new_ids=sorted(fresh), changes=(action-previous).total()
                            +(previous-action).total(), readout=readout))
    return action, found, batches, time.perf_counter()-start


def run(scenario, *, mode="progressive"):
    accepted, base = native(scenario)
    action, _, batches, seconds = progressive(scenario, list(accepted), base["valid_ids"], mode)
    return dict(action), {"native": base, "batches": batches, "extra_seconds": seconds, "extra_calls": 32}


def main(out, seed, frames):
    records = [json.loads(line) for line in (out / f"native_{seed}.jsonl").read_text().splitlines()]
    if [r["frame"] for r in records] != list(range(frames)):
        raise ValueError("Require a complete native panel in frame order")
    model, _ = geometry(seed)
    modes = ["static", "progressive", "gated", "ungated"]
    with (out / f"progressive_{seed}.jsonl").open("x") as handle:
        for row in records:
            frame = row["frame"]
            s, labels = make_frame(model, seed, frame)
            assert labels == row["truth"] and hashlib.sha256(s.y.tobytes()).hexdigest() == row["y_sha256"]
            truth, old = Counter(labels), set(row["valid_ids"])
            entry = {"seed": seed, "frame": frame, "y_sha256": row["y_sha256"], "native_seconds": row["seconds"],
                     "native_calls": row["calls"], "native_misses": s.k-(truth & Counter(row["accepted"])).total(), "arms": {}}
            offset = frame % 4
            for mode in modes[offset:] + modes[:offset]:
                action, pool, batches, seconds = progressive(s, row["accepted"], row["valid_ids"], mode)
                hit = (truth & action).total()
                entry["arms"][mode] = dict(misses=s.k-hit, false_copies=action.total()-hit, output=sorted(action.items()),
                    coverage=sum(n for w,n in truth.items() if w in pool|old), new_false_ids=len(pool-old-truth.keys()),
                    new_correct=sorted((pool-old) & truth.keys()), extra_seconds=seconds, batches=batches)
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
    print(seed, frames, "progressive frames complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    main(args.out, args.seed, args.frames)
