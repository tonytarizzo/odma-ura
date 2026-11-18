"""Paired native-style terminal SIC snapshots for the B100 rollback experiment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.polar_energy_geometry import balanced_energy
from framework.polar_tractable import PolarModel, PolarScenario, Work, integer


def geometry(seed):
    block = ODMAPolarBaseline(100, 30000, seed, prefix_bits=13, code_length=512, crc_bits=16, list_size=128)
    energy, balance = balanced_energy(block.positions, block.n)
    return PolarModel(block.polar, block.positions, np.sqrt(energy), block.n), balance


def make_frame(model, seed, frame):
    rng = np.random.default_rng([671129, seed, frame])
    k, variance = 100, 1. / (200 * 10**.05)
    messages = rng.integers(2, size=(k, 100), dtype=np.uint8)
    labels = [integer(b) for b in messages]
    y = rng.normal(0., np.sqrt(variance), model.n)
    for word in labels:
        p, values = model.symbols(word)
        y[model.positions[p]] += values
    return PolarScenario(model, y, k, variance), labels


def native(s):
    start = time.perf_counter()
    model, residual, accepted = s.model, s.y.copy(), {}
    work, calls, rounds, valid_ids = Work(model), 0, [], set()
    for step in range(30):
        remaining = s.k - len(accepted)
        if remaining == 0:
            break
        current = []
        variance = s.noise_var + (remaining - 1) / model.n
        for p in work.patterns(residual, remaining + 5):
            llr = 2 * model.amplitudes[p] * residual[model.positions[p]] / variance
            payloads, _ = model.code.decode_list(llr)
            calls += 1
            valid = [model.word(p, v) for v in payloads]
            valid_ids.update(valid)
            if valid and valid[0] not in accepted:
                accepted[valid[0]] = step
                current.append(valid[0])
                if len(accepted) == s.k:
                    break
        rounds.append(len(current))
        if not current:
            break
        for word in current:
            p, values = model.symbols(word)
            residual[model.positions[p]] -= values
    return accepted, {"calls": calls, "rounds": rounds, "seconds": time.perf_counter()-start,
                      "valid_ids": sorted(valid_ids), "residual_sha256": hashlib.sha256(residual.tobytes()).hexdigest()}


def main(out, seed, frames):
    model, balance = geometry(seed)
    path = out / f"native_{seed}.jsonl"
    with path.open("x") as handle:
        for i in range(frames):
            s, labels = make_frame(model, seed, i)
            accepted, meta = native(s)
            record = dict(seed=seed, frame=i, y_sha256=hashlib.sha256(s.y.tobytes()).hexdigest(), balance=balance,
                          accepted=list(accepted), truth=labels, **meta)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    print(seed, frames, "native frames complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    main(args.out, args.seed, args.frames)
