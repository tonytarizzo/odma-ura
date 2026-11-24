"""Independent balanced native receipts and flat GMAC paper-entry controls for Study 042."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"jobs/037_polar_tractable"), str(ROOT/"jobs/039_polar_paper")]
import rollback_frames
import paper_curve
from framework.polar_tractable import integer

SEEDS = tuple(range(984001, 984017)) + tuple(range(984101, 984117))
FLAT = (984001, 984016, 984101, 984116)


def run(seed):
    start = time.perf_counter()
    model, balance = rollback_frames.geometry(seed)
    setup = time.perf_counter() - start
    scenario, labels = rollback_frames.make_frame(model, seed, 0)
    accepted, meta = rollback_frames.native(scenario)
    record = dict(seed=seed, y_sha256=paper_curve.digest(scenario.y), geometry_s=setup, balance=balance,
                  native_order=list(accepted.items()), metrics=paper_curve.score(labels, dict.fromkeys(accepted, 1)), **meta)
    if seed in FLAT:
        flat, geometry_seconds = paper_curve.make_geometry(100, seed)
        sf, truth, source = paper_curve.make_frame(flat, 100, seed, 50)
        if labels != truth or not np.array_equal(model.positions, flat.positions):
            raise RuntimeError("Paired flat/balanced source or positions differ")
        block = paper_curve.baseline(100, seed)
        digest = paper_curve.verify_physical(sf, block, truth, seed)
        ordinary, ordinary_meta = rollback_frames.native(sf)
        words, direct = block.decode(sf.y, sf.k, sf.noise_var)
        ids = [integer(v) for v in words]
        record["flat"] = dict(source=source, y_sha256=digest, geometry_s=geometry_seconds, accepted=ids,
                              order_equal=ids == list(ordinary), calls_equal=direct["polar_trials"] == ordinary_meta["calls"],
                              metrics=paper_curve.score(labels, dict.fromkeys(ids, 1)), direct_meta=direct,
                              ordinary_meta=ordinary_meta)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        path = args.out / f"native_{seed}.json"
        if path.is_file():
            continue
        row = run(seed)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(row) + "\n")
        temp.replace(path)
        print(f"native {seed} complete", flush=True)
