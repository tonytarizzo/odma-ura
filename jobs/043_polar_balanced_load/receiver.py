"""Balanced scalar B100 geometry and independent complete DEC-38/39 calls for a fixed load."""
from importlib import import_module
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
paper_curve = import_module('jobs.039_polar_paper.paper_curve')
from framework.polar_catalogue import run

ARMS = ('DEC-38', 'DEC-39')
PHYSICAL_HASHES = ('bits_sha256', 'standard_noise_sha256', 'positions_sha256', 'amplitudes_sha256', 'information_sha256', 'y_sha256')


def geometry(seed):
    """Build the original Study 042 balanced geometry independently of load and noise."""
    start = time.perf_counter()
    model, balance = paper_curve.rollback_frames.geometry(seed)
    if (model.B, model.n, model.p, model.b, model.code.n, model.code.crc_bits, model.code.list_size) != (100, 30000, 8192, 87, 512, 16, 128):
        raise ValueError("Incompatible balanced encoder")
    if not 0 <= balance['max_relative_row_error'] <= 1e-10:
        raise ValueError("Balanced physical chip load is outside its declared tolerance")
    return model, time.perf_counter() - start


def case(model, geometry_seconds, k, seed, db):
    scenario, labels, source = paper_curve.make_frame(model, k, seed, db)
    scenario.y.flags.writeable = False
    arms = {}
    for name in (ARMS if seed % 2 == 0 else ARMS[::-1]):
        started = time.perf_counter()
        action, meta = run(name, scenario)
        seconds = time.perf_counter() - started
        if not meta['complete'] or meta['catalogue_id'] != name:
            raise RuntimeError(f"Incomplete {name} receiver on K={k}, seed {seed}, energy {db}")
        arms[name] = dict(action=sorted(action.items()), metrics=paper_curve.score(labels, action), seconds=seconds,
                          with_geometry=seconds + geometry_seconds, native_calls=meta['native']['calls'],
                          native_rounds=len(meta['native']['rounds']), continuation_calls=meta['continuation']['actual_scl_calls'])
    return dict(k=k, seed=seed, db=db, encoding='balanced', source=source, labels=labels, geometry_seconds=geometry_seconds, arms=arms)


def validate(row, k, seed, db):
    if (row['k'], row['seed'], row['db'], row['encoding']) != (k, seed, db, 'balanced') or set(row['arms']) != set(ARMS):
        raise ValueError("Wrong observation or receiver inventory")
    labels, source = row['labels'], row['source']
    if len(labels) != k or any(type(w) is not int or not 0 <= w < 1 << 100 for w in labels):
        raise ValueError("Invalid B100 source")
    if source['variance'] != 1 / (200 * 10 ** (db / 1000)):
        raise ValueError("Wrong Eb/N0 convention")
    if any(not isinstance(source[key], str) or len(source[key]) != 64 or
           any(c not in '0123456789abcdef' for c in source[key]) for key in PHYSICAL_HASHES):
        raise ValueError("Invalid physical fingerprint")
    rng = np.random.default_rng([671129, seed, 0])
    drawn = rng.integers(2, size=(k, 100), dtype=np.uint8)
    observed = np.unpackbits(np.array([list(w.to_bytes(13, 'big')) for w in labels], dtype=np.uint8), axis=1)[:, 4:]
    if not np.array_equal(observed, drawn) or source['bits_sha256'] != paper_curve.digest(drawn):
        raise ValueError("Inconsistent IID source or seed")
    if source['standard_noise_sha256'] != paper_curve.digest(rng.standard_normal(30000)):
        raise ValueError("Inconsistent standard noise or seed")
    if source['repeated_patterns'] != k - len({w >> 87 for w in labels}) or source['repeated_messages'] != k - len(set(labels)):
        raise ValueError("Inconsistent source multiplicities")
    geometry_seconds = row['geometry_seconds']
    if not math.isfinite(geometry_seconds) or geometry_seconds <= 0:
        raise ValueError("Missing balanced geometry cost")
    for arm in row['arms'].values():
        action = dict(arm['action'])
        if len(action) != len(arm['action']) or paper_curve.score(labels, action) != arm['metrics']:
            raise ValueError("Inconsistent complete-message metrics")
        if not math.isfinite(arm['seconds']) or arm['seconds'] <= 0 or arm['with_geometry'] != arm['seconds'] + geometry_seconds:
            raise ValueError("Invalid complete-receiver cost")
        if not (0 < arm['native_calls'] <= 30 * (k + 5) and 0 < arm['native_rounds'] <= 30 and arm['continuation_calls'] >= 0):
            raise ValueError("Invalid SCL call accounting")
