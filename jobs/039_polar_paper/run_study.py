"""Run an independently resumable PBS shard, or reduce the complete paper study."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import scipy

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[1]
sys.path.insert(0, str(JOB))
from paper_curve import CURVE_DB, CURVE_SEEDS, run_curve_frame, summarize_curve

CONTROL_SEEDS = range(933001, 933129)
OUTPUT = JOB / "results"


def shards():
    batches = []
    for k in (100, 50):
        seeds = CURVE_SEEDS[k]
        for snr in CURVE_DB[k]:
            for i in range(0, len(seeds), 16):
                batches.append([("curve", k, seed, snr) for seed in seeds[i:i + 16]])
    for i in range(0, len(CONTROL_SEEDS), 4):
        batches.append([("control", 100, seed, 50) for seed in CONTROL_SEEDS[i:i + 4]])
    return batches


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release():
    manifest = JOB / "paper_inputs.json"
    for path, expected in json.loads(manifest.read_bytes())["scientific_inputs"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Job 039 source differs from release: {path}")
    return sha(manifest)


def path_for(out, key):
    kind, k, seed, db = key
    return out / kind / f"k{k}_s{seed}_db{db}.json"


def save_once(path, data):
    # On the shared filesystem, expose a complete record only once, even after concurrent resubmission.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".writing-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            json.dump(data, stream, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(pending, path)
            except FileExistsError:
                return False
        finally:
            pending.unlink(missing_ok=True)
    return True


def checked(path, key, version):
    saved = json.loads(path.read_bytes())
    if saved["release"] != version or saved["key"] != list(key):
        raise ValueError(f"Mismatched persisted case {path}")
    row = saved["data"]
    if row["seed"] != key[2] or (key[0] == "curve" and (row["k"], row["snr_cdb"]) != (key[1], key[3])):
        raise ValueError(f"Incorrect physical case in {path}")
    return row


def run_shard(idx, batches, out, version):
    from paper_controls import run_control_frame
    import paper_quadrature
    if not 1 <= idx <= len(batches):
        raise ValueError(f"Expected a PBS array index between 1 and {len(batches)}")
    metadata = dict(job=os.environ.get("PBS_JOBID"), shard=idx, host=platform.node(), python=sys.version,
                    numpy=np.__version__, scipy=scipy.__version__, threads=os.environ.get("OPENBLAS_NUM_THREADS"))
    for key in batches[idx-1]:
        kind, k, seed, db = key
        dest = path_for(out, key)
        if dest.exists():
            checked(dest, key, version)
            print("Verified saved case", key, flush=True)
            continue
        started = time.perf_counter()
        if kind == "curve":
            row = run_curve_frame(k, seed, db)
            if paper_quadrature.selected(k,seed):
                row["complex_screen"] = paper_quadrature.run_case(k,seed,db,row)
        else:
            row = run_control_frame(seed)
        save_once(dest, dict(key=key, release=version, worker=metadata, data=row))
        checked(dest, key, version)
        print("Completed", key, round(time.perf_counter() - started, 2), "s", flush=True)


def merge(out, batches, version):
    from paper_controls import summarize_controls
    from paper_outputs import write_crossings, write_tables_and_figure
    from paper_quadrature import summarize as summarize_quadrature
    from paper_terminal import summarize_terminal, write_terminal_tables
    cases = [key for batch in batches for key in batch]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate study case in manifest")
    files = {key:path_for(out, key) for key in cases}
    missing = [name for name in files.values() if not name.is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} expected records; first missing is {missing[0]}")
    def rows(kind):
        return (checked(f, key, version) for key, f in files.items() if key[0] == kind)
    curve = summarize_curve(rows("curve"))
    controls = summarize_controls(list(rows("control")))
    quadrature = summarize_quadrature(rows("curve"))
    terminal = summarize_terminal(rows("curve"))
    write_terminal_tables(terminal, out)
    write_tables_and_figure(curve, out)
    write_crossings(curve, quadrature, out)
    summary = dict(release=version, cases=len(cases), curve=curve, controls=controls, quadrature=quadrature, terminal=terminal,
                   record_sha256={str(f.relative_to(out)):sha(f) for f in files.values()})
    dest = out / "summary.json"
    if not save_once(dest, summary):
        if json.loads(dest.read_bytes()) != json.loads(json.dumps(summary, allow_nan=False)):
            raise ValueError("Existing summary disagrees with the current complete records")
    print(f"Verified {len(cases)} cases in {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--shard", type=int)
    which.add_argument("--merge", action="store_true")
    which.add_argument("--inventory", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    batches = shards()
    if args.inventory:
        print(json.dumps(dict(shards=len(batches), curves=sum(map(len, batches))-len(CONTROL_SEEDS),
                              controls=len(CONTROL_SEEDS), snr_hundredths_db=CURVE_DB)))
        return
    version = release()
    if args.merge:
        merge(args.out, batches, version)
    else:
        run_shard(args.shard, batches, args.out, version)


if __name__ == "__main__":
    main()
