"""Twelve-geometry local preview with paired bisection of the two low-load thresholds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

import load_curve as study
from analysis import analyze, crossing

N = 12
STEP_CDB = 5
ROUNDS = 4
OUT = study.OUT / 'preview12'


def samples(k):
    start = study.seeds(k, 'preview').start
    return range(start, start + N)


def receipt(k, seed, db):
    if seed in study.seeds(k, 'preview') and db in study.PREVIEW[k]:
        return study.row_path(study.OUT, k, 'preview', seed, db)
    return study.row_path(OUT, k, 'frames', seed, db)


def measure(task):
    revision, k, seed, grid = task
    model = None
    added = 0
    for db in grid:
        path = receipt(k, seed, db)
        if not path.exists():
            if model is None:
                model, construction = study.geometry(seed)
            row = study.case(model, construction, k, seed, db)
            row['receiver_release'] = revision
            added += int(study.save_once(path, row))
        study.checked(path, revision, k, seed, db)
    return added


def load(k, grid, revision):
    counts = np.zeros((N, len(grid), len(study.ARMS)), dtype=np.int64)
    false = np.zeros_like(counts)
    seconds = 0.
    seen = set()
    for i, seed in enumerate(samples(k)):
        signature = None
        for j, db in enumerate(grid):
            row = study.checked(receipt(k, seed, db), revision, k, seed, db)
            sig = tuple(row['source'][key] for key in study.PAIR_HASHES)
            if signature is not None and sig != signature:
                raise ValueError('Source, noise or encoder changed across energies')
            signature = sig
            y = row['source']['y_sha256']
            if y in seen:
                raise ValueError('Repeated physical observation in preview')
            seen.add(y)
            for a, name in enumerate(study.ARMS):
                arm = row['arms'][name]
                counts[i, j, a] = arm['metrics']['count_misses']
                false[i, j, a] = arm['metrics']['false_copies']
                seconds += arm['seconds']
    return counts, false, seconds


def midpoints(grid, totals, users):
    points = set()
    for column in totals.T:
        status, _, _, bracket = crossing(grid, column, users)
        if status == 'interpolated':
            lo, hi = map(int, bracket)
            if hi - lo > STEP_CDB:
                points.add((lo + hi) // 2)
        elif status == 'at_grid':
            point = int(bracket[0])
            points.update(x for x in (point - STEP_CDB, point + STEP_CDB) if grid[0] < x < grid[-1])
    return sorted(points.difference(grid))


def run(workers):
    started = time.monotonic()
    revision = study.receiver_version()
    grids = {k: list(v) for k, v in study.PREVIEW.items()}
    todo = {k: list(v) for k, v in grids.items()}
    rounds = []
    added = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for depth in range(ROUNDS + 1):
            tasks = [(revision, k, s, work) for k, work in todo.items() if work for s in samples(k)]
            if not tasks:
                break
            added += sum(pool.map(measure, tasks))
            rounds.append({str(k): work for k, work in todo.items()})
            print(f'Completed round {depth}, energies in hundredths of a dB: {todo}', flush=True)
            for k, work in todo.items():
                grids[k] = sorted(set(grids[k]) | set(work))
            if depth < ROUNDS:
                todo = {k: midpoints(grid, load(k, grid, revision)[0].sum(axis=0), N*k) for k, grid in grids.items()}
    result = {}
    for k, grid in grids.items():
        counts, false, seconds = load(k, grid, revision)
        row = analyze(k, grid, counts)
        row['receiver_seconds_sum'] = seconds
        row['false_copies'] = {name: false.sum(axis=0)[:, a].tolist() for a, name in enumerate(study.ARMS)}
        result[str(k)] = row
    summary = dict(receiver_release=revision, exploratory=True, frames_per_load=N, target=.05, rounds=rounds, loads=result,
                   interval_scope='Frame bootstrap on the evaluated grid. These same 12 frames also selected the refinement energies.')
    study.equal_file(OUT/'summary.json', summary)
    elapsed = time.monotonic() - started
    study.save_once(OUT/'first_run_time.json', dict(workers=workers, added_paired_cases=added, wall_seconds=elapsed))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    result = run(args.workers)
    from grid_precision import report
    from plot_preview12 import render
    report(True)
    render(result, OUT)
    print({k: {a:z['snr_db'] for a,z in v['arms'].items()} for k,v in result['loads'].items()})
