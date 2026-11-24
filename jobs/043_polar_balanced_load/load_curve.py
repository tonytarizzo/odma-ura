"""Scout and confirm required energy at 5% count-PUPE for two balanced B100 receivers."""
import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[1]
sys.path[:0] = [str(JOB), str(ROOT)]
from receiver import ARMS, case, geometry, validate
from analysis import analyze
parent = import_module('jobs.042_polar_combinations.paper_curve_extension')
save_once = parent.save_once

OUT = ROOT / 'results/dec43_balanced_load'
LOADS = (50, 100, 150, 200, 250, 300)
COARSE = (-150, 0, 150, 400, 800, 1600)
LIMITS = (-300, 3000)
PILOT_FRAMES, CONFIRM_FRAMES, SEEDS_PER_TASK = 16, 256, 2
PREVIEW = {50: (-100, -50, 0, 50), 100: (-50, 0, 25, 50)}
PAIR_HASHES = ('bits_sha256', 'standard_noise_sha256', 'positions_sha256', 'amplitudes_sha256', 'information_sha256')


def seeds(k, panel):
    first = 1043000 + 2000 * LOADS.index(k)
    start, n = {'pilot': (0, PILOT_FRAMES), 'confirm': (100, CONFIRM_FRAMES), 'preview': (800, 8)}[panel]
    return range(first + start, first + start + n)


def receiver_version():
    frozen = parent.release()[1]
    return hashlib.sha256((frozen + parent.sha(JOB / 'receiver.py')).encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def row_path(out, k, panel, seed, db):
    return out / panel / f'k{k}' / f's{seed}_e{db}.json'


def checked(path, revision, k, seed, db):
    row = read(path)
    if row['receiver_release'] != revision:
        raise ValueError(f'Incompatible receiver release at {path}')
    validate(row, k, seed, db)
    return row


def observations(out, revision, k, panel, sample, energies):
    for seed in sample:
        model = None
        for db in energies:
            path = row_path(out, k, panel, seed, db)
            if not path.exists():
                if model is None:
                    model, construction = geometry(seed)
                row = case(model, construction, k, seed, db)
                row['receiver_release'] = revision
                save_once(path, row)
            checked(path, revision, k, seed, db)
        del model


def collect(out, revision, k, panel, energies):
    panel_seeds = seeds(k, panel)
    shape = (len(panel_seeds), len(energies), len(ARMS))
    arrays = {key: np.zeros(shape, dtype=np.int64) for key in ('count_misses', 'list_misses', 'false_copies')}
    seconds = np.zeros(shape)
    seen = set()
    for i, seed in enumerate(panel_seeds):
        first = None
        for j, db in enumerate(energies):
            row = checked(row_path(out, k, panel, seed, db), revision, k, seed, db)
            sig = tuple(row['source'][key] for key in PAIR_HASHES)
            if first is not None and first != sig:
                raise ValueError(f'Unpaired source, geometry or standardized noise for K={k}, seed {seed}')
            first = sig
            obs = row['source']['y_sha256']
            if obs in seen:
                raise ValueError(f'Duplicate observation for K={k}, seed={seed}, energy={db}')
            seen.add(obs)
            for a, name in enumerate(ARMS):
                for metric, arr in arrays.items():
                    arr[i, j, a] = row['arms'][name]['metrics'][metric]
                seconds[i, j, a] = row['arms'][name]['with_geometry']
    return arrays, seconds


def results(out, revision, k, panel, energies):
    arrays, seconds = collect(out, revision, k, panel, energies)
    data = analyze(k, energies, arrays['count_misses'])
    list_result = analyze(k, energies, arrays['list_misses']) if np.any(arrays['list_misses'] != arrays['count_misses']) else data
    data['membership_thresholds'] = {name: list_result['arms'][name]['snr_db'] for name in ARMS}
    for a, name in enumerate(ARMS):
        data['arms'][name].update(list_misses=arrays['list_misses'][:, :, a].sum(0).tolist(),
                                false_copies=arrays['false_copies'][:, :, a].sum(0).tolist(),
                                mean_seconds_with_geometry=seconds[:, :, a].mean(0).tolist())
    return data


def choose_grid(energies, totals, users, confirmation):
    """Refine the last failing energy and its next passing neighbor for each receiver."""
    proposed = set()
    for misses in np.asarray(totals).T:
        passing = [20 * int(v) <= users for v in misses]
        failures = [i for i, ok in enumerate(passing) if not ok]
        if not failures or not passing[-1]:
            low_edge = not failures
            limit = LIMITS[0 if low_edge else 1]
            if confirmation:
                edge = energies[0 if low_edge else -1]
                sign = 1 if low_edge else -1
                proposed.update((edge, edge + 25 * sign, edge + 100 * sign))
            elif limit not in energies:
                proposed.add(limit)
        else:
            i = failures[-1]
            lo, hi = energies[i:i+2]
            if confirmation:
                left, right = int(misses[i]), int(misses[i+1])
                center = 5 * round((lo + (hi - lo) * (left - users / 20) / (left - right)) / 5)
                proposed.update((lo, hi, center-50, center-20, center, center+20, center+50))
            elif hi - lo > 10:
                proposed.add((lo + hi) // 2)
        # A noisy recrossing is evidence on two sides of the target. Retain witnesses for the independent panel.
        if confirmation and failures and any(passing[:failures[-1]]):
            proposed.update((energies[passing.index(True)], energies[failures[-1]]))
    accepted = {min(LIMITS[1], max(LIMITS[0], v)) for v in proposed}
    return sorted(accepted if confirmation else accepted.difference(energies))


def declaration(revision):
    files = [JOB / n for n in ('load_curve.py', 'receiver.py', 'analysis.py', 'plot_load.py')]
    files += [ROOT / n for n in ('benchmarks/ura_bounds.py', 'src/ura_bound.py', 'uv.lock')]
    return dict(receiver_release=revision, job042=parent.release()[1], loads=list(LOADS), coarse_cdb=list(COARSE),
                limits_cdb=list(LIMITS), pilot_frames=PILOT_FRAMES, confirmation_frames=CONFIRM_FRAMES,
                scientific_inputs={str(p.relative_to(ROOT)): parent.sha(p) for p in files})


def equal_file(path, data):
    save_once(path, data)
    if read(path) != data:
        raise ValueError(f'Existing experiment record differs at {path}')


def plan(out, revision, k, stage, *, recompute=False):
    if stage == 0:
        return dict(history=[], energies_cdb=list(COARSE))
    path = out / 'plans' / f'k{k}_stage{stage}.json'
    if path.exists() and not recompute:
        result = read(path)
        if (result['k'], result['stage'], result['receiver_release']) != (k, stage, revision):
            raise ValueError(f'Wrong pilot plan at {path}')
        return result
    previous = plan(out, revision, k, stage - 1, recompute=recompute)
    history = sorted(set(previous['history']) | set(previous['energies_cdb']))
    arrays, _ = collect(out, revision, k, 'pilot', history)
    totals = arrays['count_misses'].sum(axis=0)
    next_grid = choose_grid(history, totals, PILOT_FRAMES * k, confirmation=stage == 4)
    result = dict(k=k, stage=stage, receiver_release=revision, history=history, pilot_misses=totals.tolist(),
                  energies_cdb=next_grid, pilot_users_per_energy=PILOT_FRAMES*k)
    equal_file(path, result)
    return result


def shard(index, panel):
    width = {'pilot': PILOT_FRAMES, 'confirm': CONFIRM_FRAMES, 'preview': 8}[panel] // SEEDS_PER_TASK
    loads = list(PREVIEW) if panel == 'preview' else LOADS
    if not 1 <= index <= width * len(loads):
        raise ValueError(f'Invalid {panel} shard {index}')
    group, batch = divmod(index - 1, width)
    k = loads[group]
    start = batch * SEEDS_PER_TASK
    return k, seeds(k, panel)[start:start+SEEDS_PER_TASK]


def merge(out, revision, preview):
    from plot_load import render
    by_load = {}
    for k in (PREVIEW if preview else LOADS):
        grid = list(PREVIEW[k]) if preview else plan(out, revision, k, 4, recompute=True)['energies_cdb']
        by_load[str(k)] = results(out, revision, k, 'preview' if preview else 'confirm', grid)
    report = dict(B=100, n=30000, target=.05, encoder='balanced, unit energy, 13 placement bits plus 87 payload',
                  metric='missed copies divided by transmitted copies', receiver_release=revision, preview=preview, loads=by_load)
    if not preview:
        report['release'] = read(out/'release.json')
    equal_file(out/('preview_summary.json' if preview else 'summary.json'), report)
    render(report, out, preview=preview)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', type=int, metavar='ROUND', choices=range(4))
    group.add_argument('--confirm', action='store_true')
    group.add_argument('--preview', action='store_true')
    group.add_argument('--merge', action='store_true')
    group.add_argument('--merge-preview', action='store_true')
    group.add_argument('--inventory', action='store_true')
    p.add_argument('--shard', type=int)
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    revision = receiver_version()
    settings = declaration(revision)
    if args.inventory:
        print(f'{LOADS}, {PILOT_FRAMES} pilot and {CONFIRM_FRAMES} independent confirmation frames/load/energy.')
        print('Four pilot arrays of 48 tasks, one confirmation array of 768 tasks and one figure/summary job.')
        print(f'Original DEC-38/39 receiver fingerprint {revision}')
        return
    panel = 'preview' if args.preview else 'confirm' if args.confirm else 'pilot'
    if not (args.preview or args.merge_preview):
        equal_file(args.out/'release.json', settings)
    if args.merge or args.merge_preview:
        if args.shard is not None:
            p.error('a merge processes all shards')
        merge(args.out, revision, args.merge_preview)
        return
    if args.shard is None:
        p.error('supply the one-based --shard index')
    k, sample = shard(args.shard, panel)
    energies = PREVIEW[k] if args.preview else plan(args.out, revision, k, 4 if args.confirm else args.pilot)['energies_cdb']
    print(f'{panel} K={k}, seeds={list(sample)}, energy (centidB)={energies}', flush=True)
    observations(args.out, revision, k, panel, sample, energies)
    if args.pilot is not None:
        # The last pilot writer prepares the next grid before the dependent PBS array starts.
        complete = all(row_path(args.out, k, panel, seed, db).exists() for seed in seeds(k, panel) for db in energies)
        if complete:
            plan(args.out, revision, k, args.pilot + 1)


if __name__ == '__main__':
    main()
