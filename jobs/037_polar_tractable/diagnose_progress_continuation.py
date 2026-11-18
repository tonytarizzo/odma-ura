"""Reused96 development of one conditional fourth original fixed-queue count-feedback block."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rollback_frames import geometry, make_frame
from framework.polar_count_feedback import continue_feedback, support_completion

BASE = ROOT / 'results/polar_dec35_exploration_2026-11-11/progress_continuation'
SEEDS = list(range(6501, 6533)) + list(range(6601, 6633)) + list(range(6701, 6717)) + list(range(6901, 6917))
MODES = ('three', 'conditional_fourth')
SOURCES = ['jobs/037_polar_tractable/diagnose_progress_continuation.py', 'jobs/037_polar_tractable/rollback_frames.py',
    'framework/polar_count_feedback.py', 'tests/polar_progress_continuation_test.py',
    'tests/polar_progress_continuation_driver_test.py', 'framework/polar_count_cavity.py',
    'framework/polar_full_pattern.py', 'framework/polar_tractable.py', 'baselines/polar.py',
    'baselines/odma_polar.py', 'benchmarks/polar_energy_geometry.py', 'uv.lock', 'pyproject.toml']
QUERY_KEYS = ('query', 'sweep', 'stage', 'pattern', 'list_size', 'context', 'restored', 'restored_count',
    'outside_count', 'pending', 'variance', 'pre_action_sha256', 'llr_sha256', 'ids', 'new_ids', 'refit_index')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value))


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def source_path(seed):
    if 6501 <= seed <= 6532:
        folder = 'polar_continued_largeB_2026-11-11/raw_confirmation'
    elif 6601 <= seed <= 6632:
        folder = 'polar_count_feedback_2026-11-11/main'
    elif 6701 <= seed <= 6716:
        folder = 'polar_dec35_exploration_2026-11-11/rate_zero/receiver/main'
    elif 6901 <= seed <= 6916:
        folder = 'polar_dec35_exploration_2026-11-11/repetition/receiver/main'
    else:
        raise ValueError('Seed is outside the fixed development panel')
    return ROOT / 'results' / folder / f'frame_{seed}.json'


def saved_inputs(saved):
    seed = saved['seed']
    if seed < 6700:
        prior = saved['arms']['raw128' if seed < 6600 else 'reference']
        paid = saved['native_wall_seconds'] + saved['rollback_wall_seconds'] if seed < 6600 else saved['initialization_seconds']
        return Counter(dict(saved['rollback_action'])), set(saved['initial_bank']), prior['meta'], prior['action'], prior['bank'], paid
    arm = saved['arms']['native_only' if seed < 6800 else 'candidate']['meta']
    setup = sum(value for key, value in arm.items() if key.endswith('_preparation_seconds'))
    paid = arm['native']['seconds'] + arm['rollback_seconds'] + setup
    return Counter(dict(arm['rollback_action'])), set(arm['base_bank']), arm['continuation'], arm['cancellation_action'], arm['bank'], paid


def current_hashes():
    assessment = ROOT / 'results/polar_dec35_exploration_2026-11-11/algebra/progress_triggered_continuation_assessment.json'
    return dict(source_sha256={name: digest(ROOT / name) for name in SOURCES},
        saved_frame_sha256={str(source_path(seed).relative_to(ROOT)): digest(source_path(seed)) for seed in SEEDS},
        design_sha256=digest(BASE / 'design.json'), assessment_sha256=digest(assessment),
        source_before_sha256=digest(BASE / 'source_before/framework/polar_count_feedback.py'))


def metrics(action, bank, labels, k):
    truth = Counter(labels)
    matches = (truth & action).total()
    return dict(misses=k-matches, false_copies=action.total()-matches, output_copies=action.total(),
        bank_misses=sum(count for word, count in truth.items() if word not in bank))


def verify_queries(actual, expected, *, cache=False):
    keys = QUERY_KEYS + (('cache_hit',) if cache else ())
    assert len(actual) == len(expected)
    assert all(all(a[key] == b[key] for key in keys) for a, b in zip(actual, expected, strict=True))


def verify_refits(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        for key in ('query', 'sweep', 'new_ids'):
            assert a[key] == b[key]
        assert a['meta']['moves'] == b['meta']['moves']


def verify_three(arm, saved):
    _, _, prior, action, bank, _ = saved_inputs(saved)
    new = normalized(arm['meta'])
    assert normalized(arm['cancellation_action']) == action and arm['bank'] == bank
    for key in ('initial_refit_action', 'complete', 'stop', 'sweep_cap_with_new_ids', 'sweeps'):
        assert new[key] == prior[key]
    assert new['initial_refit']['moves'] == prior['initial_refit']['moves']
    verify_queries(new['queries'], prior['queries'])
    verify_refits(new['refits'], prior['refits'])


def verify_prefix(three, fourth):
    old, new = normalized(three['meta']), normalized(fourth['meta'])
    trigger = old['complete'] and len(old['sweeps']) == 3 and old['stop'] == 'sweep_cap' and bool(old['sweeps'][2]['moves'])
    assert new['extra_sweep_triggered'] == trigger
    assert old['initial_refit_action'] == new['initial_refit_action']
    assert old['initial_refit']['moves'] == new['initial_refit']['moves']
    assert old['sweeps'] == new['sweeps'][:len(old['sweeps'])]
    verify_queries(new['queries'][:len(old['queries'])], old['queries'], cache=True)
    verify_refits(new['refits'][:len(old['refits'])], old['refits'])
    if trigger:
        assert len(new['sweeps']) == 4 and new['extra_sweep_initial_action'] == normalized(three['cancellation_action'])
    else:
        assert len(new['queries']) == len(old['queries']) and len(new['refits']) == len(old['refits'])
        for key in ('action', 'cancellation_action', 'bank'):
            assert normalized(three[key]) == normalized(fourth[key])
    return trigger


def run_arm(scenario, initial, ids, mode):
    started = time.perf_counter()
    if mode not in MODES:
        raise ValueError('Require a frozen arm')
    cancellation, bank, meta = continue_feedback(scenario, initial, ids, memoize=True,
        progress_extra_sweep=mode == 'conditional_fourth')
    tick = time.perf_counter()
    action, completion = support_completion(scenario, cancellation, bank)
    completion_seconds = time.perf_counter()-tick
    return dict(action=sorted(action.items()), cancellation_action=sorted(cancellation.items()), bank=sorted(bank),
        meta=meta, completion=completion, completion_wall_seconds=completion_seconds, wall_seconds=time.perf_counter()-started)


def summarize(rows):
    if [row['seed'] for row in rows] != SEEDS:
        raise ValueError('Require all96 fixed reused frames in order')
    totals = {mode: {key: sum(row['arms'][mode]['metrics'][key] for row in rows)
        for key in ('misses', 'false_copies', 'output_copies', 'bank_misses')} for mode in MODES}
    gains = [row['arms']['three']['metrics']['misses']-row['arms']['conditional_fourth']['metrics']['misses'] for row in rows]
    exclusive = set().union(*(set(row['exclusive_retained']) for row in rows))
    frames = [row['seed'] for row in rows if row['exclusive_retained']]
    complete = all(row['arms'][mode]['meta']['complete'] for row in rows for mode in MODES)
    fits = all(fit['converged'] for row in rows for mode in MODES for fit in
        [row['arms'][mode]['meta']['initial_refit'], *[r['meta'] for r in row['arms'][mode]['meta']['refits']]])
    counters = all(len(a['meta']['queries']) == a['meta']['actual_scl_calls']+a['meta']['cache_hits']
        and a['meta']['cache_hits'] == sum(q['cache_hit'] for q in a['meta']['queries'])
        for row in rows for a in row['arms'].values())
    if not counters:
        raise ValueError('Inconsistent logical/actual/cache counters')
    gates = dict(exclusive_IDs=len(exclusive) >= 2, exclusive_frames=len(frames) >= 2,
        net_miss_gain=sum(gains) >= 2, net_without_best_frame=sum(gains)-max(gains) >= 1,
        all96_complete=complete, all_fits_converged=fits,
        count_caps=all(row['arms'][mode]['metrics']['output_copies'] <= 100 for row in rows for mode in MODES),
        identical_original_prefix=all(row['three_trace_matches_original'] and row['prefix_identical'] for row in rows))
    return dict(frames=len(rows), totals=totals, exclusive_retained=sorted(exclusive), exclusive_frames=frames,
        gates=gates, advance=all(gates.values()), net_misses_saved=sum(gains), largest_frame_gain=max(gains),
        net_without_best_frame=sum(gains)-max(gains), maximum_frame_regression=max(0, -min(gains)),
        better_frames=sum(g > 0 for g in gains), worse_frames=sum(g < 0 for g in gains), tied_frames=gains.count(0),
        triggered_frames=[row['seed'] for row in rows if row['triggered']],
        new_continuation_wall_seconds={m:sum(row['arms'][m]['wall_seconds'] for row in rows) for m in MODES},
        new_geometry_seconds=sum(row['geometry_seconds'] for row in rows),
        historical_paid_initialization_seconds=sum(row['paid_initialization_seconds'] for row in rows),
        mixed_time_standalone_estimate={m:sum(row['arms'][m]['wall_seconds']+row['geometry_seconds']
            +row['paid_initialization_seconds'] for row in rows) for m in MODES},
        logical_queries={m:sum(len(row['arms'][m]['meta']['queries']) for row in rows) for m in MODES},
        actual_scl_calls={m:sum(row['arms'][m]['meta']['actual_scl_calls'] for row in rows) for m in MODES},
        cache_hits={m:sum(row['arms'][m]['meta']['cache_hits'] for row in rows) for m in MODES},
        ranking_seconds={m:sum(row['arms'][m]['meta']['ranking_seconds'] for row in rows) for m in MODES},
        scope='All96 frames are reused development; native/rollback stages were paid previously. Mixed-time estimates '
              'are not newly measured complete receiver runtimes. No timing ratio gate, fifth block or sweep-count search.')


def main(out, preflight_path=None):
    started = time.perf_counter()
    design = json.loads((BASE / 'design.json').read_text())
    preflight_path = preflight_path or BASE / 'pre_main.json'
    preflight = json.loads(preflight_path.read_text())
    hashes = current_hashes()
    if preflight.get('passed') is not True or any(preflight[key] != value for key, value in hashes.items()):
        raise ValueError('Require independently reviewed unchanged source, design, assessment and96 input hashes')
    if design['panel']['seeds'] != SEEDS or design['gates']['minimum_total_misses_saved_vs_three'] != 2:
        raise ValueError('Frozen panel or gate differs')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    write(out / 'source_freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(), **hashes,
        preflight_sha256=digest(preflight_path), preflight_path=str(preflight_path.relative_to(ROOT))))
    (out / 'design.json').write_bytes((BASE / 'design.json').read_bytes())
    (out / 'feedback_before.py').write_bytes((BASE / 'source_before/framework/polar_count_feedback.py').read_bytes())
    for name in SOURCES:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    rows = []
    for index, seed in enumerate(SEEDS):
        saved = json.loads(source_path(seed).read_text())
        tick = time.perf_counter()
        model, _ = geometry(seed)
        geometry_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        scenario, labels = make_frame(model, seed, 0)
        source_seconds = time.perf_counter()-tick
        hashes_before = {name:hashlib.sha256(data.tobytes()).hexdigest() for name, data in
            [('positions', model.positions), ('amplitudes', model.amplitudes), ('y', scenario.y)]}
        expected = saved['input_snapshot']['arrays'] if seed >= 6700 else {k:saved[k+'_sha256'] for k in hashes_before}
        assert all(sha == expected[name] for name, sha in hashes_before.items()) and labels == saved['labels']
        initial, bank, _, _, _, paid = saved_inputs(saved)
        arms = {}
        order = MODES if index % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            tick = time.perf_counter()
            arms[mode] = run_arm(scenario, initial, bank, mode)
            wall_seconds = time.perf_counter()-tick
            arms[mode]['internal_wall_seconds'] = arms[mode]['wall_seconds']
            arms[mode]['wall_seconds'] = wall_seconds
        verify_three(arms['three'], saved)
        triggered = verify_prefix(arms['three'], arms['conditional_fourth'])
        for name, data in [('positions', model.positions), ('amplitudes', model.amplitudes), ('y', scenario.y)]:
            assert hashlib.sha256(data.tobytes()).hexdigest() == hashes_before[name]
        for mode, arm in arms.items():
            arm['metrics'] = metrics(Counter(dict(arm['action'])), arm['bank'], labels, scenario.k)
            arm['cancellation_metrics'] = metrics(Counter(dict(arm['cancellation_action'])), arm['bank'], labels, scenario.k)
        retained = set(dict(arms['conditional_fourth']['action'])) & set(labels)
        row = dict(seed=seed, frame=0, arms=arms, order=list(order), geometry_seconds=geometry_seconds,
            source_seconds=source_seconds, paid_initialization_seconds=paid, input_hashes=hashes_before,
            saved_source=str(source_path(seed).relative_to(ROOT)), initial_action=sorted(initial.items()), initial_bank=sorted(bank),
            exclusive_retained=sorted(retained-set(arms['three']['bank'])), three_trace_matches_original=True,
            prefix_identical=True, triggered=triggered)
        rows.append(row)
        write(out / f'frame_{seed}.json', row)
        print(json.dumps(dict(seed=seed, triggered=triggered, misses={m:arms[m]['metrics']['misses'] for m in MODES},
            exclusive=len(row['exclusive_retained']), elapsed_seconds=time.perf_counter()-started)), flush=True)
    assert current_hashes() == hashes
    result = summarize(rows)
    result.update(elapsed_seconds=time.perf_counter()-started, sources_and_inputs_unchanged=True,
        source_freeze_sha256=digest(out / 'source_freeze.json'),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    write(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    main(args.out, args.preflight.resolve() if args.preflight is not None else None)
