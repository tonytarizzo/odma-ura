"""Frozen reused128 test of count-growth continuation; initialization costs remain historical."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.polar_count_cavity import action_hash
from framework.polar_count_feedback import continue_feedback, support_completion
from rollback_frames import geometry, make_frame
from diagnose_full_pattern import digest, evaluate, write_json
from diagnose_native_rate_zero import input_snapshot
from diagnose_progress_confirmation import SOURCES as PRIOR_SOURCES, valid_action
from diagnose_progress_continuation import normalized

EXPLORATION = ROOT/'results/polar_dec35_exploration_2026-11-11'
BASE = EXPLORATION/'count_growth'
OLD = EXPLORATION/'progress_continuation/confirmation'
SEEDS = list(range(7101, 7229))
MODES = ('conditional_fourth', 'count_growth')
SOURCES = list(dict.fromkeys(PRIOR_SOURCES+['framework/polar_count_growth.py',
    'tests/polar_count_growth_test.py', 'jobs/037_polar_tractable/diagnose_count_growth.py',
    'tests/polar_count_growth_driver_test.py']))
GATES = dict(all128_complete_legal_count_actions=True, all_source_physical_input_and_reference_replay_checks=True,
    identical_original_prefix_through_fourth_including_refits_bank_and_cache=True,
    all_started_blocks_and_calls_within_integer_bound=True, minimum_net_saved_count_copies=4,
    minimum_improved_frames=2, minimum_net_saved_copies_after_removing_best_frame=1,
    maximum_single_frame_regression_copies=2)


def semantic(value):
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items() if not key.endswith('seconds')}
    if isinstance(value, (list, tuple)):
        return [semantic(item) for item in value]
    return value


def saved_path(seed):
    return OLD/'main'/f'frame_{seed}.json'


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCES}


def evidence_hashes():
    paths = [saved_path(seed) for seed in SEEDS]
    paths += [OLD/'main'/name for name in ('source_freeze.json', 'summary.json')]
    paths += [OLD/'audit'/name for name in ('outcomes.py', 'outcomes.json')]
    paths += [EXPLORATION/'algebra/count_growth'/name for name in ('declaration.json', 'finite.py', 'finite.json')]
    paths += [BASE/'source_before/framework/polar_count_feedback.py', BASE/'interface_version.json']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def physical_input_hashes():
    return {str(saved_path(seed).relative_to(ROOT)): hashlib.sha256(json.dumps(
        json.loads(saved_path(seed).read_text())['input_snapshot'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for seed in SEEDS}


def check_design(design):
    assert all(design['frozen_development_gates'][key] == value for key, value in GATES.items())
    assert design['development_panel']['seeds'].startswith('All128 frame-zero seeds7101 through7228')
    assert design['development_panel']['arms'] == ['Verified conditional-fourth reference',
        'Identical first-four policy with strict endpoint count-growth continuation']


def freeze(out):
    if out.exists():
        raise FileExistsError(out)
    design, checked = [json.loads((BASE/name).read_text()) for name in ('design.json', 'pre_main.json')]
    check_design(design)
    sources, evidence, physical = source_hashes(), evidence_hashes(), physical_input_hashes()
    assert checked['passed'] and checked['design_sha256'] == digest(BASE/'design.json')
    assert checked['source_sha256'] == sources and checked['evidence_sha256'] == evidence
    assert checked['physical_input_sha256'] == physical
    assert json.loads((EXPLORATION/'algebra/count_growth/finite.json').read_text())['passed']
    assert json.loads((OLD/'audit/outcomes.json').read_text())['passed']
    previous = json.loads((OLD/'main/source_freeze.json').read_text())
    assert previous['source_sha256']['framework/polar_count_feedback.py'] == digest(BASE/'source_before/framework/polar_count_feedback.py')
    out.mkdir(parents=True)
    (out/'design.json').write_bytes((BASE/'design.json').read_bytes())
    for name in SOURCES:
        destination = out/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, evidence_sha256=evidence,
        physical_input_sha256=physical, design_sha256=digest(BASE/'design.json'), preflight_sha256=digest(BASE/'pre_main.json'))
    write_json(out/'source_freeze.json', manifest)
    return manifest


def saved_initialization(saved):
    prior = saved['arms']['C']['meta']
    return Counter(dict(prior['rollback_action'])), set(prior['base_bank']), dict(
        native_seconds=prior['native']['seconds'], rollback_seconds=prior['rollback_seconds'],
        native_preparation_seconds=prior['native_repetition_preparation_seconds'],
        scope='Previously measured initialization components only; never added to new continuation costs as a receiver runtime.')


def apply_refit(action, fit):
    action = action.copy()
    for move in fit['moves']:
        old, new = move['old'], move['new']
        if old is not None:
            assert action[old] > 0
            action[old] -= 1
            if not action[old]:
                del action[old]
        if new is not None:
            action[new] += 1
    return action


def verify_reference(arm, saved):
    prior = saved['arms']['C']
    assert semantic(arm['meta']) == semantic(prior['meta']['continuation'])
    assert normalized(arm['action']) == prior['action']
    assert normalized(arm['cancellation_action']) == prior['meta']['cancellation_action']
    assert arm['bank'] == prior['meta']['bank']
    assert semantic(arm['completion']) == semantic(prior['meta']['support_completion'])


def verify_prefix(reference, candidate, initial_bank, k=100):
    old, new = reference['meta'], candidate['meta']
    assert old['initial_refit_action'] == new['initial_refit_action']
    assert semantic(old['initial_refit']) == semantic(new['initial_refit'])
    count, fits = len(old['queries']), len(old['refits'])
    assert semantic(new['queries'][:count]) == semantic(old['queries'])
    assert semantic(new['refits'][:fits]) == semantic(old['refits'])
    assert new['sweeps'][:len(old['sweeps'])] == old['sweeps']
    for key in ('extra_sweep_triggered', 'third_sweep_count_moves', 'extra_sweep_initial_action'):
        assert new[key] == old[key]
    assert new['count_action_path'][:len(old['count_action_path'])] == old['count_action_path']
    bank = set(initial_bank)
    for query in new['queries'][:count]:
        bank.update(query['ids'])
    assert sorted(bank) == reference['bank']
    state = Counter(dict(new['initial_refit_action']))
    for fit in new['refits'][:fits]:
        state = apply_refit(state, fit['meta'])
    assert sorted(state.items()) == reference['cancellation_action']
    hits = sum(query['cache_hit'] for query in new['queries'][:count])
    assert hits == old['cache_hits'] and count-hits == old['actual_scl_calls']
    eligible = (old['complete'] and len(old['sweeps']) == 4 and old['stop'] == 'sweep_cap'
                and bool(old['sweeps'][3]['new_ids'])
                and state.total() > sum(value for _, value in old['extra_sweep_initial_action']))
    assert (len(new['sweeps']) > len(old['sweeps'])) == eligible
    if eligible:
        first = new['queries'][count]
        assert first['sweep'] == 4 and first['pre_action_sha256'] == action_hash(state)
        assert first['outside_count'] == first['pending'] == k-state.total()
    else:
        assert len(new['queries']) == count and len(new['refits']) == fits
        assert candidate['bank'] == reference['bank']
        assert candidate['action'] == reference['action']
        assert candidate['cancellation_action'] == reference['cancellation_action']
    return dict(identical=True, continued=eligible, shared_prefix_queries=count, shared_prefix_refits=fits,
        shared_prefix_bank=sorted(bank), shared_prefix_cancellation=sorted(state.items()))


def verify_growth(arm, k):
    meta, current = arm['meta'], Counter(dict(arm['meta']['initial_refit_action']))
    assert meta['count_growth'] and meta['maximum_sweeps'] == k+4
    assert len(meta['sweeps']) <= k+4 and len(meta['queries']) <= (k+4)*(k+5)
    trace, fourth = [], None
    for block in meta['sweeps']:
        sweep, start = block['sweep'], current.total()
        for refit in meta['refits']:
            if refit['sweep'] == sweep:
                current = apply_refit(current, refit['meta'])
        end = current.total()
        assert end == block['output_count'] and 0 <= end <= k
        assert block['queries'] <= min(len(block['ranked_patterns']), k+5)
        if sweep >= 3:
            grew, complete, fresh = end > start, block['complete_sweep'], block['new_ids']
            reason = (meta['stop'] if not complete else 'full_sweep_no_new_ids' if not fresh
                      else 'strict_endpoint_growth' if grew else 'no_endpoint_count_growth')
            trace.append(dict(sweep=sweep, start_count=start, end_count=end, grew=grew, complete_sweep=complete,
                new_ids=fresh, reason=reason, **{'continue': complete and bool(fresh) and grew}))
            if sweep == 3 and complete and grew:
                fourth = end
            if sweep >= 4:
                assert fourth is not None and sweep-3 <= k-fourth+1 and start >= fourth+sweep-4
    assert normalized(meta['count_growth_trace']) == trace
    assert meta['fourth_endpoint_count'] == fourth
    assert meta['later_sweeps'] == max(0, len(meta['sweeps'])-4)
    assert meta['later_sweep_bound'] == (k-fourth+1 if fourth is not None else 0)
    assert meta['later_sweeps'] <= meta['later_sweep_bound']
    assert not trace or not trace[-1]['continue']
    assert sorted(current.items()) == arm['cancellation_action']
    return True


def run_arm(scenario, initial, bank, mode, progress=None):
    if mode not in MODES:
        raise ValueError('Require one of the two frozen policies')
    started = time.perf_counter()
    cancellation, seen, meta = continue_feedback(scenario, initial, bank, memoize=True, progress=progress,
        progress_extra_sweep=True, count_growth=mode == 'count_growth')
    tick = time.perf_counter()
    action, completion = support_completion(scenario, cancellation, seen)
    completion_wall = time.perf_counter()-tick
    return dict(action=sorted(action.items()), cancellation_action=sorted(cancellation.items()), bank=sorted(seen),
        meta=meta, completion=completion, completion_wall_seconds=completion_wall, wall_seconds=time.perf_counter()-started)


def summarize(rows):
    if [(row['seed'], row['frame']) for row in rows] != [(seed, 0) for seed in SEEDS]:
        raise ValueError('Require all128 frozen reused frames in seed order')
    keys = ('matched_copies', 'misses', 'false_copies', 'output_copies', 'multiplicity_capable_oracle_bank_miss_floor')
    totals = {name: {key: sum(row['metrics'][name][key] for row in rows) for key in keys} for name in MODES}
    gains = [row['metrics']['conditional_fourth']['misses']-row['metrics']['count_growth']['misses'] for row in rows]
    fits = [fit for row in rows for arm in row['arms'].values()
            for fit in [arm['meta']['initial_refit'], *[refit['meta'] for refit in arm['meta']['refits']]]]
    for row in rows:
        for arm in row['arms'].values():
            meta = arm['meta']
            assert len(meta['queries']) == meta['actual_scl_calls']+meta['cache_hits']
            assert meta['cache_hits'] == sum(query['cache_hit'] for query in meta['queries'])
            assert math.isfinite(arm['wall_seconds']) and arm['wall_seconds'] > 0.
    gates = dict(four_net_saved_copies=sum(gains) >= 4, two_improved_frames=sum(g > 0 for g in gains) >= 2,
        net_without_best_frame=sum(gains)-max(gains) >= 1, no_frame_regression_over_two=min(gains) >= -2,
        all128_complete=all(arm['meta']['complete'] for row in rows for arm in row['arms'].values()),
        all_refits_converged=all(fit['converged'] for fit in fits),
        all_legal_actions=all(all(row['legal_actions'].values()) for row in rows),
        physical_inputs_and_reference_replay=all(row['reference_matches_archive'] and all(row['inputs_unchanged'].values())
                                               for row in rows),
        identical_first_four=all(row['prefix_checks']['identical'] for row in rows),
        integer_bounds=all(row['integer_bounds_verified'] for row in rows))
    return dict(frames=len(rows), passed=all(gates.values()), gates=gates, totals=totals,
        net_misses_saved=sum(gains), best_frame_gain=max(gains), net_without_best_frame=sum(gains)-max(gains),
        maximum_frame_regression=max(0, -min(gains)), better_frames=sum(g > 0 for g in gains),
        worse_frames=sum(g < 0 for g in gains), tied_frames=gains.count(0),
        extended_frames=[row['seed'] for row in rows if row['prefix_checks']['continued']],
        new_continuation_and_completion_seconds={name: sum(row['arms'][name]['wall_seconds'] for row in rows) for name in MODES},
        continuation_internal_seconds={name: sum(row['arms'][name]['meta']['seconds'] for row in rows) for name in MODES},
        completion_wall_seconds={name: sum(row['arms'][name]['completion_wall_seconds'] for row in rows) for name in MODES},
        logical_queries={name: sum(len(row['arms'][name]['meta']['queries']) for row in rows) for name in MODES},
        actual_scl_calls={name: sum(row['arms'][name]['meta']['actual_scl_calls'] for row in rows) for name in MODES},
        cache_hits={name: sum(row['arms'][name]['meta']['cache_hits'] for row in rows) for name in MODES},
        completed_blocks={name: sum(len(row['arms'][name]['meta']['sweeps']) for row in rows) for name in MODES},
        later_blocks=sum(row['arms']['count_growth']['meta']['later_sweeps'] for row in rows),
        geometry_seconds=sum(row['geometry_seconds'] for row in rows), source_seconds=sum(row['source_seconds'] for row in rows),
        verification_seconds=sum(row['verification_seconds'] for row in rows),
        historical_initialization_components={key: sum(row['historical_initialization'][key] for row in rows)
            for key in ('native_seconds', 'rollback_seconds', 'native_preparation_seconds')},
        scope='All128 reused physical frames; paired newly timed cold continuations plus terminal completion only. '
              'Geometry, source generation and historical initialization are separate. No mixed-time complete receiver claim or cost gate.')


def main(out):
    started = time.perf_counter()
    manifest, rows, serialization_seconds = freeze(out), [], 0.
    for index, seed in enumerate(SEEDS):
        saved = json.loads(saved_path(seed).read_text())
        tick = time.perf_counter()
        model, balance = geometry(seed)
        geometry_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        scenario, labels = make_frame(model, seed, 0)
        source_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        before = input_snapshot(scenario)
        assert before == saved['input_snapshot'] and labels == saved['labels']
        initial, bank, historical = saved_initialization(saved)
        initial_before, bank_before = initial.copy(), bank.copy()
        original_model, original_code = scenario.model, scenario.model.code
        verification_seconds = time.perf_counter()-tick
        order = MODES if index % 2 == 0 else tuple(reversed(MODES))
        arms, unchanged, legal = {}, {}, {}
        for name in order:
            def progress(record):
                print(json.dumps(dict(seed=seed, arm=name, stage='continuation', **record)), flush=True)
            tick = time.perf_counter()
            arm = run_arm(scenario, initial, bank, name, progress)
            wall = time.perf_counter()-tick
            arm['internal_wall_seconds'], arm['wall_seconds'] = arm['wall_seconds'], wall
            arms[name] = arm
            tick = time.perf_counter()
            unchanged[name] = (input_snapshot(scenario) == before and scenario.model is original_model
                and scenario.model.code is original_code and initial == initial_before and bank == bank_before)
            assert unchanged[name]
            legal[name] = valid_action(scenario, arm['action']) and valid_action(scenario, arm['cancellation_action'])
            verification_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        verify_reference(arms['conditional_fourth'], saved)
        prefix = verify_prefix(arms['conditional_fourth'], arms['count_growth'], bank, int(scenario.k))
        bounds = verify_growth(arms['count_growth'], int(scenario.k))
        metrics = {name: evaluate(Counter(dict(arm['action'])), arm['bank'], labels, scenario.k) for name, arm in arms.items()}
        cancellation = {name: evaluate(Counter(dict(arm['cancellation_action'])), arm['bank'], labels, scenario.k)
                        for name, arm in arms.items()}
        verification_seconds += time.perf_counter()-tick
        row = dict(seed=seed, frame=0, labels=labels, balance=balance, order=list(order), arms=arms, metrics=metrics,
            cancellation_metrics=cancellation, input_snapshot=before, initial_action=sorted(initial.items()), initial_bank=sorted(bank),
            saved_source=str(saved_path(seed).relative_to(ROOT)), historical_initialization=historical,
            inputs_unchanged=unchanged, legal_actions=legal, reference_matches_archive=True, prefix_checks=prefix,
            integer_bounds_verified=bounds, geometry_seconds=geometry_seconds, source_seconds=source_seconds,
            verification_seconds=verification_seconds)
        rows.append(row)
        tick = time.perf_counter()
        write_json(out/f'frame_{seed}.json', row)
        serialization_seconds += time.perf_counter()-tick
        print(json.dumps(dict(seed=seed, stage='frame_complete', extended=prefix['continued'],
            misses={name: metrics[name]['misses'] for name in MODES}, elapsed_seconds=time.perf_counter()-started)), flush=True)
    assert source_hashes() == manifest['source_sha256'] and evidence_hashes() == manifest['evidence_sha256']
    assert physical_input_hashes() == manifest['physical_input_sha256']
    assert digest(BASE/'design.json') == manifest['design_sha256'] and digest(BASE/'pre_main.json') == manifest['preflight_sha256']
    result = summarize(rows)
    result.update(sources_and_inputs_unchanged=True, elapsed_seconds=time.perf_counter()-started,
        serialization_seconds=serialization_seconds, source_freeze_sha256=digest(out/'source_freeze.json'),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
    write_json(out/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
