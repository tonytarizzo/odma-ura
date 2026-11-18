"""Fresh paired DEC-35 query-reuse and decision-only support-completion test."""

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
from framework.polar_count_cavity import continue_cavity
from framework.polar_count_feedback import continue_feedback, support_completion
from rollback_frames import geometry, make_frame, native
from rollback_progressive import progressive
from diagnose_full_pattern import digest, evaluate, write_json

BASE = ROOT / 'results/polar_count_feedback_2026-11-11'
SEEDS = list(range(6601, 6633))
SOURCES = ['baselines/polar.py', 'baselines/odma_polar.py', 'benchmarks/polar_energy_geometry.py',
    'framework/polar_tractable.py', 'framework/polar_full_pattern.py', 'framework/polar_count_cavity.py',
    'framework/polar_count_feedback.py', 'jobs/037_polar_tractable/rollback_frames.py',
    'jobs/037_polar_tractable/rollback_progressive.py', 'jobs/037_polar_tractable/rollback_rescue.py',
    'jobs/037_polar_tractable/diagnose_full_pattern.py', 'jobs/037_polar_tractable/diagnose_count_feedback.py']


def trace_signature(meta):
    keys = ('query', 'sweep', 'pattern', 'list_size', 'context', 'restored', 'restored_count', 'outside_count',
            'pending', 'variance', 'pre_action_sha256', 'llr_sha256', 'ids', 'new_ids', 'refit_index')
    return dict(initial=meta['initial_refit_action'], initial_moves=meta['initial_refit']['moves'],
        queries=[{k:q[k] for k in keys} for q in meta['queries']],
        refits=[dict(query=r['query'], new_ids=r['new_ids'], moves=r['meta']['moves']) for r in meta['refits']],
        sweeps=[{k:r[k] for k in ('sweep', 'ranked_patterns', 'queries', 'new_ids', 'moves', 'output_count',
                                 'bank_size', 'complete_sweep')} for r in meta['sweeps']],
        complete=meta['complete'], stop=meta['stop'])


def summarize(rows):
    if [(r['seed'], r['frame']) for r in rows] != [(seed, 0) for seed in SEEDS]:
        raise ValueError('Require all32 frozen frames in order')
    names = ('rollback', 'reference', 'reuse', 'completed')
    totals = {a:{key:sum(r['metrics'][a][key] for r in rows) for key in
        ('matched_copies', 'misses', 'false_copies', 'output_copies', 'multiplicity_capable_oracle_bank_miss_floor')}
        for a in names}
    costs = {a:sum(r['receiver_with_geometry_seconds'][a] for r in rows) for a in names}
    continuation = {a:sum(r['arms'][a]['wall_seconds'] for r in rows) for a in ('reference', 'reuse')}
    calls = sum(len(r['arms']['reference']['meta']['queries']) for r in rows)
    hits = sum(r['arms']['reuse']['meta']['cache_hits'] for r in rows)
    actual_calls = sum(r['arms']['reuse']['meta']['actual_scl_calls'] for r in rows)
    for row in rows:
        reused = row['arms']['reuse']['meta']
        if (reused['actual_scl_calls'] + reused['cache_hits'] != len(reused['queries'])
                or len(reused['queries']) != len(row['arms']['reference']['meta']['queries'])):
            raise ValueError('Physical-call, cache-hit and logical-query counters disagree')
    cache_frames = sum(r['arms']['reuse']['meta']['cache_hits'] > 0 for r in rows)
    cache_bytes = max(r['arms']['reuse']['meta']['cache_bytes'] for r in rows)
    gains = [r['metrics']['reference']['misses'] - r['metrics']['completed']['misses'] for r in rows]
    complete = all(r['arms'][a]['meta']['complete'] for r in rows for a in ('reference', 'reuse'))
    reuse_gate = dict(all_complete=complete, identical_traces=all(r['identical_trace'] for r in rows),
        ten_percent_fewer_calls=actual_calls <= .9 * calls, hits_on_four_frames=cache_frames >= 4,
        ten_percent_faster_continuation=continuation['reuse'] <= .9 * continuation['reference'],
        one_percent_faster_complete_receiver=costs['reuse'] <= .99 * costs['reference'],
        cache_within_eight_MiB=cache_bytes <= 8 * 1024**2)
    completion_seconds = sum(r['completion_wall_seconds'] for r in rows)
    completion_gate = dict(all_complete=complete, two_saved_copies=sum(gains) >= 2,
        two_improved_frames=sum(g > 0 for g in gains) >= 2, no_miss_regression=min(gains) >= 0,
        preserves_counts_and_capacity=all(r['completion_preserves_counts_and_cap'] for r in rows),
        added_time_within_five_percent=completion_seconds <= .05 * costs['reference'])
    return dict(totals=totals, receiver_with_geometry_seconds=costs, continuation_seconds=continuation,
        cache=dict(logical_queries=calls, hits=hits, actual_calls=actual_calls, hit_frames=cache_frames,
                   maximum_bytes=cache_bytes), completion_seconds=completion_seconds, paired_completion_gains=gains,
        completion_false_copy_change=totals['completed']['false_copies']-totals['reference']['false_copies'],
        reuse_gate=reuse_gate, reuse_passed=all(reuse_gate.values()), completion_gate=completion_gate,
        completion_passed=all(completion_gate.values()),
        scope='Independent cost and terminal count-PUPE tests; no new discovery mechanism or false-copy dominance')


def freeze(out):
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    design = json.loads((BASE / 'design.json').read_text())
    checked = json.loads((BASE / 'audit/pre_main.json').read_text())
    hashes = {p:digest(ROOT / p) for p in SOURCES}
    if not checked['passed'] or checked['design_sha256'] != digest(BASE / 'design.json'):
        raise ValueError('Require a passing independent preflight of this design')
    if checked['source_sha256'] != hashes or design['panel']['seeds'] != SEEDS:
        raise ValueError('Sources or panel differ from independent preflight')
    out.mkdir(parents=True)
    (out / 'design.json').write_bytes((BASE / 'design.json').read_bytes())
    for relative in SOURCES:
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), sources=hashes,
        design_sha256=digest(out / 'design.json'), pre_main_sha256=digest(BASE / 'audit/pre_main.json'))
    write_json(out / 'source_freeze.json', manifest)
    return manifest


def main(out):
    started = time.perf_counter()
    manifest = freeze(out)
    rows = []
    for seed in SEEDS:
        tick = time.perf_counter()
        model, balance = geometry(seed)
        geometry_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        scenario, labels = make_frame(model, seed, 0)
        source_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        accepted, native_meta = native(scenario)
        initial, found, batches, _ = progressive(scenario, list(accepted), native_meta['valid_ids'], 'ungated')
        initialization_seconds = time.perf_counter() - tick
        base_bank = set(native_meta['valid_ids']).union(found, initial)
        arms = {}
        for mode in (('reference', 'reuse') if seed % 2 else ('reuse', 'reference')):
            tick = time.perf_counter()
            if mode == 'reference':
                action, bank, meta = continue_cavity(scenario, initial, base_bank, mode='raw128')
            else:
                action, bank, meta = continue_feedback(scenario, initial, base_bank, memoize=True)
            arms[mode] = dict(action=sorted(action.items()), bank=sorted(bank), meta=meta,
                              wall_seconds=time.perf_counter()-tick)
        identical = (arms['reference']['action'] == arms['reuse']['action']
            and arms['reference']['bank'] == arms['reuse']['bank']
            and trace_signature(arms['reference']['meta']) == trace_signature(arms['reuse']['meta']))
        if not identical:
            raise AssertionError(f'Query reuse changed the complete receiver trajectory on seed{seed}')
        original = Counter(dict(arms['reuse']['action']))
        tick = time.perf_counter()
        completed, completion_meta = support_completion(scenario, original, arms['reuse']['bank'])
        completion_seconds = time.perf_counter()-tick
        metrics = dict(rollback=evaluate(initial, base_bank, labels, scenario.k))
        metrics.update({a:evaluate(Counter(dict(r['action'])), r['bank'], labels, scenario.k) for a,r in arms.items()})
        metrics['completed'] = evaluate(completed, arms['reuse']['bank'], labels, scenario.k)
        costs = dict(rollback=geometry_seconds+initialization_seconds)
        costs.update({a:costs['rollback']+r['wall_seconds'] for a,r in arms.items()})
        costs['completed'] = costs['reuse']+completion_seconds
        row = dict(seed=seed, frame=0, labels=labels, balance=balance, geometry_seconds=geometry_seconds,
            source_seconds=source_seconds, initialization_seconds=initialization_seconds,
            positions_sha256=hashlib.sha256(model.positions.tobytes()).hexdigest(),
            amplitudes_sha256=hashlib.sha256(model.amplitudes.tobytes()).hexdigest(),
            y_sha256=hashlib.sha256(scenario.y.tobytes()).hexdigest(), native_accepted=list(accepted),
            native_meta=native_meta, progressive_found=sorted(found), rollback_batches=batches,
            rollback_action=sorted(initial.items()), initial_bank=sorted(base_bank), arms=arms,
            completed_action=sorted(completed.items()), completion_meta=completion_meta,
            completion_wall_seconds=completion_seconds, metrics=metrics, receiver_with_geometry_seconds=costs,
            identical_trace=identical, completion_preserves_counts_and_cap=(completed.total() <= scenario.k
                and all(completed[w] >= count for w,count in original.items())))
        write_json(out / f'frame_{seed}.json', row)
        rows.append(row)
        print(json.dumps(dict(seed=seed, misses={a:r['misses'] for a,r in metrics.items()},
            cache_hits=arms['reuse']['meta']['cache_hits'], elapsed_seconds=time.perf_counter()-started)), flush=True)
    result = summarize(rows)
    if any(digest(ROOT / p) != h for p,h in manifest['sources'].items()):
        raise AssertionError('Executed source changed after freeze')
    result.update(seeds=SEEDS, all_sources_unchanged=True, elapsed_seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        source_freeze_sha256=digest(out / 'source_freeze.json'))
    write_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
