"""Fresh128 complete-receiver confirmation of a conditional fourth block and measured cost reinvestment."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.polar_count_feedback import run as original
from framework.polar_native_repetition import run as native_repetition
from framework.polar_progress_continuation import run as conditional_fourth
from rollback_frames import geometry, make_frame
from diagnose_full_pattern import digest, evaluate, write_json
from diagnose_native_rate_zero import NATIVE_KEYS, SOURCES as ORIGINAL_SOURCES, input_snapshot, semantic_signature
from diagnose_progress_continuation import normalized, verify_prefix

BASE = ROOT / 'results/polar_dec35_exploration_2026-11-11/progress_continuation/confirmation'
SEEDS = list(range(7101, 7229))
MODES = ('A', 'B', 'C')
ORDERS = list(permutations(MODES))
SOURCES = ORIGINAL_SOURCES + ['framework/polar_repetition_scl.py', 'framework/polar_native_repetition.py',
    'framework/polar_progress_continuation.py', 'tests/polar_progress_continuation_test.py',
    'tests/polar_progress_receiver_test.py', 'tests/polar_progress_confirmation_test.py',
    'jobs/037_polar_tractable/diagnose_progress_continuation.py', 'jobs/037_polar_tractable/diagnose_progress_confirmation.py']


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def evidence_hashes():
    development = BASE.parent / 'development'
    repetition = BASE.parents[1] / 'repetition'
    paths = [development / name for name in ('summary.json', 'source_freeze.json')]
    paths += [BASE.parent / 'audit' / name for name in ('outcomes.py', 'outcomes.json')]
    paths += [BASE.parent / name for name in ('design.json', 'pre_main.json')]
    paths += [repetition / 'finite_full.json', repetition / 'receiver/audit_review.json',
        repetition / 'receiver/main/summary.json', repetition / 'receiver/main/source_freeze.json']
    paths += [BASE / name for name in ('design_initial.json', 'formatting_revision.json')]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def freeze(out):
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    design = json.loads((BASE / 'design.json').read_text())
    checked = json.loads((BASE / 'pre_main.json').read_text())
    sources, evidence = source_hashes(), evidence_hashes()
    if (checked.get('passed') is not True or checked['design_sha256'] != digest(BASE / 'design.json')
            or checked['source_sha256'] != sources or checked['evidence_sha256'] != evidence):
        raise ValueError('Require independent preflight binding this design, sources and audited development evidence')
    assert design['panel']['seeds'] == SEEDS and design['panel']['frame'] == 0
    expected = dict(minimum_net_missed_copies_saved_C_vs_B=4, minimum_better_frames_C_vs_B=2,
        minimum_net_save_after_removing_best_frame=1, maximum_individual_miss_regression=2)
    assert all(design['gates'][key] == value for key, value in expected.items())
    assert all(design['reinvestment_gates'].values())
    assert json.loads((BASE.parent / 'development/summary.json').read_text())['advance']
    assert json.loads((BASE.parent / 'audit/outcomes.json').read_text())['passed']
    out.mkdir(parents=True)
    (out / 'design.json').write_bytes((BASE / 'design.json').read_bytes())
    for name in SOURCES:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, evidence_sha256=evidence,
        design_sha256=digest(BASE / 'design.json'), pre_main_sha256=digest(BASE / 'pre_main.json'))
    write_json(out / 'source_freeze.json', manifest)
    return manifest


def run_arm(scenario, mode):
    if mode == 'A':
        return original(scenario, memoize=True, complete_support=True)
    if mode == 'B':
        return native_repetition(scenario)
    if mode == 'C':
        return conditional_fourth(scenario)
    raise ValueError('Require one of the three frozen complete receiver arms')


def original_prefix_checks(arms):
    b, c = (arms[name]['meta'] for name in ('B', 'C'))
    initialization = all(normalized(b[key]) == normalized(c[key]) for key in
        ('rollback_action', 'rollback_batches', 'base_bank'))
    initialization = initialization and all(b['native'][key] == c['native'][key] for key in NATIVE_KEYS)
    b_cont, c_cont = b['continuation'], c['continuation']
    initial_fit_equal = ({key:value for key, value in b_cont['initial_refit'].items() if key != 'seconds'}
        == {key:value for key, value in c_cont['initial_refit'].items() if key != 'seconds'})
    fits_equal = len(c_cont['refits']) >= len(b_cont['refits']) and initial_fit_equal
    for left, right in zip(b_cont['refits'], c_cont['refits'][:len(b_cont['refits'])]):
        fits_equal = fits_equal and ({key:value for key, value in left['meta'].items() if key != 'seconds'}
            == {key:value for key, value in right['meta'].items() if key != 'seconds'})
    initialization = initialization and initial_fit_equal
    prefix_arms = []
    for name in ('B', 'C'):
        meta = arms[name]['meta']
        prefix_arms.append(dict(meta=meta['continuation'], action=arms[name]['action'],
            cancellation_action=meta['cancellation_action'], bank=meta['bank']))
    try:
        trigger = verify_prefix(*prefix_arms)
        prefix = bool(fits_equal)
    except AssertionError:
        trigger, prefix = None, False
    return dict(B_C_initialization=bool(initialization), B_C_original_prefix=prefix,
        B_C_count_fit_metadata=bool(fits_equal), trigger=trigger,
        scope='Recorded native aggregates and full continuation semantics; native ordered lists were not recorded.')


def valid_action(scenario, pairs):
    if len(dict(pairs)) != len(pairs):
        return False
    if any(type(word) is not int or type(count) is not int or count <= 0 for word, count in pairs):
        return False
    if sum(count for _, count in pairs) > scenario.k:
        return False
    try:
        for word, _ in pairs:
            scenario.model.split(word)
    except ValueError:
        return False
    return True


def summarize(rows):
    if [(row['seed'], row['frame']) for row in rows] != [(seed, 0) for seed in SEEDS]:
        raise ValueError('Require all128 predetermined frames in fixed order')
    keys = ('matched_copies', 'misses', 'false_copies', 'output_copies', 'multiplicity_capable_oracle_bank_miss_floor')
    totals = {name:{key:sum(row['metrics'][name][key] for row in rows) for key in keys} for name in MODES}
    costs = {name:sum(row['receiver_with_geometry_seconds'][name] for row in rows) for name in MODES}
    gains = [row['metrics']['B']['misses']-row['metrics']['C']['misses'] for row in rows]
    fits = [fit for row in rows for name in MODES for fit in [row['arms'][name]['meta']['continuation']['initial_refit'],
        *[refit['meta'] for refit in row['arms'][name]['meta']['continuation']['refits']]]]
    for row in rows:
        for name in MODES:
            meta = row['arms'][name]['meta']['continuation']
            if (len(meta['queries']) != meta['actual_scl_calls']+meta['cache_hits']
                    or meta['cache_hits'] != sum(query['cache_hit'] for query in meta['queries'])):
                raise ValueError('Inconsistent physical, logical and cached continuation counters')
    capability = dict(four_net_saved_copies=sum(gains) >= 4, two_better_frames=sum(g > 0 for g in gains) >= 2,
        net_without_best_frame=sum(gains)-max(gains) >= 1, no_individual_regression_over_two=min(gains) >= -2,
        all_complete=all(row['arms'][name]['meta']['complete'] for row in rows for name in MODES),
        all_refits_converged=all(fit['converged'] for fit in fits),
        all_output_counts_legal=all(all(row['legal_actions'].values()) for row in rows),
        inputs_unchanged=all(all(row['original_inputs_unchanged'].values()) for row in rows),
        B_C_initialization_and_original_prefix=all(row['prefix_checks']['B_C_initialization']
            and row['prefix_checks']['B_C_original_prefix'] for row in rows))
    reinvestment = dict(C_cost_at_most_A=costs['C'] <= costs['A'], C_misses_at_most_A=totals['C']['misses'] <= totals['A']['misses'])
    attribution = dict(A_B_native_aggregates=all(row['A_B_native_aggregates'] for row in rows),
        A_B_complete_semantics=all(row['A_B_complete_semantics'] for row in rows),
        A_B_output_counts=all(row['arms']['A']['action'] == row['arms']['B']['action'] for row in rows))
    stages = {name:dict(native_seconds=sum(row['arms'][name]['meta']['native']['seconds'] for row in rows),
        rollback_seconds=sum(row['arms'][name]['meta']['rollback_seconds'] for row in rows),
        continuation_seconds=sum(row['arms'][name]['meta']['continuation']['seconds'] for row in rows),
        support_completion_seconds=sum(row['arms'][name]['meta']['support_completion']['seconds'] for row in rows),
        native_preparation_seconds=sum(row['arms'][name]['meta'].get('native_repetition_preparation_seconds', 0.) for row in rows),
        complete_wall_seconds=sum(row['arms'][name]['wall_seconds'] for row in rows)) for name in MODES}
    return dict(frames=len(rows), totals=totals, capability_gate=capability, capability_passed=all(capability.values()),
        reinvestment_gate=reinvestment, reinvestment_passed=all(reinvestment.values()),
        attribution_gate=attribution, pure_acceleration_attribution_passed=all(attribution.values()),
        passed=all(capability.values()) and all(reinvestment.values()),
        net_misses_saved_C_vs_B=sum(gains), net_without_best_frame=sum(gains)-max(gains), largest_frame_gain=max(gains),
        maximum_frame_regression=max(0, -min(gains)), better_frames=sum(g > 0 for g in gains),
        worse_frames=sum(g < 0 for g in gains), tied_frames=gains.count(0),
        triggered_frames=[row['seed'] for row in rows if row['arms']['C']['meta']['continuation']['extra_sweep_triggered']],
        receiver_with_geometry_seconds=costs, C_over_A_cost_ratio=costs['C']/costs['A'], C_over_B_cost_ratio=costs['C']/costs['B'],
        C_minus_B_seconds=costs['C']-costs['B'], geometry_seconds=sum(row['geometry_seconds'] for row in rows), stage_seconds=stages,
        source_seconds=sum(row['source_seconds'] for row in rows), verification_seconds=sum(row['verification_seconds'] for row in rows),
        logical_queries={name:sum(len(row['arms'][name]['meta']['continuation']['queries']) for row in rows) for name in MODES},
        actual_continuation_scl_calls={name:sum(row['arms'][name]['meta']['continuation']['actual_scl_calls'] for row in rows) for name in MODES},
        cache_hits={name:sum(row['arms'][name]['meta']['continuation']['cache_hits'] for row in rows) for name in MODES},
        retained_true_IDs_absent_B_bank=sorted(set().union(*(set(row['exclusive_retained']) for row in rows))),
        source_repeated_copies=sum(row['source_repeated_copies'] for row in rows),
        source_pattern_collision_copies=sum(row['source_pattern_collision_copies'] for row in rows),
        scope='Fresh complete original-system comparison at one fixed configuration. Capability, measured reinvestment '
              'and empirical A/B acceleration attribution are separate. No native ordered-list or universal float-equivalence claim.')


def main(out):
    started = time.perf_counter()
    manifest = freeze(out)
    rows = []
    for index, seed in enumerate(SEEDS):
        tick = time.perf_counter()
        model, balance = geometry(seed)
        geometry_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        scenario, labels = make_frame(model, seed, 0)
        source_seconds = time.perf_counter()-tick
        original_model, original_code = scenario.model, scenario.model.code
        tick = time.perf_counter()
        before = input_snapshot(scenario)
        verification_seconds = time.perf_counter()-tick
        arms, unchanged, legal = {}, {}, {}
        order = ORDERS[index % len(ORDERS)]
        for name in order:
            tick = time.perf_counter()
            counts, meta = run_arm(scenario, name)
            wall = time.perf_counter()-tick
            arms[name] = dict(action=sorted(counts.items()), meta=meta, wall_seconds=wall)
            tick = time.perf_counter()
            unchanged[name] = (scenario.model is original_model and scenario.model.code is original_code
                and input_snapshot(scenario) == before)
            legal[name] = valid_action(scenario, arms[name]['action']) and valid_action(scenario, meta['cancellation_action'])
            verification_seconds += time.perf_counter()-tick
            print(json.dumps(dict(seed=seed, stage='complete_arm', arm=name, seconds=wall)), flush=True)
        tick = time.perf_counter()
        a, b = (arms[name]['meta'] for name in ('A', 'B'))
        prefix = original_prefix_checks(arms)
        native_same = all(a['native'][key] == b['native'][key] for key in NATIVE_KEYS)
        semantic_same = normalized(semantic_signature(a)) == normalized(semantic_signature(b))
        metrics = {name:evaluate(Counter(dict(arm['action'])), arm['meta']['bank'], labels, scenario.k) for name, arm in arms.items()}
        cancellation = {name:evaluate(Counter(dict(arm['meta']['cancellation_action'])), arm['meta']['bank'], labels, scenario.k)
            for name, arm in arms.items()}
        retained = set(dict(arms['C']['action'])) & set(labels)
        truth = Counter(labels)
        patterns = Counter(word >> model.b for word in labels)
        verification_seconds += time.perf_counter()-tick
        row = dict(seed=seed, frame=0, labels=labels, balance=balance, order=list(order), arms=arms, metrics=metrics,
            cancellation_metrics=cancellation, input_snapshot=before, original_inputs_unchanged=unchanged, legal_actions=legal,
            prefix_checks=prefix, A_B_native_aggregates=native_same, A_B_complete_semantics=semantic_same,
            geometry_seconds=geometry_seconds, source_seconds=source_seconds, verification_seconds=verification_seconds,
            receiver_with_geometry_seconds={name:geometry_seconds+arm['wall_seconds'] for name, arm in arms.items()},
            exclusive_retained=sorted(retained-set(arms['B']['meta']['bank'])),
            source_repeated_copies=sum(count-1 for count in truth.values()),
            source_pattern_collision_copies=sum(count-1 for count in patterns.values()))
        rows.append(row)
        write_json(out / f'frame_{seed}.json', row)
        print(json.dumps(dict(seed=seed, stage='frame_complete', misses={name:metrics[name]['misses'] for name in MODES},
            prefix_checks=prefix, elapsed_seconds=time.perf_counter()-started)), flush=True)
    assert source_hashes() == manifest['source_sha256'] and evidence_hashes() == manifest['evidence_sha256']
    assert digest(BASE / 'design.json') == manifest['design_sha256']
    result = summarize(rows)
    result.update(all_sources_and_evidence_unchanged=True, elapsed_seconds=time.perf_counter()-started,
        source_freeze_sha256=digest(out / 'source_freeze.json'),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    write_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
