"""Frozen one-slot terminal rescue on the audited 128 development endpoints."""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.polar_count_feedback import support_completion
from framework.polar_folded_terminal import terminal_query
from diagnose_full_pattern import digest, evaluate, write_json
from diagnose_native_rate_zero import input_snapshot
from rollback_frames import geometry, make_frame

BASE = ROOT/'results/polar_dec35_exploration_2026-11-11'
PHASE = BASE/'algebra/folded_terminal'
SEEDS = range(7101, 7229)
MODES = ('magnitude', 'folded')


def freeze(out):
    read = lambda path: json.loads(path.read_text())
    audit = read(BASE/'count_growth/audit/outcomes.json')
    assert audit['passed']
    previous = BASE/'count_growth/main/source_freeze.json'
    assert digest(previous) == audit['input_sha256'][str(previous.relative_to(ROOT))]
    inherited = read(previous)['source_sha256']
    assert all(digest(ROOT/name) == checksum for name, checksum in inherited.items())
    sources = list(inherited)
    sources += ['framework/polar_folded_terminal.py', 'jobs/037_polar_tractable/diagnose_folded_terminal.py',
                'tests/polar_folded_terminal_test.py']
    evidence = [previous, PHASE/'design.json', PHASE/'finite_final.log', PHASE/'independent_source_review.json', BASE/'count_growth/audit/outcomes.json',
                BASE/'count_growth/main/summary.json']
    frames = [BASE/'count_growth/main'/f'frame_{seed}.json' for seed in SEEDS]
    for path in frames:
        assert digest(path) == audit['input_sha256'][str(path.relative_to(ROOT))]
    assert digest(PHASE/'design.json') == '4f88162a23bfde6176aa123c34af51aa10ffb21cd9c7cfdce57597dec3aaed75'
    assert (PHASE/'finite_final.log').read_text().rstrip().endswith('OK')
    manifest = {name:digest(ROOT/name) for name in sources}
    manifest.update({str(path.relative_to(ROOT)):digest(path) for path in evidence+frames})
    out.mkdir(parents=True, exist_ok=False)
    for name in sources:
        path = out/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
    shutil.copyfile(PHASE/'design.json', out/'design.json')
    write_json(out/'freeze.json', manifest)
    return manifest


def metrics(action, bank, truth):
    return {key:v for key,v in evaluate(action, bank, truth, len(truth)).items() if key != 'evaluation_seconds'}


def one_frame(seed, index):
    saved = json.loads((BASE/'count_growth/main'/f'frame_{seed}.json').read_text())
    old = saved['arms']['count_growth']
    assert (saved['seed'], saved['frame']) == (seed, 0) and old['meta']['complete']
    cancellation, bank = Counter(dict(old['cancellation_action'])), set(old['bank'])
    labels = saved['labels']
    baseline = Counter(dict(old['action']))
    reference = metrics(baseline, bank, labels)
    assert all(v == saved['metrics']['count_growth'][k] for k,v in reference.items())
    eligible = len(labels) - cancellation.total() == 1
    record = dict(seed=seed, eligible=eligible, fitted_count=cancellation.total(), reference=reference, arms={},
        repeated_messages=len(labels)-len(set(labels)), repeated_patterns=len(labels)-len({w >> 87 for w in labels}))
    if not eligible:
        return record
    tick = time.perf_counter()
    model, _ = geometry(seed)
    record['geometry_seconds'] = time.perf_counter() - tick
    tick = time.perf_counter()
    scenario, regenerated = make_frame(model, seed, 0)
    assert regenerated == labels and input_snapshot(scenario) == saved['input_snapshot']
    assert support_completion(scenario, cancellation, bank)[0] == baseline
    record['source_and_reference_verification_seconds'] = time.perf_counter() - tick
    record['order'] = list(MODES if index % 2 == 0 else reversed(MODES))
    for mode in record['order']:
        tick = time.perf_counter()
        output, trace = terminal_query(scenario, cancellation, bank, mode)
        wall_seconds = time.perf_counter() - tick
        assert trace['fit'] is None or trace['fit']['converged']
        assert all(isinstance(v, int) and v > 0 for v in output.values()) and output.total() <= scenario.k
        assert input_snapshot(scenario) == saved['input_snapshot']
        assert all(model.split(w)[0] == trace['pattern'] for w in trace['query_ids'])
        final_bank = bank.union(trace['query_ids'])
        assert set(trace['new_ids']) == final_bank-bank and set(output) <= final_bank
        truth = Counter(labels)
        uncovered = truth - cancellation
        record['arms'][mode] = dict(trace=trace, action=sorted(output.items()), wall_seconds=wall_seconds,
            metrics=metrics(output, final_bank, labels), retained_new_true=sorted(set(output).intersection(trace['new_ids'], truth)),
            retained_new_false=sorted(set(output).intersection(trace['new_ids']).difference(truth)),
            remaining_true_copies_on_queried_pattern=sum(v for w,v in uncovered.items() if model.split(w)[0]==trace['pattern']))
    record['correct_cancellation'] = not (cancellation-Counter(labels))
    return record


def summarize(rows):
    assert [row['seed'] for row in rows] == list(SEEDS)
    eligible = [row for row in rows if row['eligible']]
    def arm_metrics(row, name):
        return row['reference'] if name == 'reference' or not row['eligible'] else row['arms'][name]['metrics']
    def diffs(first, second):
        return [arm_metrics(row, first)['misses']-arm_metrics(row, second)['misses'] for row in rows]
    totals = {name:{key:sum(arm_metrics(row,name)[key] for row in rows) for key in
        ('misses','matched_copies','false_copies','output_copies','multiplicity_capable_oracle_bank_miss_floor')}
        for name in ('reference', *MODES)}
    def contrast(gains):
        return dict(saved=sum(gains), better=sum(g>0 for g in gains), tied=gains.count(0), worse=sum(g<0 for g in gains),
            net_without_best=sum(gains)-max(gains), max_regression=max(0,-min(gains)),
            changed_frames=[(r['seed'],g) for r,g in zip(rows,gains,strict=True) if g])
    performance = {name:contrast(diffs('reference',name)) for name in MODES}
    attribution = contrast(diffs(*MODES))
    exclusive = [r['seed'] for r in eligible if r['arms']['folded']['trace']['pattern']!=r['arms']['magnitude']['trace']['pattern']
        and set(r['arms']['folded']['retained_new_true'])-set(r['arms']['magnitude']['retained_new_true'])]
    gates = {name: dict(four_net=c['saved']>=4, three_frames=c['better']>=3, leave_best_two=c['net_without_best']>=2,
        max_regression_one=c['max_regression']<=1) for name,c in performance.items()}
    gates['specific'] = dict(two_net=attribution['saved']>=2, two_frames=attribution['better']>=2, novel_different_pattern=bool(exclusive))
    return dict(frames=len(rows), repeated_messages=sum(r['repeated_messages'] for r in rows),
        repeated_patterns=sum(r['repeated_patterns'] for r in rows), eligible=len(eligible), correct_cancellation=sum(r['correct_cancellation'] for r in eligible),
        totals=totals, vs_reference=performance, folded_vs_magnitude=attribution, exclusive_different_pattern_seeds=exclusive,
        gates=gates, could_advance={'magnitude':all(gates['magnitude'].values()),
        'folded':all(gates['folded'].values()) and all(gates['specific'].values())},
        same_selection=sum(r['arms'][MODES[0]]['trace']['pattern']==r['arms'][MODES[1]]['trace']['pattern'] for r in eligible),
        costs={name:{**{k:sum(r['arms'][name]['trace'][k] for r in eligible) for k in
            ('setup_seconds','ranking_seconds','search_seconds','refit_completion_seconds')},
            'wall_seconds':sum(r['arms'][name]['wall_seconds'] for r in eligible)} for name in MODES},
        geometry_seconds=sum(r['geometry_seconds'] for r in eligible),
        source_and_reference_verification_seconds=sum(r['source_and_reference_verification_seconds'] for r in eligible),
        scope='All128 reused, separately charged terminal extensions. Ineligible endpoints use audited saved outputs. '
              'Only eligible physical frames regenerated. Cold complete receiver costs require a fresh comparison.')


def main(out):
    start = time.perf_counter()
    frozen = freeze(out)
    rows = []
    for index,seed in enumerate(SEEDS):
        row = one_frame(seed, index)
        rows.append(row)
        write_json(out/f'frame_{seed}.json', row)
        if row['eligible']:
            print(seed, 'terminal queries complete', flush=True)
    result = summarize(rows)
    assert all(digest(ROOT/name) == checksum for name,checksum in frozen.items())
    result.update(study_seconds=time.perf_counter()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
        (1 if sys.platform=='darwin' else 1024), source_freeze_sha256=digest(out/'freeze.json'))
    write_json(out/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
