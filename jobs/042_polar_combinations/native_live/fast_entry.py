"""Qualify the public DEC-38 repetition-tail execution on cold and roundoff-scale physical inputs."""
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2])]
from framework.polar_conditional_repetition import run as fast
from framework.polar_catalogue import run as catalogue
from framework.polar_tractable import PolarScenario
from tests.polar_count_cavity_test import smoke_scenario
from frames import ROOT, digest, frame, write


def compare(scenario, policy, *, fresh=None, order=('old', 'fast'), geometry=0.):
    raw = hashlib.sha256(scenario.y.tobytes()).hexdigest()
    original = scenario.model.code
    outputs, signatures, times = {}, {}, {}
    for arm in order:
        cpu = time.process_time()
        if arm == 'old':
            action, meta = catalogue('DEC-38', scenario, terminal_policy=policy)
        else:
            paths = list(sys.path)
            action, meta = fast(scenario, terminal_policy=policy)
            assert paths == sys.path and meta['execution'] == 'repetition_tail'
        times[arm] = dict(all_in_wall=meta['total_seconds'] + geometry, receiver_cpu=time.process_time() - cpu)
        assert scenario.model.code is original and hashlib.sha256(scenario.y.tobytes()).hexdigest() == raw and meta['complete']
        assert all(type(w) is int and 0 <= w < 2**scenario.model.B and type(v) is int and v > 0 for w, v in action.items())
        assert sum(action.values()) <= scenario.k
        outputs[arm] = action
        signatures[arm] = {k: meta.get(k) for k in ('native_accepted', 'rollback_action', 'base_bank', 'bank', 'cancellation_action',
                                                     'terminal_queries', 'rate_challenge')}
        signatures[arm].update(rolls=meta['native']['rounds'], valid=meta['native']['valid_ids'], residual=meta['native']['residual_sha256'],
                               native_calls=meta['native']['calls'], continuation_calls=meta['continuation']['actual_scl_calls'],
                               rollback_batches=meta['rollback_batches'], output=action)
    different = [k for k, v in signatures['old'].items() if v != signatures['fast'][k]]
    counts = {k: sorted(v.items()) for k, v in outputs.items()}
    return dict(y=raw, policy=policy, different=different, counts=counts, times=times,
                errors=None if fresh is None else {k: (Counter(fresh)-Counter(a)).total() for k, a in outputs.items()},
                native=signatures['fast']['native_calls'], repair=signatures['fast']['continuation_calls'])


def stress():
    scenario, truth = smoke_scenario()
    truth[210] += 1
    repeat = PolarScenario(scenario.model, scenario.y + scenario.model.waveform(210), 4, scenario.noise_var)
    m = scenario.model
    left = m.waveform(209) + 2*m.waveform(1258)
    right = m.waveform(210) + 2*m.waveform(1258)
    mid = (left + right)/2
    results = []
    for policy in ('two_raw', 'scan_rate'):
        r = compare(repeat, policy)
        assert not r['different'] and dict(r['counts']['fast']) == dict(truth)
        results.append(dict(case='repeat', **r))
        for direction in (-1, 1):
            y = mid + direction * np.finfo(float).eps * (left - right)
            variance = .03
            physical_gap = float(np.sum((y - right)**2 - (y - left)**2) / (2*variance))
            assert 0 < direction*physical_gap < 1e-12
            trial = PolarScenario(m, y, 3, variance)
            r = compare(trial, policy)
            results.append(dict(case=f'near_tie_{direction}', physical_loglik_gap=physical_gap, **r))
    return results


def main():
    out = ROOT/'results/dec42_study/native/fast_entry'
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), HERE/'fast_entry_design.md', ROOT/'framework/polar_conditional_repetition.py',
             ROOT/'framework/polar_conditional_serial.py', ROOT/'framework/polar_serial_repair.py', ROOT/'framework/polar_catalogue.py']
    write(out/'freeze.json', dict(seeds=[988007, 988008], sources={p.name: digest(p) for p in files}))
    ss = stress()
    write(out/'finite.json', ss)
    print([(r['case'], r['policy'], r['different']) for r in ss], flush=True)
    rows = []
    for seed, db in ((988007, .25), (988008, .5)):
        s, labels, geometry = frame(seed, db)
        for i, policy in enumerate(('two_raw', 'scan_rate')):
            order = ('old', 'fast') if (seed+i)%2 else ('fast', 'old')
            r = compare(s, policy, fresh=labels, order=order, geometry=geometry)
            row = dict(seed=seed, db=db, **r)
            write(out/f'{seed}_{policy}.json', row)
            print(seed, policy, r['different'], r['errors'], flush=True)
            assert not r['different']
            rows.append(row)
    write(out/'summary.json', dict(fresh_semantic_matches=len(rows), finite=ss, rows=rows))


if __name__ == '__main__':
    main()
