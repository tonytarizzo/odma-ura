"""Cold DEC-35 prefix and paired standalone terminal policies on 128 fresh physical frames."""

import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from framework.polar_count_growth import run as count_growth
from framework.polar_terminal_query import finish, run as ordinary
from diagnose_folded_terminal import MODES, metrics
from diagnose_full_pattern import digest, write_json
from diagnose_native_rate_zero import input_snapshot
from diagnose_count_growth import semantic
from rollback_frames import geometry, make_frame

SEEDS=range(7501,7629)


def freeze(out):
    """Verify the released source tree. The archived frame and preflight checks were run at release."""
    manifest = Path(__file__).with_name('terminal_confirmation_inputs.json')
    release = json.loads(manifest.read_bytes())
    bound = dict(release['source_sha256'])
    for name, checksum in bound.items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Changed terminal confirmation input: {name}')
    bound[str(manifest.relative_to(ROOT))] = digest(manifest)
    out.mkdir(parents=True, exist_ok=False)
    for name in bound:
        destination = out/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    write_json(out/'freeze.json', bound)
    return bound


def summary(rows):
    if [r['seed'] for r in rows]!=list(SEEDS):
        raise ValueError('Fresh summary requires every declared seed in order')
    def values(row,name):
        return row['reference'] if name=='reference' or not row['eligible'] else row['arms'][name]['metrics']
    def contrast(first,second):
        gains=[values(r,first)['misses']-values(r,second)['misses'] for r in rows]
        return dict(saved=sum(gains),better=sum(g>0 for g in gains),worse=sum(g<0 for g in gains),tied=gains.count(0),
            leave_best=sum(gains)-max(gains),max_regression=max(0,-min(gains)),
            changed_frames=[(r['seed'],v) for r,v in zip(rows,gains,strict=True) if v])
    totals={name:{key:sum(values(r,name)[key] for r in rows) for key in
        ('misses','matched_copies','false_copies','output_copies','multiplicity_capable_oracle_bank_miss_floor')}
        for name in ('reference',*MODES)}
    vs={m:contrast('reference',m) for m in MODES}
    specific=contrast(*MODES)
    eligible=[r for r in rows if r['eligible']]
    exclusive=[r['seed'] for r in eligible if r['arms']['folded']['trace']['pattern']!=r['arms']['magnitude']['trace']['pattern']
        and set(r['arms']['folded']['retained_new_true'])-set(r['arms']['magnitude']['retained_new_true'])]
    gates={m:dict(four_net=c['saved']>=4,three_frames=c['better']>=3,leave_best_two=c['leave_best']>=2,
        max_regression_one=c['max_regression']<=1) for m,c in vs.items()}
    gates['specific']=dict(two_net=specific['saved']>=2,two_frames=specific['better']>=2,novel_different_pattern=bool(exclusive))
    result=dict(frames=len(rows),eligible=len(eligible),correct_cancellation=sum(r['correct_cancellation'] for r in eligible),
        totals=totals,vs_reference=vs,folded_vs_magnitude=specific,exclusive_different_pattern_seeds=exclusive,gates=gates,
        repeated_messages=sum(r['repeated_messages'] for r in rows),repeated_patterns=sum(r['repeated_patterns'] for r in rows),
        same_selection=sum(r['arms']['folded']['trace']['pattern']==r['arms']['magnitude']['trace']['pattern'] for r in eligible),
        passes={m:all(gates[m].values()) and (m=='magnitude' or all(gates['specific'].values())) for m in MODES},
        eligible_stage_seconds={m:{k:sum(r['arms'][m]['trace'][k] for r in eligible) for k in
        ('setup_seconds','ranking_seconds','search_seconds','refit_completion_seconds')} for m in MODES})
    result['geometry_seconds']=sum(r['geometry_seconds'] for r in rows)
    base=sum(r['base_seconds']+r['geometry_seconds'] for r in rows)
    result.update(common_cold_prefix_with_geometry_seconds=base, geometry_inclusive_standalone_seconds=dict(reference=base,
        **{m:base+sum(r['finish_seconds'][m] for r in rows) for m in MODES}),
        eligible_seeds=[r['seed'] for r in rows if r['eligible']],
        ordinary_replay_seeds=[r['seed'] for r in rows if r['ordinary_replay_seconds']],
        ordinary_replay_seconds=sum(sum(r['ordinary_replay_seconds'].values()) for r in rows),
        source_seconds=sum(r['source_seconds'] for r in rows),verification_seconds=sum(r['verification_seconds'] for r in rows),
        scope='Unseen complete observations, one newly timed common cold prefix per frame. Its entire geometry, native, rollback, '
              'count growth and original terminal completion charged to every policy plus measured selector and new readout. '
              'Cold normal-entry replays of both variants on fixed endpoint/eligible samples are separate.')
    return result


def main(out):
    start=time.perf_counter()
    frozen=freeze(out)
    rows=[]
    first_eligible={}
    for index,seed in enumerate(SEEDS):
        tick=time.perf_counter()
        model,_=geometry(seed)
        geometry_s=time.perf_counter()-tick
        tick=time.perf_counter()
        s,labels=make_frame(model,seed,0)
        source_s=time.perf_counter()-tick
        tick=time.perf_counter()
        before=input_snapshot(s)
        verification=time.perf_counter()-tick
        tick=time.perf_counter()
        base_counts,base_meta=count_growth(s)
        wall=time.perf_counter()-tick
        assert base_meta['complete']
        fitting=Counter(dict(base_meta['cancellation_action']))
        eligible=s.k-fitting.total()==1
        if eligible:
            first_eligible.setdefault(index//64,seed)
        row=dict(seed=seed,frame=0,eligible=eligible,fitted_count=fitting.total(),labels=labels,input_snapshot=before,
            baseline_action=sorted(base_counts.items()),baseline=base_meta,reference=metrics(base_counts,base_meta['bank'],labels),
            repeated_messages=len(labels)-len(set(labels)),repeated_patterns=len(labels)-len({w>>87 for w in labels}),
            order=list(MODES if index%2==0 else reversed(MODES)),geometry_seconds=geometry_s,base_seconds=wall,
            source_seconds=source_s,arms={},finish_seconds={},ordinary_replay_seconds={})
        if eligible:
            row['correct_cancellation']=not (fitting-Counter(labels))
        def baseline_identity():
            return hashlib.sha256(json.dumps([base_counts,base_meta],sort_keys=True).encode()).hexdigest()
        tick=time.perf_counter()
        baseline_hash=baseline_identity()
        verification+=time.perf_counter()-tick
        for m in row['order']:
            tick=time.perf_counter()
            output,meta=finish(s,base_counts,base_meta,m)
            seconds=time.perf_counter()-tick
            row['finish_seconds'][m]=seconds
            tick=time.perf_counter()
            assert meta['complete'] and meta['terminal_selector']==m and (meta['terminal_query'] is not None)==eligible
            assert all(isinstance(c,int) and c>0 for c in output.values()) and sum(output.values())<=s.k
            assert set(output)<=set(meta['bank']) and baseline_identity()==baseline_hash and input_snapshot(s)==before
            if eligible:
                trace=meta['terminal_query']
                new_true=set(output).intersection(trace['new_ids'],labels)
                row['arms'][m]=dict(action=sorted(output.items()),metrics=metrics(output,meta['bank'],labels),
                    retained_new_true=sorted(new_true),trace=trace,wall_seconds=seconds)
            else:
                assert output==base_counts and meta['bank']==base_meta['bank']
            verification+=time.perf_counter()-tick
            if seed in (SEEDS[0],SEEDS[-1],first_eligible.get(index//64)):
                tick=time.perf_counter()
                full,rep=ordinary(s,selector=m)
                row['ordinary_replay_seconds'][m]=time.perf_counter()-tick
                tick=time.perf_counter()
                assert full==output and semantic(meta)==semantic(rep) and input_snapshot(s)==before
                verification+=time.perf_counter()-tick
        row['verification_seconds']=verification
        write_json(out/f'frame_{seed}.json',row)
        rows.append(row)
        print(json.dumps(dict(seed=seed,eligible=eligible,ordinary_replay=bool(row['ordinary_replay_seconds']),
            elapsed=time.perf_counter()-start)),flush=True)
    result=summary(rows)
    assert all(digest(ROOT/n)==v for n,v in frozen.items())
    result.update(elapsed_seconds=time.perf_counter()-start,peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
        (1 if sys.platform=='darwin' else 1024),source_freeze_sha256=digest(out/'freeze.json'))
    write_json(out/'summary.json',result)
    print(json.dumps(result['gates']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
