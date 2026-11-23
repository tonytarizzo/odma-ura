"""Read back the frozen 128-frame Job 037 terminal confirmation, independently of Job 039."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from terminal_audit_physics import (COUNTS, ERRORS, ROOT, PhysicalBank, action_from_pairs, audit_completion,
                                    close, encode, measurements, regenerate)
from baselines.polar import PolarCode
from framework.polar_terminal_query import run as entry
from framework.polar_tractable import PolarScenario

SEEDS = range(7501,7629)
MODES = ('magnitude','folded')
TIMES = ('setup_seconds','ranking_seconds','search_seconds','refit_completion_seconds')
METRICS = ('misses','matched_copies','false_copies','output_copies','multiplicity_capable_oracle_bank_miss_floor')
MANIFEST = ROOT/'jobs/037_polar_tractable/terminal_confirmation_inputs.json'


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def semantic(obj):
    if isinstance(obj,dict):
        return {k:semantic(v) for k,v in obj.items() if not k.endswith('seconds')}
    if isinstance(obj,(list,tuple)):
        return [semantic(v) for v in obj]
    return obj


def check_sources(folder):
    release, frozen = read(MANIFEST),read(folder/'freeze.json')
    unavailable_prior = []
    for name,h in frozen.items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        snapshot = folder/'source'/name
        if snapshot.exists():
            assert sha(snapshot)==h,name
        else:
            assert name.startswith('results/'),name
        here = ROOT/name
        if here.exists():
            assert sha(here)==h,name
        else:
            assert name.startswith('results/'),name
            unavailable_prior.append(name)
        if name in release['source_sha256']:
            assert release['source_sha256'][name]==h,name
    normal = 'jobs/037_polar_tractable/run_terminal_confirmation.py'
    archived = 'jobs/037_polar_tractable/diagnose_terminal_confirmation.py'
    if normal in frozen:
        assert frozen=={**release['source_sha256'],str(MANIFEST.relative_to(ROOT)):sha(MANIFEST)}
    else:
        assert frozen[archived]==release['archived_driver_sha256']
        for tail,key in (('/confirmation/design.json','archived_design_sha256'),('/confirmation/preflight.json','archived_preflight_sha256')):
            matches = [h for name,h in frozen.items() if name.endswith(tail)]
            assert matches==[release[key]]
    return dict(freeze_sha256=sha(folder/'freeze.json'),release_sha256=sha(MANIFEST),historical_hashes_without_local_bytes=unavailable_prior,
                frozen_inputs=len(frozen),scope=release['design']['audit'])


def physical(row):
    model,labels,y,_,_ = regenerate(row['seed'])
    assert labels==row['labels']
    arrays = dict(y=y,positions=model.positions,amplitudes=model.amplitudes,mean=model.mean,
                  information=model.code.information,frozen=model.code.frozen)
    assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}==row['input_snapshot']['arrays']
    assert row['input_snapshot']['code_configuration']==[87,512,16,128]
    assert row['input_snapshot']['scenario_configuration']==[100,1/(200*10**.05),'iid',30000,8192,100]
    return model,y


def check_selection(row, phys, state, wide):
    model,raw,scale = phys.model,phys.raw(state),phys.variance
    generator = encode(np.eye(87,dtype=np.uint8),model.code.information).T
    assert np.all(np.any(generator,axis=1)==(np.arange(512)!=0)) and np.all(model.positions[:,1:]>model.positions[:,:-1])
    basis,end = {},int(model.positions[:,0].max())
    for i,bits in enumerate(generator[1:],1):
        v = bits.copy()
        for j in sorted(basis):
            if v[j]:
                v ^= basis[j]
        pivots = np.flatnonzero(v)
        if not len(pivots):
            end = min(end,int(model.positions[:,i].min())-1)
            break
        basis[int(pivots[0])] = v
    scores = {m:np.empty(8192) for m in MODES}
    for lo in range(0,8192,96):
        idx = slice(lo,lo+96)
        pos,amp = model.positions[idx],model.amplitudes[idx]
        field = raw[pos]*amp/scale
        absolute = abs(field)
        terms = absolute+np.log1p(np.exp(-2*absolute))-math.log(2.)
        keep = pos[:,0]<=end
        terms[keep,0] = field[keep,0]
        scores['folded'][idx] = terms.sum(axis=1)-np.square(amp).sum(axis=1)/(2*scale)
        scores['magnitude'][idx] = abs(raw[pos]).sum(axis=1)
    for mode in MODES:
        tr = row['arms'][mode]['trace']
        p = int(np.argmax(scores[mode]))
        assert tr['pattern']==p
        if mode=='folded':
            assert tr['signed_prefix_end']==end
            close(1/np.exp(scores[mode]-scores[mode][p]).sum(),tr['model_top_probability'],'coarsened_top_probability',2e-12)
        else:
            assert tr['signed_prefix_end'] is None and tr['model_top_probability'] is None
        ids,metric = tr['query_ids'],tr['list_metrics']
        assert all(type(w) is int and w>>87==p for w in ids) and len(ids)==len(set(ids))==len(metric)<=4096
        assert all(math.isfinite(v) for v in metric) and all(x<=z+2e-9 for x,z in zip(metric,metric[1:]))
        llr = 2*model.amplitudes[p]*raw[model.positions[p]]/scale
        codes = phys.codewords[[phys.index[w] for w in ids]]
        close(np.logaddexp(0,-(1.-2.*codes)*llr).sum(axis=1),metric,'returned_codeword_metrics',2e-9)
        if wide:
            found,metrics = PolarCode(87,512,16,4096).decode_list(llr)
            assert [model.word(p,x) for x in found]==ids
            close(metrics,metric,'wide_list_metrics',2e-9)
            COUNTS['ordinary_wide_list_replays'] += 1
    COUNTS['all_pattern_double_rankings'] += 1


def completion(base, bank, final, meta, wall, labels, phys):
    if phys is not None:
        proxy = dict(completed_action=final,completion_meta=meta,labels=labels,completion_wall_seconds=wall,
                     completion_preserves_counts_and_cap=True)
        return audit_completion(proxy,phys,base,bank)[0]
    out = action_from_pairs(final)
    additions = out-base
    absent = bank-set(base)
    capacity = 100-base.total()
    assert not (base-out) and set(additions)<=absent and all(c==1 for c in additions.values())
    assert len(additions)==min(capacity,len(absent)) and sorted([w,c] for w,c in additions.items())==meta['additions']
    assert meta['bank_size']==len(bank) and meta['absent_bank_ids']==len(absent)
    assert meta['input_count']==base.total() and meta['output_count']==out.total()
    assert meta['method']==('no_capacity' if not capacity else 'all_fit' if len(absent)<=capacity else 'count_add_gain')
    assert meta['seconds']<=wall and meta['ranking_seconds']+meta['bank_construction_seconds']<=meta['seconds']
    return out


def check_row(row, wide=False, normal=False):
    labels,meta = row['labels'],row['baseline']
    truth,bank = Counter(labels),set(meta['bank'])
    state = action_from_pairs(meta['cancellation_action'])
    assert row['frame']==0 and len(labels)==100 and all(type(w) is int and 0<=w<1<<100 for w in labels)
    assert meta['complete'] and meta['complete_support'] and meta['continuation']['count_growth']
    assert row['eligible']==(state.total()==99) and row['fitted_count']==state.total()
    assert row['repeated_messages']==100-len(truth) and row['repeated_patterns']==100-len({w>>87 for w in truth})
    assert set(state)<=bank and meta['bank']==sorted(bank) and all(0<=w<1<<100 for w in bank)
    assert row['order']==list(MODES if (row['seed']-SEEDS.start)%2==0 else reversed(MODES))
    assert row['base_seconds']>=meta['total_seconds'] and all(row[k]>0 for k in ('geometry_seconds','source_seconds','verification_seconds'))
    assert set(row['finish_seconds'])==set(MODES) and all(math.isfinite(t) and t>0 for t in row['finish_seconds'].values())
    assert set(row['ordinary_replay_seconds'])==(set(MODES) if normal else set())
    assert set(row['arms'])==(set(MODES) if row['eligible'] else set())
    model = y = phys = None
    if row['eligible'] or normal:
        model,y = physical(row)
    if row['eligible']:
        assert row['correct_cancellation']==(not (state-truth))
        allwords = bank.union(*(part['trace']['query_ids'] for part in row['arms'].values()))
        phys = PhysicalBank(model,y,allwords)
        check_selection(row,phys,state,wide)
        finalfit = (meta['continuation']['refits'][-1]['meta'] if meta['continuation']['refits'] else meta['continuation']['initial_refit'])
        _,mu,kappa,log_out = phys.outside(bank)
        close(phys.objective(state,phys.raw(state),mu,kappa,log_out),finalfit['final_objective'],'baseline_final_objective')
        close(phys.best_gain(state,bank,phys.raw(state),mu,kappa,log_out),finalfit['next_positive_gain'],'baseline_terminal_gain')
        assert finalfit['converged']
    base = completion(state,bank,row['baseline_action'],meta['support_completion'],row['base_seconds'],labels,phys)
    assert measurements(base,bank,labels)==row['reference']
    for mode in MODES:
        expected = {**meta,'terminal_selector':mode,'terminal_query':None}
        final = base
        if row['eligible']:
            arm = row['arms'][mode]
            tr = arm['trace']
            new = set(tr['query_ids'])-bank
            assert tr['new_ids']==sorted(new) and (tr['fit'] is not None)==bool(new)
            larger = bank|new
            fitted = phys.refit(state,larger,tr['fit']) if new else state
            assert action_from_pairs(tr['cancellation'])==fitted and (not new or tr['fit']['converged'])
            final = completion(fitted,larger,arm['action'],tr['completion'],tr['refit_completion_seconds'],labels,phys)
            assert measurements(final,larger,labels)==arm['metrics']
            assert arm['retained_new_true']==sorted(set(final)&new&set(truth))
            assert all(math.isfinite(tr[k]) and tr[k]>=0 for k in TIMES)
            assert sum(tr[k] for k in TIMES)<=tr['seconds']+1e-8 and tr['seconds']<=arm['wall_seconds']==row['finish_seconds'][mode]
            expected.update(terminal_query=tr,bank=sorted(larger),cancellation_action=tr['cancellation'],support_completion=tr['completion'])
        if normal:
            start = time.perf_counter()
            actual,observed = entry(PolarScenario(model,y,100,1/(200*10**.05)),selector=mode)
            assert actual==dict(final) and semantic(observed)==semantic(expected)
            COUNTS['normal_entry_replays'] += 1
            COUNTS['normal_entry_replay_seconds'] += time.perf_counter()-start
    if row['eligible']:
        first,second = [row['arms'][mode] for mode in MODES]
        if first['trace']['pattern']==second['trace']['pattern']:
            for k in ('query_ids','list_metrics','new_ids','cancellation','fit','completion'):
                assert semantic(first['trace'][k])==semantic(second['trace'][k])
            assert first['action']==second['action']


def aggregate(rows):
    eligible = [r for r in rows if r['eligible']]
    def metrics(r,name):
        return r['reference'] if name=='reference' or not r['eligible'] else r['arms'][name]['metrics']
    def delta(a,b):
        v = [metrics(r,a)['misses']-metrics(r,b)['misses'] for r in rows]
        return dict(saved=sum(v),better=sum(i>0 for i in v),worse=sum(i<0 for i in v),tied=v.count(0),
            leave_best=sum(v)-max(v),max_regression=max(0,-min(v)),changed_frames=[[r['seed'],g] for r,g in zip(rows,v,strict=True) if g])
    totals = {m:{k:sum(metrics(r,m)[k] for r in rows) for k in METRICS} for m in ('reference',*MODES)}
    vs = {m:delta('reference',m) for m in MODES}
    specific = delta(*MODES)
    exclusive = [r['seed'] for r in eligible if r['arms']['magnitude']['trace']['pattern']!=r['arms']['folded']['trace']['pattern']
        and set(r['arms']['folded']['retained_new_true'])-set(r['arms']['magnitude']['retained_new_true'])]
    gates = {m:dict(four_net=v['saved']>=4,three_frames=v['better']>=3,leave_best_two=v['leave_best']>=2,
                    max_regression_one=v['max_regression']<=1) for m,v in vs.items()}
    gates['specific'] = dict(two_net=specific['saved']>=2,two_frames=specific['better']>=2,novel_different_pattern=bool(exclusive))
    base = sum(r['base_seconds']+r['geometry_seconds'] for r in rows)
    return dict(frames=len(rows),eligible=len(eligible),correct_cancellation=sum(r['correct_cancellation'] for r in eligible),
        totals=totals,vs_reference=vs,folded_vs_magnitude=specific,exclusive_different_pattern_seeds=exclusive,gates=gates,
        passes={m:all(gates[m].values()) and (m=='magnitude' or all(gates['specific'].values())) for m in MODES},
        same_selection=sum(r['arms']['magnitude']['trace']['pattern']==r['arms']['folded']['trace']['pattern'] for r in eligible),
        repeated_messages=sum(r['repeated_messages'] for r in rows),repeated_patterns=sum(r['repeated_patterns'] for r in rows),
        eligible_stage_seconds={m:{k:sum(r['arms'][m]['trace'][k] for r in eligible) for k in TIMES} for m in MODES},
        geometry_seconds=sum(r['geometry_seconds'] for r in rows),common_cold_prefix_with_geometry_seconds=base,
        geometry_inclusive_standalone_seconds=dict(reference=base,**{m:base+sum(r['finish_seconds'][m] for r in rows) for m in MODES}),
        eligible_seeds=[r['seed'] for r in eligible],ordinary_replay_seeds=[r['seed'] for r in rows if r['ordinary_replay_seconds']],
        ordinary_replay_seconds=sum(sum(r['ordinary_replay_seconds'].values()) for r in rows),
        source_seconds=sum(r['source_seconds'] for r in rows),verification_seconds=sum(r['verification_seconds'] for r in rows))


def audit(folder, target):
    if not __debug__:
        raise RuntimeError('Audit assertions must be enabled')
    if target.exists():
        raise FileExistsError(target)
    start = time.perf_counter()
    sources = check_sources(folder)
    expected = set(f'frame_{s}.json' for s in SEEDS)
    if {f.name for f in folder.glob('frame_*.json')}!=expected or not (folder/'summary.json').is_file():
        raise ValueError('Require the complete 128-row pilot-independent 7501..7628 confirmation and its aggregate')
    hashes = {name:sha(folder/name) for name in sorted(expected|{'summary.json','freeze.json'})}
    rows = [read(folder/f'frame_{seed}.json') for seed in SEEDS]
    assert [r['seed'] for r in rows]==list(SEEDS)
    halves = [[r['seed'] for r in rows[i:i+64] if sum(c for w,c in r['baseline']['cancellation_action'])==99] for i in (0,64)]
    wide = {s for half in halves if half for s in (half[0],half[-1])}
    normal = {SEEDS[0],SEEDS[-1]}|{half[0] for half in halves if half}
    for row in rows:
        check_row(row,wide=row['seed'] in wide,normal=row['seed'] in normal)
    outcome = aggregate(rows)
    summary = read(folder/'summary.json')
    assert all(summary[k]==v for k,v in outcome.items()),[k for k,v in outcome.items() if summary[k]!=v]
    assert set(summary)-set(outcome)=={'elapsed_seconds','peak_bytes','scope','source_freeze_sha256'}
    assert summary['source_freeze_sha256']==sources['freeze_sha256'] and summary['peak_bytes']>0
    charged = outcome['common_cold_prefix_with_geometry_seconds']+outcome['ordinary_replay_seconds']+outcome['source_seconds']
    charged += sum(sum(r['finish_seconds'].values())+r['verification_seconds'] for r in rows)
    assert charged<=summary['elapsed_seconds'] and all(sha(folder/name)==h for name,h in hashes.items())
    result = dict(passed=True,reconstructed=outcome,selected_wide_seeds=sorted(wide),selected_normal_seeds=sorted(normal),
        sources=sources,raw_sha256=hashes,audit_source_sha256={p.name:sha(p) for p in
        (Path(__file__),Path(__file__).with_name('terminal_audit_physics.py'))},
        counts=dict(COUNTS),maximum_absolute_errors=dict(ERRORS),elapsed_seconds=time.perf_counter()-start,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__),
        scope='All 128 stored actions, bank floors, timings, source hashes and gates. All eligible physical observations, both '
              'all-pattern selectors, every returned-word metric, fitted terminal counts and decisions. Ordinary 4096 lists on '
              'the earliest/latest eligible per half. Cold ordinary entire receiver on producer normal-entry endpoints. '
              'Other stored upstream prefixes are not independent complete native or rollback reruns.')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True,gates=outcome['passes'],elapsed_seconds=result['elapsed_seconds'],sha256=sha(target))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    audit(args.input.resolve(),args.out.resolve())
