"""One terminal magnitude query added to the frozen DEC-35 flat-encoder curves."""

from collections import Counter
import csv
import math
import time
from unittest.mock import patch

import numpy as np

import paper_curve as curve
from paper_curve_analysis import BOOTSTRAPS, METRICS, crossing, finite, interval, report_crossing
from framework import polar_folded_terminal
from framework.polar_count_cavity import action_hash
from framework.polar_terminal_query import finish

NAMES = ('growth','magnitude')


def terminal_endpoint(s, baseline, stages, base_cost, geometry_seconds):
    # with_endpoints rejects incomplete prefixes; these are all the original fields finish reads.
    prefix = dict(complete=True, bank=stages['bank'], cancellation_action=stages['cancellation_action'])
    query = polar_folded_terminal.cavity_query
    inputs = []

    def observed(*args):
        llr, context = query(*args)
        inputs.append((llr,context))
        return llr,context

    with patch.object(polar_folded_terminal,'cavity_query',observed):
        start = time.perf_counter()
        output,meta = finish(s,baseline,prefix,'magnitude')
        seconds = time.perf_counter()-start
    trace = meta['terminal_query']
    if not meta['complete'] or len(inputs)!=int(trace is not None):
        raise RuntimeError('Incomplete terminal count refit or unobserved physical query')
    total = base_cost['receiver_seconds']+seconds
    return dict(action=sorted(output.items()),triggered=trace is not None,trace=trace,
                prefix_cancellation_sha256=action_hash(dict(prefix['cancellation_action'])),y_sha256=curve.digest(s.y),
                llr_sha256=curve.digest(inputs[0][0]) if inputs else None,query_context=inputs[0][1] if inputs else None,
                **{k:v for k,v in base_cost.items() if k!='receiver_seconds'},terminal_scl4096=len(inputs),
                incremental_seconds=seconds,receiver_seconds=total,seconds_with_geometry=total+geometry_seconds)


def verify(row):
    k = row['k']
    t,base,s = row['terminal_magnitude'],row['endpoints']['growth'],row['stages']
    action,initial = dict(t['action']),dict(base['action'])
    fitted = dict(s['cancellation_action'])
    active = k-sum(fitted.values())==1
    if (t['triggered'] is not active or t['terminal_scl4096']!=int(active) or t['y_sha256']!=row['source']['y_sha256']
            or t['prefix_cancellation_sha256']!=action_hash(fitted) or t['metrics']!=curve.score(row['labels'],action)
            or len(t['action'])!=len(action) or base['metrics']!=curve.score(row['labels'],initial)):
        raise ValueError('Invalid terminal trigger, physical input or output metrics')
    if (not math.isfinite(t['incremental_seconds']) or t['incremental_seconds']<=0 or
            t['receiver_seconds']!=base['receiver_seconds']+t['incremental_seconds'] or
            t['seconds_with_geometry']!=t['receiver_seconds']+row['geometry_seconds'] or
            any(t[key]!=base[key] for key in ('native_scl128','continuation_scl128','rollback_scl512'))):
        raise ValueError('Terminal work inconsistent with the shared prefix')
    trace = t['trace']
    if not active:
        if any(t[key] is not None for key in ('trace','llr_sha256','query_context')) or action!=initial:
            raise ValueError('Inactive terminal changed the action or issued a query')
        return
    pattern = trace['pattern']
    ids,metrics,fresh = trace['query_ids'],trace['list_metrics'],trace['new_ids']
    bits = 88 if k==50 else 87
    ctx = t['query_context']
    if (len(ids)!=len(set(ids)) or len(ids)!=len(metrics) or len(ids)>4096 or not all(map(math.isfinite,metrics)) or
            metrics!=sorted(metrics) or any(w>>bits!=pattern for w in ids) or not 0<=pattern<1<<(100-bits) or
            fresh!=sorted(set(ids)-set(s['bank'])) or trace['signed_prefix_end'] is not None or trace['model_top_probability'] is not None or
            len(t['llr_sha256'])!=64 or any(c not in '0123456789abcdef' for c in t['llr_sha256']) or
            ctx!=dict(context='raw',restored=[],restored_count=0,outside_count=1,pending=1,variance=row['source']['variance'])):
        raise ValueError('Inconsistent terminal SCL4096 list or input')
    fit,final_fit = trace['fit'],dict(trace['cancellation'])
    if (bool(fresh)!=(fit is not None) or (fit is not None and not fit['converged']) or
            (not fresh and (final_fit!=fitted or action!=initial)) or not set(action)<=set(s['bank']).union(fresh) or
            Counter(action)!=Counter(final_fit)+Counter(dict(trace['completion']['additions']))):
        raise ValueError('Inconsistent terminal refit or completion')


def target(db, total, boot, users):
    points = [crossing(db,t,users) for t in total.T]
    draws = [[crossing(db,t,users) for t in rep.T] for rep in boot]
    thresholds = {}
    for i,name in enumerate(NAMES):
        group = [r[i] for r in draws]
        thresholds[name] = dict(**report_crossing(points[i]),bootstrap_status=dict(Counter(v[0] for v in group)),
                                paired_bootstrap_95_db=interval([v[1] for v in group],[v[2] for v in group]))
    a,b = points
    lo = [a[1]-b[2] for a,b in draws]
    hi = [a[2]-b[1] for a,b in draws]
    return dict(thresholds=thresholds, snr_db_saved=a[1]-b[1] if a[1]==a[2] and b[1]==b[2] else None,
                censor_bounds_db=[finite(a[1]-b[2]),finite(a[2]-b[1])],paired_bootstrap_95_db=interval(lo,hi),
                both_crossings_observed_resamples=int(np.sum(np.isfinite(lo)&np.isfinite(hi))),resamples=len(draws))


def summarize_terminal(rows):
    records = {}
    for row in rows:
        key = row['k'],row['seed'],row['snr_cdb']
        if key in records:
            raise ValueError('Duplicate terminal curve case')
        verify(row)
        t,base = row['terminal_magnitude'],row['endpoints']['growth']
        losses = [[arm['metrics'][m] for m in METRICS] for arm in (base,t)]
        work = [t[m] for m in ('terminal_scl4096','incremental_seconds','receiver_seconds','seconds_with_geometry')]
        records[key] = losses,work,action_hash(dict(t['action']))
    expected = {(k,s,d) for k,seeds in curve.CURVE_SEEDS.items() for s in seeds for d in curve.CURVE_DB[k]}
    if records.keys()!=expected:
        raise ValueError('Incomplete or undeclared terminal curve')
    result = {}
    for k,seeds in curve.CURVE_SEEDS.items():
        grid,n = curve.CURVE_DB[k],len(seeds)
        raw = np.asarray([[records[k,s,d][0] for d in grid] for s in seeds],dtype=np.int64)
        work = np.asarray([[records[k,s,d][1] for d in grid] for s in seeds])
        weights = np.random.default_rng([939039,k]).multinomial(n,np.full(n,1/n),size=BOOTSTRAPS)
        boot = (weights @ raw[:,:,:,:3].reshape(n,-1)).reshape(BOOTSTRAPS,len(grid),2,3)
        cells = {}
        for j,d in enumerate(grid):
            arms,paired = {},{}
            for i,name in enumerate(NAMES):
                arms[name] = dict(zip(METRICS,map(int,raw[:,j,i,:].sum(axis=0)),strict=True))
                for m,rate in enumerate(('list_pupe','count_pupe','false_copies_per_user')):
                    arms[name][rate] = float(raw[:,j,i,m].sum()/(n*k))
                    arms[name][rate+'_95'] = np.quantile(boot[:,j,i,m]/(n*k),[.025,.975]).tolist()
            for m,rate in enumerate(('list_pupe','count_pupe','false_copies_per_user')):
                delta = raw[:,j,0,m]-raw[:,j,1,m]
                paired[rate] = dict(saved=int(delta.sum()),wins=int((delta>0).sum()),ties=int((delta==0).sum()),
                                    losses=int((delta<0).sum()),rate_saved_95=np.quantile(
                                        (boot[:,j,0,m]-boot[:,j,1,m])/(k*n),[.025,.975]).tolist())
            cells[str(d)] = dict(arms=arms,paired=paired,terminal_scl4096=int(work[:,j,0].sum()),
                                 incremental_seconds_total=float(work[:,j,1].sum()),incremental_seconds_median=float(np.median(work[:,j,1])),
                                 receiver_seconds_total=float(work[:,j,2].sum()),seconds_with_geometry_total=float(work[:,j,3].sum()))
        db = np.asarray(grid)/100
        crossings = {m:target(db,raw[:,:,:,i].sum(axis=0),boot[:,:,:,i],n*k) for i,m in enumerate(('list_pupe','count_pupe'))}
        frames = []
        for seed in seeds:
            for d in grid:
                losses,costs,signature = records[k,seed,d]
                frame = dict(seed=seed,snr_cdb=d,terminal_scl4096=costs[0],incremental_seconds=costs[1],
                             receiver_seconds=costs[2],seconds_with_geometry=costs[3],magnitude_action_sha256=signature)
                for name,values in zip(NAMES,losses,strict=True):
                    frame.update({name+'_'+m:v for m,v in zip(METRICS,values,strict=True)})
                frames.append(frame)
        result[str(k)] = dict(frames_per_point=n,users_per_point=n*k,by_snr=cells,crossings=crossings,frames=frames)
    return result


def write_terminal_tables(summary, out):
    with (out/'terminal_frames.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['K',*next(iter(summary.values()))['frames'][0]])
        writer.writeheader()
        for k,data in summary.items():
            writer.writerows({'K':k,**row} for row in data['frames'])
    with (out/'terminal_curve.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['K','ebn0_db','frames','users','metric','growth_rate','magnitude_rate','growth_lo95','growth_hi95',
                         'magnitude_lo95','magnitude_hi95','saved','wins','ties','losses','paired_saving_lo95','paired_saving_hi95',
                         'terminal_scl4096','incremental_seconds_total','incremental_seconds_median','receiver_seconds_total',
                         'seconds_with_geometry_total'])
        for k,data in summary.items():
            for db,cell in data['by_snr'].items():
                base,term = (cell['arms'][name] for name in NAMES)
                for m,gain in cell['paired'].items():
                    writer.writerow([k,int(db)/100,data['frames_per_point'],data['users_per_point'],m,base[m],term[m],
                                     *base[m+'_95'],*term[m+'_95'],gain['saved'],gain['wins'],gain['ties'],gain['losses'],
                                     *gain['rate_saved_95'],cell['terminal_scl4096'],cell['incremental_seconds_total'],
                                     cell['incremental_seconds_median'],cell['receiver_seconds_total'],cell['seconds_with_geometry_total']])
    with (out/'terminal_energy.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['K','metric','estimate','status','dB','bracket_lo','bracket_hi','censor_lo','censor_hi','ci95_lo','ci95_hi'])
        for k,data in summary.items():
            for m,res in data['crossings'].items():
                for name,v in res['thresholds'].items():
                    writer.writerow([k,m,name,v['status'],v['snr_db'],*(v['observed_grid_bracket_db'] or [None,None]),
                                     *v['censor_bounds_db'],*v['paired_bootstrap_95_db']])
                status = 'observed' if res['snr_db_saved'] is not None else 'partially_identified'
                writer.writerow([k,m,'growth_minus_magnitude',status,res['snr_db_saved'],None,None,*res['censor_bounds_db'],
                                 *res['paired_bootstrap_95_db']])
