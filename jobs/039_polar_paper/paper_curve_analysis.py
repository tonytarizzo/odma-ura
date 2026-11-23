"""Paired frame-level estimation on the declared SNR grid with explicit crossing censoring."""

from collections import Counter
import math

import numpy as np

from paper_curve import CENTRAL_DB, CURVE_DB, CURVE_SEEDS, ENDPOINTS, audit_cell, digest, score

METRICS = ('list_misses','count_misses','false_copies','foreign_ids','output_copies','list_size')
COSTS = ('receiver_seconds','seconds_with_geometry','native_scl128','rollback_scl512','continuation_scl128')
BOOTSTRAPS = 10000
TARGET = .05


def crossing(db, errors, users):
    """First persistent downward crossing of the 5% level, evaluated on ordered raw sample means."""
    # Compare integer copy counts to exactly 1/20 without rounded PUPE or accidental 5% ties.
    diff = 20*np.asarray(errors,dtype=np.int64)-users
    below = np.flatnonzero(diff<=0)
    if not len(below):
        return 'right_censored',float(db[-1]),math.inf,None
    j = int(below[0])
    if np.any(diff[j:]>0):
        return 'ambiguous',-math.inf,math.inf,None
    if j==0 and diff[j]<0:
        return 'left_censored',-math.inf,float(db[0]),None
    if diff[j]==0:
        v = float(db[j])
        return 'at_grid',v,v,[v,v]
    a,b = float(db[j-1]),float(db[j])
    v = a+(b-a)*float(diff[j-1])/float(diff[j-1]-diff[j])
    return 'interpolated',v,v,[a,b]


def finite(x):
    return float(x) if math.isfinite(x) else None


def report_crossing(c):
    status,lo,hi,bracket = c
    return dict(status=status,snr_db=finite(lo) if lo==hi else None,observed_grid_bracket_db=bracket,
                censor_bounds_db=[finite(lo),finite(hi)])


def interval(lower, upper, alpha=.05):
    """Empirical inverse-CDF endpoints including all censored or ambiguous repetitions."""
    n = len(lower)
    lo,hi = np.sort(lower),np.sort(upper)
    return [finite(lo[max(0,math.ceil(n*alpha/2)-1)]),finite(hi[math.ceil(n*(1-alpha/2))-1])]


def crossing_analysis(cells, boot, db, k, n):
    # Arrays use (SNR,endpoint) for the point estimate and (resample,SNR,endpoint) for replicates.
    point = [crossing(db,cells[:,a],n*k) for a in range(3)]
    all_cases = [[crossing(db,panel[:,a],n*k) for a in range(3)] for panel in boot]
    estimates = {}
    for i,name in enumerate(ENDPOINTS):
        draws = [b[i] for b in all_cases]
        estimates[name] = dict(**report_crossing(point[i]),bootstrap_status=dict(Counter(v[0] for v in draws)),
            paired_bootstrap_95_db=interval([v[1] for v in draws],[v[2] for v in draws]))
    gains = {}
    for a,b in ((0,2),(1,2),(0,1)):
        first,second = point[a],point[b]
        lows = [v[a][1]-v[b][2] for v in all_cases]
        highs = [v[a][2]-v[b][1] for v in all_cases]
        bounded = sum(np.isfinite(x) and np.isfinite(y) for x,y in zip(lows,highs,strict=True))
        point_db = first[1]-second[1] if first[1]==first[2] and second[1]==second[2] else None
        data = dict(snr_db_saved_by_second=point_db,paired_bootstrap_95_db=interval(lows,highs),
                    both_crossings_observed_resamples=int(bounded),resamples=len(all_cases))
        if (a,b)==(0,2):
            data['bonferroni_two_loads_97_5_db'] = interval(lows,highs,alpha=.025)
        gains[f'{ENDPOINTS[a]}_minus_{ENDPOINTS[b]}'] = data
    return dict(thresholds=estimates,required_energy_contrasts=gains)


def verify_row(row):
    k,seed,db = row['k'],row['seed'],row['snr_cdb']
    if k not in CURVE_SEEDS or seed not in CURVE_SEEDS[k] or db not in CURVE_DB[k] or row['study']!='paper_curve':
        raise ValueError(f'Unexpected curve row {(k,seed,db)}')
    def check(cond):
        if not cond:
            raise ValueError(f'Invalid curve row {(k,seed,db)}')
    labels,source,s = row['labels'],row['source'],row['stages']
    check(len(labels)==k and all(type(w) is int and 0<=w<1<<100 for w in labels))
    data = np.frombuffer(b''.join(w.to_bytes(13,'big') for w in labels),dtype=np.uint8).reshape(k,13)
    check(source['bits_sha256']==digest(np.unpackbits(data,axis=1)[:,4:]))
    for field in ('bits','standard_noise','y','positions','amplitudes','information'):
        h = source[f'{field}_sha256']
        check(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h))
    check(source['variance']==1/(200*10**(db/1000)) and source['repeated_messages']==k-len(set(labels)))
    check(source['repeated_patterns']==k-len({w>>(88 if k==50 else 87) for w in labels}))
    check(set(row['endpoints'])==set(ENDPOINTS) and row['geometry_seconds']>0 and row['source_seconds']>0)
    check(s['native_calls']>0 and 0<=s['cache_hits']<=s['logical_continuation'])
    ac = {}
    for name in ENDPOINTS:
        arm = row['endpoints'][name]
        action = dict(arm['action'])
        check(len(action)==len(arm['action']) and score(labels,action)==arm['metrics'])
        check(arm['receiver_seconds']>0 and arm['seconds_with_geometry']==arm['receiver_seconds']+row['geometry_seconds'])
        check(arm['native_scl128']==s['native_calls'] and arm['rollback_scl512']==(0 if name=='native' else 32))
        check(arm['continuation_scl128']==(s['logical_continuation']-s['cache_hits'] if name=='growth' else 0))
        ac[name] = Counter(action)
    native = dict(s['native_order_round'])
    check(len(native)==len(s['native_order_round'])==sum(s['native_round_additions']) and set(native)<=set(s['native_valid_ids']))
    check(ac['native']==Counter(native.keys()) and all(n==1 for n in ac['native'].values()))
    check(Counter(native.values())==Counter(dict(enumerate(s['native_round_additions']))))
    check(len(s['rollback_batches'])==4 and all(len(b['patterns'])==8 for b in s['rollback_batches']))
    check(set(ac['rollback'])<=set(s['base_bank']) and set(s['base_bank'])<=set(s['bank']))
    check(ac['growth']==Counter(dict(s['cancellation_action']))+Counter(dict(s['terminal_additions'])))
    check(set(ac['growth'])<=set(s['bank']))
    control = row['ordinary_control']
    if audit_cell(k,seed,db):
        check(control and control['passed'] and control['accepted_ids']==list(native) and control['ordinary_native_calls']==s['native_calls'])
        check(control['independent_y_sha256']==source['y_sha256'])
        check(control['baseline_order_equal']==(control['baseline_accepted_ids']==control['accepted_ids']))
        check(control['baseline_calls_equal']==(control['baseline_scl128_calls']==control['ordinary_native_calls']))
    else:
        check(control is None)


def summarize(rows):
    bank = {}
    common = {}
    observations = {}
    repeats = Counter()
    controls = {}
    for row in rows:
        verify_row(row)
        k,seed,db = row['k'],row['seed'],row['snr_cdb']
        key = k,seed,db
        if key in bank:
            raise ValueError(f'Duplicate curve row {key}')
        source = row['source']
        signature = (tuple(row['labels']),*(source[f'{name}_sha256'] for name in
                     ('bits','standard_noise','positions','amplitudes','information')))
        if common.setdefault((k,seed),signature)!=signature:
            raise ValueError(f'Paired source, noise or code differs across SNR at {key}')
        code = (k,db,source['y_sha256'])
        if code in observations:
            raise ValueError('Identical physical observations for supposedly independent seeds')
        observations[code] = seed
        if db==CENTRAL_DB[k]:
            for term in ('repeated_patterns','repeated_messages'):
                repeats[k,term] += source[term]
        if row['ordinary_control']:
            c = row['ordinary_control']
            controls[key] = dict(ordered_ids_match=c['baseline_order_equal'],call_counts_match=c['baseline_calls_equal'],
                independent_paper_entry=score(row['labels'],dict.fromkeys(c['baseline_accepted_ids'],1)))
        arms = row['endpoints']
        bank[key] = [[*(arms[a]['metrics'][v] for v in METRICS),*(arms[a][v] for v in COSTS)] for a in ENDPOINTS]
    expected = {(k,seed,db) for k,panel in CURVE_SEEDS.items() for seed in panel for db in CURVE_DB[k]}
    if bank.keys()!=expected:
        raise ValueError(f'Incomplete curve, missing {len(expected-bank.keys())} cells')
    result = {}
    for k,seeds in CURVE_SEEDS.items():
        snrs = CURVE_DB[k]
        db = np.asarray(snrs)/100
        n = len(seeds)
        values = np.asarray([[bank[k,seed,snr] for snr in snrs] for seed in seeds])
        raw = values[:,:,:,:len(METRICS)].astype(np.int64)
        # Same frame-resampling multiplicities for all SNRs, receivers and declared primary/secondary outputs at one load.
        weights = np.random.default_rng([939039,k]).multinomial(n,np.full(n,1/n),size=BOOTSTRAPS)
        boot = (weights @ raw[:,:,:,:2].reshape(n,-1)).reshape(BOOTSTRAPS,len(snrs),3,2)
        totals = raw.sum(axis=0)
        curve = {}
        for j,snr in enumerate(snrs):
            point = {}
            for i,a in enumerate(ENDPOINTS):
                arm = {k0:int(v) for k0,v in zip(METRICS,totals[j,i],strict=True)}
                for t,m in enumerate(('list_pupe','count_pupe')):
                    arm[m] = float(totals[j,i,t]/(k*n))
                    band = np.quantile(boot[:,j,i,t]/(k*n),[.025,.975])
                    arm[m+'_pointwise_95'] = band.tolist()
                for t,key in enumerate(COSTS,start=len(METRICS)):
                    vs = values[:,j,i,t]
                    arm[key] = dict(total=float(vs.sum()),mean=float(vs.mean()),p50=float(np.median(vs)),
                                    p95=float(np.quantile(vs,.95)),maximum=float(vs.max()))
                point[a] = arm
            pairs = {}
            for a,b in ((0,2),(1,2),(0,1)):
                diff = raw[:,j,a,1]-raw[:,j,b,1]
                pairs[f'{ENDPOINTS[a]}_minus_{ENDPOINTS[b]}'] = dict(count_misses_saved=int(diff.sum()),
                    wins=int((diff>0).sum()),ties=int((diff==0).sum()),losses=int((diff<0).sum()),
                    count_pupe_saved_pointwise_95=np.quantile((boot[:,j,a,1]-boot[:,j,b,1])/(n*k),[.025,.975]).tolist())
            curve[str(snr)] = dict(endpoints=point,paired=pairs)
        result[str(k)] = dict(frames_per_snr=n,users_per_snr=k*n,snr_cdb=snrs,by_snr=curve,
            independent_entry_checks=[dict(seed=seed,snr_cdb=snr,**info) for (load,seed,snr),info in sorted(controls.items()) if load==k],
            unique_source_repeated_messages=repeats[k,'repeated_messages'],unique_source_repeated_patterns=repeats[k,'repeated_patterns'],
            crossings={m:crossing_analysis(totals[:,:,t],boot[:,:,:,t],db,k,n) for t,m in enumerate(('list_pupe','count_pupe'))})
    return dict(loads=result,bootstrap_replicates=BOOTSTRAPS,
        inference='Resample independent complete geometries, paired across SNR and receivers within load. Raw empirical curves, '
                  'piecewise linear interpolation only between measured adjacent levels; censored or ambiguous replicates retain unbounded tails. '
                  'Null interval endpoints are unbounded. Primary native-minus-growth list-PUPE crossing has two-load Bonferroni bounds. '
                  'Grid interpolation carries additional unquantified model error. Native and rollback clocks are common REP prefixes; '
                  'growth clock is an external cold receiver measurement. Geometry construction is separately charged to each endpoint.')
