"""Matched flat-encoder paper curves for the prospectively selected complete catalogue receiver."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2];JOB=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'jobs/039_polar_paper'),str(ROOT/'jobs/037_polar_tractable')]
import rollback_frames
from framework.polar_catalogue import REGISTRY,run
from paper_curve import CURVE_DB,CURVE_SEEDS,ENDPOINTS,digest,make_frame,make_geometry,score
from paper_curve_analysis import crossing_analysis
from run_study import save_once,sha,release as parent_release

OUT=ROOT/'results/dec42_paper_curve'
METRICS=('count_misses','list_misses','false_copies')
JOINED='DEC-39'


def arm_names(k,decoder):
    return ('native','original',decoder,JOINED) if k==100 else ('native','original',decoder)


def plans():
    return [[(k,seed,db) for seed in panel[i:i+16]]
            for k in (100,50) for panel in [list(CURVE_SEEDS[k])] for db in CURVE_DB[k] for i in range(0,len(panel),16)]


def balance_plans():
    seeds=list(CURVE_SEEDS[100])[:64]
    return [[(100,seed,db) for seed in seeds[i:i+16]] for db in (25,40,50) for i in range(0,64,16)]


def release():
    manifest=JOB/'curve_inputs.json';config=json.loads(manifest.read_bytes())
    if config['039_release']!=parent_release():raise ValueError("Frozen Job 039 geometry reference changed")
    for path,digest in config['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError(f"Changed scientific input {path}")
    if config.get('k100_joined')!=JOINED or JOINED not in REGISTRY or config['decoder'] not in REGISTRY:raise ValueError("Selected complete receiver absent from catalogue")
    return config['decoder'],sha(manifest)


def case(k,seed,db,decoder,encoding='flat'):
    if encoding=='flat':model,geometry=make_geometry(k,seed)
    elif encoding=='balanced' and k==100:
        start=time.perf_counter();model,_=rollback_frames.geometry(seed);geometry=time.perf_counter()-start
    else:raise ValueError("Unsupported encoder/load")
    s,truth,fingerprint=make_frame(model,k,seed,db);arms={}
    def store(key,action,sec,calls):
        arms[key]=dict(action=sorted(action.items()),metrics=score(truth,action),seconds=sec,with_geometry=sec+geometry,native_calls=calls)
    order=arm_names(k,decoder);shift=seed%len(order)
    for key in order[shift:]+order[:shift]:
        start=time.perf_counter()
        if key=='native':
            accepted,m=rollback_frames.native(s)
            store(key,{int(w):1 for w in accepted},time.perf_counter()-start,m['calls'])
            continue
        a,m=run('DEC-35' if key=='original' else key,s)
        elapsed=time.perf_counter()-start
        if not m.get('complete',m.get('continuation',{}).get('complete',False)):
            raise RuntimeError(f"Incomplete {key} receiver")
        store(key,a,elapsed,m['native']['calls'])
    return dict(k=k,seed=seed,db=db,encoding=encoding,source=fingerprint,labels=truth,geometry_seconds=geometry,arms=arms)


def file_for(key,encoding='flat'):
    k,seed,db=key
    return OUT/('frames' if encoding=='flat' else 'balanced')/f'k{k}_s{seed}_db{db}.json'


def check(row,key,dec,rev,encoding='flat'):
    if row['encoding']!=encoding:raise ValueError("Wrong physical encoder")
    if row['release']!=rev or row['decoder']!=dec or (row['k'],row['seed'],row['db'])!=key:raise ValueError("Wrong case/release")
    k,_,db=key
    if len(row['labels'])!=k or row['source']['variance']!=1/(200*10**(db/1000)):raise ValueError("Wrong source or noise")
    labels=row['labels'];source=row['source']
    if any(type(w) is not int or not 0<=w<1<<100 for w in labels):raise ValueError("Invalid B100 message")
    binary=np.unpackbits(np.array([list(w.to_bytes(13,'big')) for w in labels],dtype=np.uint8),axis=1)[:,4:]
    if source['bits_sha256']!=digest(binary) or source['repeated_messages']!=k-len(set(labels)):
        raise ValueError("Unmatched source message counts")
    if source['repeated_patterns']!=k-len({w>>(88 if k==50 else 87) for w in labels}):
        raise ValueError("Unmatched source placement counts")
    if any(len(source[f'{k}_sha256'])!=64 for k in ('bits','standard_noise','positions','amplitudes','information','y')):
        raise ValueError("Missing observation fingerprint")
    if set(row['arms'])!=set(arm_names(k,dec)):raise ValueError("Incomplete arm set")
    for arm in row['arms'].values():
        action=dict(arm['action'])
        if sum(action.values())>k or any(type(w) is not int or not 0<=w<1<<100 or type(v) is not int or v<=0 for w,v in action.items()):
            raise ValueError("Invalid complete count action")
        if len(action)!=len(arm['action']) or score(row['labels'],action)!=arm['metrics']:raise ValueError("Inconsistent count metric")
        if arm['seconds']<=0 or arm['with_geometry']!=arm['seconds']+row['geometry_seconds']:raise ValueError("Incomplete charged time")


def summarize(dec,rev):
    report={}
    for k in (100,50):
        seeds=list(CURVE_SEEDS[k]);db=CURVE_DB[k];names=arm_names(k,dec)
        data=np.zeros((len(seeds),len(db),len(names),len(METRICS)),dtype=int);seconds=np.zeros(data.shape[:-1])
        physical={};seen=set()
        for i,seed in enumerate(seeds):
            for j,v in enumerate(db):
                key=(k,seed,v);r=json.loads(file_for(key).read_bytes());check(r,key,dec,rev)
                token=tuple(r['source'][f'{nm}_sha256'] for nm in ('bits','standard_noise','positions','amplitudes','information'))
                if physical.setdefault(seed,token)!=token:raise ValueError("Within-seed physical mismatch across energies")
                y=r['source']['y_sha256']
                if y in seen:raise ValueError("Duplicate channel observation")
                seen.add(y)
                for a,name in enumerate(names):
                    z=r['arms'][name];data[i,j,a]=[z['metrics'][m] for m in METRICS];seconds[i,j,a]=z['with_geometry']
        rng=np.random.default_rng(423991+k);boot=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=2000)
        full=data.sum(0);axis=np.asarray(db)/100;crossings={};diffs={}
        for metric in ('count_misses','list_misses'):
            m=METRICS.index(metric);curves=np.einsum('bi,ija->bja',boot,data[:,:,:,m])
            comparisons=[(0,1,2)] if k==50 else [(0,1,2),(0,1,3),(0,2,3)]
            for subgroup in comparisons:
                subnames=[names[i] for i in subgroup]
                analysis=crossing_analysis(full[:,subgroup,m],curves[:,:,subgroup],axis,k,len(seeds))
                for old,new in zip(ENDPOINTS,subnames,strict=True):
                    crossings[metric+'_'+new]=analysis['thresholds'][old]
                for left,right in ((0,2),(1,2),(0,1)):
                    old=f'{ENDPOINTS[left]}_minus_{ENDPOINTS[right]}'
                    key=f'{metric}_{subnames[left]}_minus_{subnames[right]}'
                    # A two-load band applies only to a comparison available at both loads.
                    value=analysis['required_energy_contrasts'][old].copy()
                    if JOINED in (subnames[left],subnames[right]):value.pop('bonferroni_two_loads_97_5_db',None)
                    if key not in diffs:diffs[key]=value
        report[k]=dict(frame_count=len(seeds),metric_names=METRICS,arms=names,db=axis.tolist(),counts=full.tolist(),
                       seconds=seconds.sum(0).tolist(),crossings=crossings,paired_db_saved=diffs)
    return report


def balance_report(dec,rev):
    summary={};rng=np.random.default_rng(42802)
    for db in (25,40,50):
        rows=[]
        for seed in CURVE_SEEDS[100][:64]:
            key=(100,seed,db);pair=[]
            for enc in ('flat','balanced'):
                r=json.loads(file_for(key,enc).read_bytes());check(r,key,dec,rev,enc);pair.append(r)
            flat,balanced=pair
            shared=('bits_sha256','standard_noise_sha256','positions_sha256','information_sha256','variance')
            if any(flat['source'][k]!=balanced['source'][k] for k in shared) or flat['labels']!=balanced['labels']:
                raise ValueError("Unmatched encoder comparison")
            if any(flat['source'][k]==balanced['source'][k] for k in ('y_sha256','amplitudes_sha256')):
                raise ValueError("Encoder comparison requires different transmitted amplitudes")
            rows.append(pair)
        by_arm={};draws=rng.integers(len(rows),size=(10000,len(rows)))
        for arm in arm_names(100,dec):
            v=np.array([[[r['arms'][arm]['metrics'][m] for m in METRICS] for r in pair] for pair in rows])
            d=v[:,0,0]-v[:,1,0]
            by_arm[arm]=dict(flat=v[:,0,:].sum(0).tolist(),balanced=v[:,1,:].sum(0).tolist(),metrics=METRICS,
                             wins=int(sum(d>0)),losses=int(sum(d<0)),saved=int(sum(d)),leave_best=int(sum(d)-max(0,*d)),
                             saved_pupe_percentage_points_95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
        summary[db]=by_arm
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--inventory',action='store_true');g.add_argument('--shard',type=int);g.add_argument('--merge',action='store_true')
    args=p.parse_args();batches=plans();dec,rev=release()
    if args.inventory:print(dec,rev,len(batches),'PBS tasks,',sum(map(len,batches)),'flat cases;',len(balance_plans()),'additional PBS tasks, 192 matched balanced cases')
    elif args.merge:
        report=summarize(dec,rev);save_once(OUT/'summary.json',dict(release=rev,decoder=dec,loads=report,paired_energy=balance_report(dec,rev)))
        if json.loads((OUT/'summary.json').read_bytes())['release']!=rev:raise ValueError("Stale merged result")
        print(OUT/'summary.json')
    else:
        encoding='flat' if args.shard<=len(batches) else 'balanced'
        batches=batches+balance_plans()
        if not 1<=args.shard<=len(batches):raise ValueError("Invalid PBS task")
        for key in batches[args.shard-1]:
            path=file_for(key,encoding)
            if not path.exists():
                row=case(*key,dec,encoding);row.update(release=rev,decoder=dec,host=platform.node(),job=os.environ.get('PBS_JOBID'))
                save_once(path,row)
            check(json.loads(path.read_bytes()),key,dec,rev,encoding);print(key,flush=True)
