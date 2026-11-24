"""Recompute the balanced common-16 paper comparison from source, noise and catalogue entries."""
from argparse import ArgumentParser
from collections import Counter
import time

from paper_native import SEEDS,paper_curve,rollback_frames
from paper_soft import run as soft_data
from framework.polar_catalogue import run as catalogue


def replay(seed):
    t=time.perf_counter();model,_=rollback_frames.geometry(seed);geometry=time.perf_counter()-t
    s,labels=rollback_frames.make_frame(model,seed,0)
    def standard(sc):
        accepted,_=rollback_frames.native(sc)
        return dict.fromkeys(accepted,1)
    arms={'native':standard,'DEC-35':lambda sc:catalogue('DEC-35',sc)[0],
          'scalar_soft_five':lambda sc:soft_data(sc,full_user=True)[0]}
    arms.update({name:lambda sc,name=name:catalogue(name,sc)[0] for name in ('DEC-38','DEC-39')})
    records={}
    for name,fn in arms.items():
        t=time.perf_counter();out=fn(s);seconds=geometry+time.perf_counter()-t;score=paper_curve.score(labels,out)
        records[name]=dict(misses=score['count_misses'],false=score['false_copies'],seconds=seconds)
    return records


if __name__=='__main__':
    declared=[*SEEDS[:8],*SEEDS[16:24]]
    p=ArgumentParser();p.add_argument('--seed',type=int,choices=declared);args=p.parse_args()
    seeds=[args.seed] if args.seed else declared;totals={}
    for seed in seeds:
        row=replay(seed)
        for k,record in row.items():totals.setdefault(k,Counter()).update(record)
        print(seed,{k:z['misses'] for k,z in row.items()},flush=True)
    print('frames',len(seeds),'transmitted',100*len(seeds))
    for k,r in totals.items():print(k,'misses',r['misses'],'false',r['false'],'seconds',round(r['seconds'],2))
