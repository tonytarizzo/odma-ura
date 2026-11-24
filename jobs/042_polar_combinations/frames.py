"""Cold receiver prefixes, original DEC-35 and native SIC on identical physical frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'jobs/037_polar_tractable')]
import rollback_frames
from rollback_frames import geometry, make_frame
from framework.polar_catalogue import run as catalogue
from framework.polar_count_growth import run as scalar_prefix
from framework.polar_held_interference import prefix as held_prefix
from framework.polar_tractable import PolarScenario, integer

OUT = ROOT/'results/dec42_study'
PANELS = {'development': [*range(973001,973017),*range(973049,973065)],
          'A':list(range(984001,984065)), 'B':list(range(984101,984165))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    temporary = path.with_suffix('.partial')
    with temporary.open('x') as file:
        json.dump(data,file,separators=(',',':'))
    temporary.replace(path)


def frame(seed, ebn0=.5):
    start = time.perf_counter()
    m,_ = geometry(seed)
    elapsed = time.perf_counter()-start
    if ebn0==.5:
        s,labels = make_frame(m,seed,0)
    else:
        rng = np.random.default_rng([671129,seed,0])
        variance = 1/(200*10**(ebn0/10))
        labels = [integer(v) for v in rng.integers(2,size=(100,100),dtype=np.uint8)]
        y = rng.normal(0.,np.sqrt(variance),m.n)
        for w in labels:
            p,x = m.symbols(w)
            y[m.positions[p]] += x
        s = PolarScenario(m,y,100,variance)
    return s,labels,elapsed


def counts(action, labels):
    assert all(type(w) is int and 0<=w<1<<100 and type(c) is int and c>0 for w,c in action.items())
    assert sum(action.values())<=len(labels)
    estimate,truth=Counter(action),Counter(labels)
    return dict(misses=(truth-estimate).total(),false=(estimate-truth).total(),action=sorted(action.items()))


def collect(seed, ebn0=.5, original=True):
    s,labels,g = frame(seed,ebn0)
    sy = hashlib.sha256(s.y.tobytes()).hexdigest()
    result = dict(seed=seed,ebn0=ebn0,y=sy,labels=labels,geometry_seconds=g,prefix={})
    if original:
        provider = rollback_frames.native
        def observed(s):
            accepted, meta = provider(s)
            result['native'] = dict(**counts({int(w):1 for w in accepted},labels),seconds=meta['seconds'],calls=meta['calls'])
            return accepted,meta
        with patch.object(rollback_frames,'native',observed):
            start=time.perf_counter()
            output,meta=catalogue('DEC-35',s)
            elapsed=time.perf_counter()-start
        assert meta['continuation']['complete']
        result['original']=dict(**counts(output,labels),seconds=elapsed)
    for key in (('scalar','held') if seed%2 else ('held','scalar')):
        start=time.perf_counter()
        output,meta=(scalar_prefix if key=='scalar' else held_prefix)(s)
        elapsed=time.perf_counter()-start
        assert meta['complete'] and set(output)<=set(meta['bank'])
        result['prefix'][key]=dict(output=sorted(output.items()),bank=meta['bank'],cancellation_action=meta['cancellation_action'],
            complete=True,seconds=elapsed,native_calls=meta['native']['calls'],native_rounds=meta['native']['rounds'],
            continuation_calls=meta['continuation']['actual_scl_calls'])
    assert sy==hashlib.sha256(s.y.tobytes()).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('panel',choices=PANELS);p.add_argument('task',choices=('prepare','run'))
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    out=OUT/args.panel;seeds=PANELS[args.panel]
    names=['framework/polar_held_interference.py','framework/polar_count_growth.py','framework/polar_count_cavity.py',
           'jobs/037_polar_tractable/rollback_frames.py',str(Path(__file__).relative_to(ROOT))]
    sources={path:digest(ROOT/path) for path in names}
    if args.task=='prepare':
        out.mkdir(parents=True,exist_ok=False)
        write(out/'prefix_freeze.json',dict(seeds=seeds,sources=sources));return
    assert read(out/'prefix_freeze.json')==dict(seeds=seeds,sources=sources)
    for seed in seeds[args.worker::args.workers]:
        path=out/f'prefix_{seed}.json'
        if path.exists():raise FileExistsError(path)
        write(path,collect(seed,original=args.panel!='development'));print(seed,flush=True)


if __name__=='__main__':
    main()
