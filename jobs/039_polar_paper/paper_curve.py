"""Fixed flat, single-power GMAC comparison at the two low loads in Özates et al., Sec. IV."""

from collections import Counter
import hashlib
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'jobs/037_polar_tractable'))
from baselines.odma_polar import ODMAPolarBaseline
from framework.polar_decoders import run as registered
from framework.polar_tractable import PolarModel, PolarScenario, integer
from framework import polar_count_growth
import rollback_frames
import rollback_progressive

CURVE_DB = {50:[-80,-40,-20,-10,0,10,15,20,25,30,35,40,50,70], 100:[-10,0,15,25,30,35,40,45,50,55,65,80]}
CURVE_SEEDS = {50:list(range(932001,932257)), 100:list(range(931001,931257))}
CENTRAL_DB = {50:25, 100:45}
ENDPOINTS = ('native','rollback','growth')
OPTIONS = dict(native_repetition=True,memoize=True,complete_support=True,progress_extra_sweep=True,count_growth=True)


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def baseline(k, seed):
    if k not in (50,100):
        raise ValueError('Only the fixed K50 and K100 flat single-power configurations are declared')
    return ODMAPolarBaseline(100,30000,seed,prefix_bits={50:12,100:13}[k],code_length=512,crc_bits=16,list_size=128,
                             detector_slack=5,max_iterations=30,power_levels=(1.,))


def make_geometry(k, seed):
    start = time.perf_counter()
    block = baseline(k,seed)
    model = PolarModel(block.polar,block.positions,np.full(block.positions.shape,np.sqrt(1/512)),block.n)
    return model,time.perf_counter()-start


def make_frame(model, k, seed, snr_cdb):
    if type(snr_cdb) is not int or type(seed) is not int or k<1:
        raise ValueError('Expected integer seed and hundredths of a dB, and a positive user count')
    rng = np.random.default_rng([671129,seed,0])
    bits = rng.integers(2,size=(k,model.B),dtype=np.uint8)
    labels = [integer(row) for row in bits]
    z = rng.standard_normal(model.n)
    variance = 1/(2*model.B*10**(snr_cdb/1000))
    y = z*np.sqrt(variance)
    for w in labels:
        p,s = model.symbols(w)
        y[model.positions[p]] += s
    source = dict(bits_sha256=digest(bits),standard_noise_sha256=digest(z),y_sha256=digest(y),
                  positions_sha256=digest(model.positions),amplitudes_sha256=digest(model.amplitudes),
                  information_sha256=digest(model.code.information),variance=variance,
                  repeated_messages=k-len(set(labels)),repeated_patterns=k-len({w>>model.b for w in labels}))
    return PolarScenario(model,y,k,variance),labels,source


def score(labels, action):
    truth = Counter(labels)
    if any(type(w) is not int or not 0<=w<1<<100 or type(c) is not int or c<1 for w,c in action.items()):
        raise ValueError('Expected positive integer counts of actual 100-bit messages')
    if sum(action.values())>len(labels):
        raise ValueError('Output copy count exceeds known K')
    matched = (truth & Counter(action)).total()
    return dict(list_misses=sum(n for w,n in truth.items() if w not in action),count_misses=len(labels)-matched,
                false_copies=sum(action.values())-matched,foreign_ids=len(set(action)-set(truth)),
                output_copies=sum(action.values()),list_size=len(action))


def with_endpoints(s):
    accepted_rounds = []
    normal = rollback_frames.native

    def observe(sc):
        accepted,meta = normal(sc)
        accepted_rounds.append(list(accepted.items()))
        return accepted,meta

    start = time.perf_counter()
    with patch.object(rollback_frames,'native',observe):
        out,meta = registered('DEC-35',s,**OPTIONS)
    full_wall = time.perf_counter()-start
    if len(accepted_rounds)!=1 or not meta['complete'] or meta['dec35_options']!=OPTIONS:
        raise RuntimeError('Incomplete count-growth run or incompatible registered provider')
    accepted = accepted_rounds[0]
    native,rollback,cont = meta['native'],meta['rollback_batches'],meta['continuation']
    if sum(native['rounds'])!=len(accepted) or len(rollback)!=4 or any(len(b['patterns'])!=8 for b in rollback):
        raise RuntimeError('Invalid native or four-by-eight rollback stage')
    fitted = [cont['initial_refit']]+[r['meta'] for r in cont['refits']]
    if not all(f['converged'] for f in fitted):
        raise RuntimeError('Incomplete count-bank fit')
    actions = {'native':{w:1 for w,r in accepted},'rollback':dict(meta['rollback_action']),'growth':out}
    native_seconds = native['seconds']+meta['native_repetition_preparation_seconds']
    costs = dict(native=dict(receiver_seconds=native_seconds,native_scl128=native['calls'],rollback_scl512=0,continuation_scl128=0),
        rollback=dict(receiver_seconds=native_seconds+meta['rollback_seconds'],native_scl128=native['calls'],rollback_scl512=32,
                      continuation_scl128=0),
        growth=dict(receiver_seconds=full_wall,native_scl128=native['calls'],rollback_scl512=32,continuation_scl128=cont['actual_scl_calls']))
    stages = dict(native_order_round=accepted,native_calls=native['calls'],native_round_additions=native['rounds'],native_valid_ids=native['valid_ids'],
        native_residual_sha256=native['residual_sha256'],native_rep_setup_seconds=meta['native_repetition_preparation_seconds'],
        native_seconds=native['seconds'],rollback_seconds=meta['rollback_seconds'],rollback_batches=rollback,
        base_bank=meta['base_bank'],bank=meta['bank'],cancellation_action=meta['cancellation_action'],
        terminal_additions=meta['support_completion']['additions'],terminal_seconds=meta['support_completion']['seconds'],
        continuation_seconds=cont['seconds'],logical_continuation=len(cont['queries']),cache_hits=cont['cache_hits'],
        continuation_scl_seconds=cont['scl_seconds'],count_refit_seconds=cont['count_refit_seconds'],
        query_ranking_seconds=cont['ranking_seconds'],candidate_gain_evaluations=sum(f['candidate_gain_evaluations'] for f in fitted),
        gram_rows_constructed=sum(f['gram_rows_constructed'] for f in fitted),count_refits=len(fitted),
        count_moves=sum(len(f['moves']) for f in fitted),continuation_sweeps=len(cont['sweeps']),stop=cont['stop'],
        cache_bytes=cont['cache_bytes'],bank_sparse_bytes=cont['bank_sparse_bytes'])
    if stages['logical_continuation']!=costs['growth']['continuation_scl128']+stages['cache_hits']:
        raise RuntimeError('Continuation work did not reconcile')
    return actions,costs,stages


def verify_ordinary(s, block, actions, stages):
    start = time.perf_counter()
    accepted,native_meta = rollback_frames.native(s)
    native_seconds = time.perf_counter()-start
    if (list(accepted.items())!=stages['native_order_round'] or stages['native_valid_ids']!=native_meta['valid_ids']
            or stages['native_calls']!=native_meta['calls']):
        raise RuntimeError('Ordinary native ordered output differs')
    if native_meta['rounds']!=stages['native_round_additions'] or native_meta['residual_sha256']!=stages['native_residual_sha256']:
        raise RuntimeError('Ordinary native cancellation differs')
    start = time.perf_counter()
    recovered,m = rollback_progressive.run(s,mode='ungated')
    rollback_seconds = time.perf_counter()-start
    if recovered!=actions['rollback'] or m['batches']!=stages['rollback_batches'] or m['extra_calls']!=32:
        raise RuntimeError('Standalone ordinary rollback differs')
    if not np.array_equal(block.positions,s.model.positions):
        raise RuntimeError('Standalone single-power encoder differs')
    start = time.perf_counter()
    bits,bmeta = block.decode(s.y,s.k,s.noise_var)
    baseline_seconds = time.perf_counter()-start
    ids = [integer(row) for row in bits]
    # The independent paper entry groups floating cancellation sums by round; record any resulting branch difference.
    return dict(passed=True,ordinary_native_seconds=native_seconds,ordinary_rollback_seconds=rollback_seconds,
                baseline_seconds=baseline_seconds,ordinary_native_calls=native_meta['calls'],accepted_ids=list(accepted),
                baseline_order_equal=ids==list(accepted),baseline_calls_equal=bmeta['polar_trials']==native_meta['calls'],
                baseline_accepted_ids=ids,baseline_scl128_calls=bmeta['polar_trials'])


def audit_cell(k,seed,snr_cdb):
    return seed in (CURVE_SEEDS[k][0],CURVE_SEEDS[k][-1]) and snr_cdb in (CURVE_DB[k][0],CENTRAL_DB[k],CURVE_DB[k][-1])


def verify_physical(s, block, labels, seed):
    rng = np.random.default_rng([671129,seed,0])
    bits = rng.integers(2,size=(s.k,100),dtype=np.uint8)
    packed = [int.from_bytes(row.tobytes(),'big')>>4 for row in np.packbits(bits,axis=1)]
    if labels!=packed:
        raise RuntimeError('Independent big-endian source reconstruction differs')
    payload = bits[:,block.prefix_bits:]
    state = np.zeros(s.k,dtype=np.uint32)
    for v in payload.T:
        carry = (state>>15)^v
        state = ((state<<1)&65535)^(carry*0x1021)
    checkbits = ((state[:,None]>>np.arange(15,-1,-1))&1).astype(np.uint8)
    code = np.zeros((s.k,512),dtype=np.uint8)
    code[:,block.polar.information] = np.concatenate([payload,checkbits],axis=1)
    for b in range(9):
        left = np.flatnonzero((np.arange(512)&(1<<b))==0)
        code[:,left] ^= code[:,left|(1<<b)]
    z = rng.standard_normal(block.n)
    y = z*np.sqrt(s.noise_var)
    for w,c in zip(packed,code,strict=True):
        y[block.positions[w>>payload.shape[1]]] += (1.-2.*c)*np.sqrt(1/512)
    if not np.array_equal(s.y,y):
        raise RuntimeError('Independent CRC, polar and physical reconstruction differs')
    return digest(y)


def run_curve_frame(k, seed, snr_cdb):
    from paper_terminal import terminal_endpoint
    if k not in CURVE_DB or seed not in CURVE_SEEDS[k] or snr_cdb not in CURVE_DB[k]:
        raise ValueError('Unknown paper performance-curve cell')
    model,setup = make_geometry(k,seed)
    start = time.perf_counter()
    s,labels,source = make_frame(model,k,seed,snr_cdb)
    source_seconds = time.perf_counter()-start
    actions,costs,stages = with_endpoints(s)
    terminal = terminal_endpoint(s,actions['growth'],stages,costs['growth'],setup)
    terminal['metrics'] = score(labels,dict(terminal['action']))
    control = None
    if audit_cell(k,seed,snr_cdb):
        block = baseline(k,seed)
        control = verify_ordinary(s,block,actions,stages)
        control['independent_y_sha256'] = verify_physical(s,block,labels,seed)
    if digest(s.y)!=source['y_sha256']:
        raise RuntimeError('Receiver mutated the physical observation')
    endpoints = {a:dict(action=sorted(actions[a].items()),metrics=score(labels,actions[a]),**costs[a],
                        seconds_with_geometry=costs[a]['receiver_seconds']+setup) for a in ENDPOINTS}
    return dict(study='paper_curve',k=k,seed=seed,snr_cdb=snr_cdb,labels=labels,source=source,
                geometry_seconds=setup,source_seconds=source_seconds,endpoints=endpoints,stages=stages,ordinary_control=control,
                terminal_magnitude=terminal)


def summarize_curve(rows):
    from paper_curve_analysis import summarize
    return summarize(rows)
