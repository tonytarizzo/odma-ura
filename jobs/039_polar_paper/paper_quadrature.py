"""Published-channel sensitivity of the native magnitude detector on paired flat packets."""

from collections import Counter
import time

import numpy as np

import paper_curve as c
from paper_curve_analysis import BOOTSTRAPS, crossing, interval, report_crossing
from framework.polar_repetition_scl import RepetitionPolarCode


def selected(k, seed):
    return seed in c.CURVE_SEEDS[k][:64]


def native(s, imaginary):
    model = s.model
    accepted, rounds = {}, []
    residual = s.y.copy()
    calls = 0
    for step in range(30):
        remaining = s.k - len(accepted)
        if not remaining:
            break
        magnitude = np.hypot(residual, imaginary)
        rank = np.argsort(-magnitude[model.positions].sum(1), kind="stable")[:remaining+5]
        variance = s.noise_var + (remaining-1) / model.n
        new = []
        for pattern in rank:
            llr = 2 * model.amplitudes[pattern] * residual[model.positions[pattern]] / variance
            payloads, _ = model.code.decode_list(llr)
            calls += 1
            if len(payloads):
                word = model.word(pattern,payloads[0])
                if word not in accepted:
                    new.append(word)
                    accepted[word] = step
            if len(accepted) == s.k:
                break
        rounds.append(len(new))
        if not new:
            break
        for word in new:
            pattern, values = model.symbols(word)
            residual[model.positions[pattern]] -= values
    return accepted, rounds, calls


def run_case(k, seed, db, real):
    model, setup = c.make_geometry(k,seed)
    s,labels,source = c.make_frame(model,k,seed,db)
    if labels != real["labels"] or source["y_sha256"] != real["source"]["y_sha256"]:
        raise ValueError("Quadrature control requires the same signed observations")
    z = np.random.default_rng([390274,seed,0]).standard_normal(model.n)
    imag = z*np.sqrt(s.noise_var)
    started = time.perf_counter()
    model.code = RepetitionPolarCode(model.b, model.code.n, model.code.crc_bits, 128)
    accepted, rounds, calls = native(s,imag)
    elapsed = time.perf_counter()-started
    action = dict.fromkeys(accepted,1)
    return dict(ordered=list(accepted),rounds=rounds,calls=calls,metrics=c.score(labels,action),
                standard_imaginary_sha256=c.digest(z),imaginary_sha256=c.digest(imag),
                receiver_seconds=elapsed,seconds_with_geometry=elapsed+setup)


def summarize(rows):
    data, noises = {}, {}
    for row in rows:
        k,seed,snr = row["k"],row["seed"],row["snr_cdb"]
        if not selected(k,seed):
            if "complex_screen" in row:
                raise ValueError("Undeclared complex screen")
            continue
        q = row["complex_screen"]
        if (q["metrics"] != c.score(row["labels"],dict.fromkeys(q["ordered"],1)) or
                sum(q["rounds"]) != len(q["ordered"]) or len(q["ordered"]) != len(set(q["ordered"])) or
                noises.setdefault((k,seed),q["standard_imaginary_sha256"])!=q["standard_imaginary_sha256"]):
            raise ValueError("Inconsistent complex screen trace")
        native_real, growth = (row["endpoints"][r]["metrics"] for r in ("native","growth"))
        key = k,seed,snr
        if key in data:
            raise ValueError("Duplicate complex screen case")
        data[key] = ([m["list_misses"] for m in (native_real,q["metrics"],growth)],
                     q["metrics"]["count_misses"],q["metrics"]["false_copies"],q["calls"],q["seconds_with_geometry"])
    out = {}
    names = ("native_real","native_complex","growth")
    for k in (50,100):
        seeds, grid = c.CURVE_SEEDS[k][:64], c.CURVE_DB[k]
        raw = np.asarray([[data[k,s,d][0] for d in grid] for s in seeds],dtype=np.int64)
        n = len(seeds)
        total = raw.sum(axis=0)
        mult = np.random.default_rng([390275,k]).multinomial(n,np.full(n,1/n),size=BOOTSTRAPS)
        boot = (mult @ raw.reshape(n,-1)).reshape(-1,len(grid),3)
        db = np.array(grid)/100
        point = [crossing(db,total[:,i],n*k) for i in range(3)]
        draws = [[crossing(db,rep[:,i],n*k) for i in range(3)] for rep in boot]
        thresholds = {}
        for i,name in enumerate(names):
            subset = [r[i] for r in draws]
            thresholds[name] = dict(**report_crossing(point[i]),bootstrap_status=dict(Counter(r[0] for r in subset)),
                bootstrap_95_db=interval([r[1] for r in subset],[r[2] for r in subset]))
        contrasts = {}
        for a,b in ((0,2),(1,2),(1,0)):
            lo,hi = point[a][1:3]
            other_lo,other_hi = point[b][1:3]
            contrasts[f"{names[a]}_minus_{names[b]}"] = dict(snr_db_saved_by_second=lo-other_lo if lo==hi and other_lo==other_hi else None,
                paired_bootstrap_95_db=interval([r[a][1]-r[b][2] for r in draws],[r[a][2]-r[b][1] for r in draws]))
        cells = {}
        for j,d in enumerate(grid):
            count_misses,false,calls,wall = np.sum([data[k,s,d][1:] for s in seeds],axis=0)
            cells[str(d)] = dict(list_misses=dict(zip(names,map(int,total[j]),strict=True)), complex_count_misses=int(count_misses),
                                complex_false_copies=int(false),complex_scl128_calls=int(calls),complex_seconds_with_geometry=float(wall))
        out[str(k)] = dict(frames=n, users_per_point=n*k,thresholds=thresholds,contrasts=contrasts,by_snr=cells)
    return out
