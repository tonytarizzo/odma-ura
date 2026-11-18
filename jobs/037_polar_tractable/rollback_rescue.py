"""Bounded B100 SIC rollback, unchanged-residual controls and restricted multiset decisions."""

from collections import Counter
from itertools import combinations_with_replacement
import math
import sys
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollback_frames import PolarScenario, Work, native
from baselines.polar import PolarCode
from framework.polar_tractable import sparse_readout


def prepare(s, accepted):
    start = time.perf_counter()
    m, residual, atoms = s.model, s.y.copy(), {}
    for w in accepted:
        p, values = m.symbols(w)
        atoms[w] = m.positions[p], values
        residual[atoms[w][0]] -= values
    # With unit columns the removal cost is (r.phi + 1/2)/variance, so its ranking needs only r.phi.
    products = {w: float(residual[idx] @ values) for w, (idx, values) in atoms.items()}
    weak = sorted(accepted, key=lambda w: (products[w], w))[:2]
    return residual, atoms, products, weak, time.perf_counter()-start


def polar_list(model, code, p, residual, variance):
    llr = 2 * model.amplitudes[p] * residual[model.positions[p]] / variance
    payloads, _ = code.decode_list(llr)
    return [model.word(p, v) for v in payloads]


def discoveries(s, prepared, arm, *, wall_budget=None):
    residual, atoms, _, weak, _ = prepared
    kleft = s.k - len(atoms)
    start = time.perf_counter()
    found, trace = set(), []
    code = s.model.code
    if arm == "rollback":
        drops = [(), weak[:1], weak[1:2], weak]
        limit = 16
    elif arm == "flat":
        drops, limit = [()], 64
    elif arm == "wide":
        drops, limit = [()], 16 if wall_budget is None else s.model.p
        code = PolarCode(s.model.b, code.n, code.crc_bits, 512)
    else:
        raise ValueError(arm)
    for released in drops:
        r = residual.copy()
        for w in released:
            idx, values = atoms[w]
            r[idx] += values
        variance = s.noise_var + max(kleft + len(released) - 1, 0) / s.model.n
        for p in Work(s.model).patterns(r, limit):
            ids = polar_list(s.model, code, p, r, variance)
            found.update(ids)
            elapsed = time.perf_counter() - start
            trace.append(dict(p=int(p), released=list(released), ids=ids, seconds=elapsed))
            if wall_budget is not None and elapsed >= wall_budget and len(trace) >= 16:
                break
    return found, trace, time.perf_counter()-start


def count_action(s, prepared, native_ids, fresh):
    residual, native_atoms, products, weak, _ = prepared
    m, prior, cache = s.model, Counter(native_atoms.keys()), dict(native_atoms)
    trusted = prior.copy()
    r = residual.copy()
    for w in weak:
        del trusted[w]
        idx, values = cache[w]
        r[idx] += values
    repeated = {w for w, dot in products.items() if dot > .5 + s.noise_var * math.log(2)}
    ids = set(weak) | (set(native_ids) - set(prior)) | set(fresh) | repeated
    for w in sorted(ids - cache.keys()):
        p, values = m.symbols(w)
        cache[w] = m.positions[p], values
    h = s.k - trusted.total()
    # Leave room for both original weak words before truncating the other complete IDs.
    dots = {w: float(r[cache[w][0]] @ cache[w][1]) for w in ids}
    rest = sorted(ids - set(weak), key=lambda w: (-dots[w]/s.noise_var + math.log1p(trusted[w]), w))
    words = list(weak) + rest[:10-len(weak)]
    if h > 6 or len(words) < h:
        # Partial action: preserve native decisions, add only favourable increments within the output budget.
        action, left = prior.copy(), residual.copy()
        for _ in range(s.k - action.total()):
            if not words:
                break
            scores = [(left[cache[w][0]] @ cache[w][1] - .5)/s.noise_var - math.log1p(action[w]) for w in words]
            j = int(np.argmax(scores))
            if scores[j] <= 0:
                break
            w = words[j]
            action[w] += 1
            left[cache[w][0]] -= cache[w][1]
        return action, {"mode": "partial", "states": 0, "h": h, "words": len(words)}
    # At most ten sparse columns interact. Their Gram matrix and residual correlations suffice for all scores.
    support, inverse = np.unique(np.concatenate([cache[w][0] for w in words]), return_inverse=True)
    matrix = np.zeros((len(words), len(support)))
    matrix[np.arange(len(words))[:, None], inverse.reshape(len(words), -1)] = np.stack([cache[w][1] for w in words])
    gram, field = matrix @ matrix.T, np.array([dots[w] for w in words])
    tuples = np.asarray(list(combinations_with_replacement(range(len(words)), h)), dtype=int)
    freq = np.zeros((len(tuples), len(words)), dtype=int)
    for column in tuples.T:
        freq[np.arange(len(freq)), column] += 1
    base = np.array([trusted[w] for w in words])
    logw = (freq @ field - .5 * np.sum((freq @ gram)*freq, axis=1))/s.noise_var - gammaln(1 + base + freq).sum(1)
    extras, _ = sparse_readout(tuples.tolist(), softmax(logw), h)
    action = trusted + Counter({words[j]: n for j, n in extras.items()})
    return action, {"mode": "conditional", "states": len(tuples), "h": h, "words": len(words)}


def run(scenario, *, arm="rollback"):
    accepted, baseline = native(scenario)
    prep = prepare(scenario, accepted)
    ids, trace, discovery_seconds = discoveries(scenario, prep, arm)
    action, readout = count_action(scenario, prep, baseline["valid_ids"], ids)
    return dict(action), {"native": baseline, "extra_polar_calls": len(trace), "discovery_seconds": discovery_seconds,
                          "selection_seconds": prep[-1], "readout": readout}
