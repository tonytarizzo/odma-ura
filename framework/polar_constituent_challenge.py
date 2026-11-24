"""One fitted-copy challenge using a normalized constituent projection of the original B100 CRC/polar code."""

from collections import Counter
from functools import lru_cache
import hashlib
import math
import time

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode, polar_transform
from framework.polar_count_cavity import cavity_query
from framework.polar_count_feedback import support_completion
from framework.polar_full_pattern import CountBank
from framework.polar_two_raw import run as two_raw


@lru_cache(maxsize=1)
def _geometry():
    code = PolarCode(87, 512, 16, 4096)
    generator = code.encode(np.eye(87, dtype=np.uint8)).reshape(87, 8, 64)
    positions = code.information[:11] - 192
    if np.any(positions < 0) or np.any(positions >= 64) or code.information[11] != 319:
        raise ValueError("Unexpected first-constituent or repetition position")
    inputs = np.zeros((2048, 64), dtype=np.uint8)
    inputs[:, positions] = (np.arange(2048)[:, None] >> np.arange(10, -1, -1)) & 1
    sequences = polar_transform(inputs)
    basis = sequences[1 << np.arange(10, -1, -1)]
    columns = []
    for j in range(64):
        values = np.packbits(generator[:, :, j], axis=1).ravel().tolist()
        future = {0}
        for key in values[12:]:
            if key not in future:
                future.update([word ^ key for word in future])
        a = values[int(np.flatnonzero(basis[:, j])[0])]
        h = values[11]
        if len(future) != (4 if j == 0 else 8) or any((x ^ (a if b else 0)) not in future for x, b in zip(values[:11], basis[:, j])):
            raise ValueError("Column images are incompatible with the B100 constituent projection")
        groups = np.array([sorted(word ^ phase for word in future) for phase in (0, a, h, a ^ h)], dtype=np.uint8)
        if len(np.unique(groups)) != groups.size:
            raise ValueError("Repetition and first-constituent sectors must be disjoint")
        columns.append(np.unpackbits(groups[:, :, None], axis=2).astype(float))
    return code, sequences.astype(float), columns


def _select(scenario, bank, counts, residual):
    """Sum 4096 joined sectors; the all-zero difference representing the incumbent is excluded once."""
    code, sequences, columns = _geometry()
    model = scenario.model
    if type(model.code) is not PolarCode or (model.b, model.code.n, model.code.crc_bits) != (87, 512, 16):
        raise ValueError("The constituent challenge requires the original B100 CRC16/polar512 code")
    if not np.array_equal(model.code.information, code.information):
        raise ValueError("The constituent challenge requires the original polar information set")
    ids = sorted(counts)
    variance = scenario.noise_var + (scenario.k - counts.total()) / model.n
    fields = np.empty((len(ids), 8, 64))
    for i, word in enumerate(ids):
        index, phi = bank.atoms[word]
        fields[i] = (2 * (residual[index] + phi) * phi / variance).reshape(8, 64)
    tables = np.empty((len(ids), 64, 4))
    zero_tail = np.full(len(ids), -np.inf)
    for j, groups in enumerate(columns):
        local = (-fields[:, :, j] @ groups.reshape(-1, 8).T).reshape(len(ids), 4, groups.shape[1])
        tables[:, j] = logsumexp(local, axis=2)
        # Within sector (0,0) the sorted first difference is zero. Compute the other atoms without cancellation.
        tail = logsumexp(local[:, 0, 1:], axis=1)
        zero_tail = np.logaddexp(tail, zero_tail + np.logaddexp(0., tail))
    sector_sums = np.empty((len(ids), 2, len(sequences)))
    for h in (0, 1):
        base = tables[:, :, 2 * h]
        sector_sums[:, h] = base.sum(axis=1)[:, None] + (tables[:, :, 2 * h + 1] - base) @ sequences.T
    sector_sums[:, 0, 0] = zero_tail
    relative = logsumexp(sector_sums.reshape(len(ids), -1), axis=1)
    # Each of 2^87-1 drawn alternatives has a uniform non-incumbent difference in the dimension-203 supercode.
    odds = relative + math.log((1 << 87) - 1) - math.log((1 << 203) - 1) + np.log([counts[w] for w in ids])
    best = int(np.argmax(odds))
    return ids[best], float(odds[best])


def finish(scenario, counts, meta):
    """Restore and challenge the selected fitted copy once, then recompute counts on the joint physical bank."""
    start = time.perf_counter()
    if not meta["complete"]:
        raise ValueError("A constituent challenge requires a converged prefix")
    fitted = Counter(dict(meta["cancellation_action"]))
    if fitted.total() > scenario.k:
        raise ValueError("Fitted count exceeds the known load")
    trace = dict(issued=False)
    if fitted:
        bank = CountBank(scenario, meta["bank"])
        residual = scenario.y - bank.matrix @ bank.vector(fitted)
        word, score = _select(scenario, bank, fitted, residual)
        p = scenario.model.split(word)[0]
        llr, _ = cavity_query(scenario, fitted, residual, p, word=word)
        code, _, _ = _geometry()
        payloads, metrics = code.decode_list(llr)
        ids = [scenario.model.word(p, payload) for payload in payloads]
        fresh, _ = bank.extend(ids)
        if fresh:
            fitted, work = bank.refit(fitted)
            if not work["converged"]:
                raise RuntimeError("Constituent challenge count refit exhausted its move cap")
        counts, completion = support_completion(scenario, fitted, bank.words)
        meta = {**meta, "bank":bank.words, "cancellation_action":sorted(fitted.items()), "support_completion":completion}
        trace.update(issued=True, selected=word, log_odds=score, pattern=p, ids=ids, new_ids=fresh, list_metrics=metrics.tolist(),
                     llr_sha256=hashlib.sha256(llr.tobytes()).hexdigest())
    return dict(counts), {**meta, "constituent_challenge":trace, "constituent_challenge_seconds":time.perf_counter()-start}


def run(scenario):
    start = time.perf_counter()
    counts, prefix = two_raw(scenario)
    output, meta = finish(scenario, counts, prefix)
    return output, {**meta, "total_seconds":time.perf_counter()-start}
