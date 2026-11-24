"""Rate-triggered one-copy restoration after the count-growth and terminal magnitude decoder."""

from collections import Counter
import hashlib
import math
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_feedback import support_completion
from framework.polar_full_pattern import CountBank
from framework.polar_terminal_query import run as terminal


def _select(scenario, bank, counts, residual, variance):
    model = scenario.model
    free = np.asarray(model.keys, dtype=object) != 0
    log_two = math.log(2.)
    log_alternatives = model.b * log_two + math.log1p(-math.exp(-model.b * log_two))
    scores = {}
    for word, copies in counts.items():
        indices, phi = bank.atoms[word]
        signed = 2 * (residual[indices] + phi)[free] * phi[free] / variance
        information = free.sum() * log_two - np.logaddexp(0., -signed).sum()
        scores[word] = float(log_alternatives + math.log(copies) - information)
    word = min(scores, key=lambda w: (-scores[w], w))
    return word, scores[word]


def finish(scenario, counts, meta):
    """Challenge a fitted word after a completed discovery stage."""
    started = time.perf_counter()
    if not meta["complete"]:
        raise RuntimeError("The rate challenge requires a converged terminal count fit")
    fitted = Counter(dict(meta["cancellation_action"]))
    trace = dict(issued=False, G=None, selected=None)
    if fitted:
        model = scenario.model
        bank = CountBank(scenario, meta["bank"])
        residual = scenario.y - bank.matrix @ bank.vector(fitted)
        variance = scenario.noise_var + (scenario.k - fitted.total()) / model.n
        word, score = _select(scenario, bank, fitted, residual, variance)
        trace.update(G=score, selected=word)
        if score >= 0.:
            pattern = model.split(word)[0]
            indices, phi = bank.atoms[word]
            llr = 2 * model.amplitudes[pattern] * (residual[indices] + phi) / variance
            code = PolarCode(model.b, model.code.n, model.code.crc_bits, 4096)
            if not np.array_equal(code.information, model.code.information):
                raise ValueError("Incompatible wide-list polar information set")
            payloads, metrics = code.decode_list(llr)
            ids = [model.word(pattern, payload) for payload in payloads]
            fresh, _ = bank.extend(ids)
            if fresh:
                fitted, log = bank.refit(fitted)
                if not log["converged"]:
                    raise RuntimeError("Rate-challenge count fit exhausted its move cap")
            counts, completion = support_completion(scenario, fitted, bank.words)
            trace.update(issued=True, pattern=pattern, ids=ids, list_metrics=metrics.tolist(), new_ids=fresh,
                         llr_sha256=hashlib.sha256(llr.tobytes()).hexdigest())
            meta = {**meta, "bank": bank.words, "cancellation_action": sorted(fitted.items()), "support_completion": completion}
    return dict(counts), {**meta, "rate_challenge": trace, "rate_challenge_seconds": time.perf_counter() - started}


def run(scenario):
    started = time.perf_counter()
    counts, meta = terminal(scenario, selector="magnitude")
    counts, meta = finish(scenario, counts, meta)
    meta["total_seconds"] = time.perf_counter() - started
    return counts, meta
