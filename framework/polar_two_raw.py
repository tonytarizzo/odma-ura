"""DEC-35 terminal option using two ordinary wide-list searches with an intermediate count refit."""

from collections import Counter
import hashlib
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_cavity import cavity_query
from framework.polar_count_feedback import support_completion
from framework.polar_count_growth import run as count_growth
from framework.polar_full_pattern import CountBank


def finish(scenario, baseline, original):
    started = time.perf_counter()
    initial = Counter(dict(original["cancellation_action"]))
    if not original["complete"] or initial.total() > scenario.k:
        raise ValueError("Two raw queries require a complete prefix and a valid fitted count")
    meta = {**original, "terminal_mode": "two_raw", "terminal_queries": []}
    if not initial or initial.total() == scenario.k:
        meta["terminal_seconds"] = time.perf_counter() - started
        return dict(baseline), meta
    model = scenario.model
    code = PolarCode(model.b, model.code.n, model.code.crc_bits, 4096)
    if type(model.code) is not PolarCode or not np.array_equal(code.information, model.code.information):
        raise ValueError("Two raw queries require the original polar information set")
    bank = CountBank(scenario, original["bank"])
    fitted = initial
    for _ in range(2):
        residual = scenario.y - bank.matrix @ bank.vector(fitted)
        magnitude = np.abs(residual[model.positions]).sum(axis=1)
        if meta["terminal_queries"] and fitted == initial:
            magnitude[meta["terminal_queries"][0]["pattern"]] = -np.inf
        pattern = int(np.argmax(magnitude))
        llr, context = cavity_query(scenario, fitted, residual, pattern)
        payloads, metrics = code.decode_list(llr)
        ids = [model.word(pattern, payload) for payload in payloads]
        fresh, _ = bank.extend(ids)
        if fresh:
            fitted, log = bank.refit(fitted)
            if not log["converged"]:
                raise RuntimeError("Wide-query count fit exhausted its move cap")
        meta["terminal_queries"].append(dict(pattern=pattern, ids=ids, list_metrics=metrics.tolist(), new_ids=fresh,
                                             variance=context["variance"], llr_sha256=hashlib.sha256(llr.tobytes()).hexdigest(),
                                             fit=sorted(fitted.items())))
    output, completion = support_completion(scenario, fitted, bank.words)
    meta.update(bank=bank.words, cancellation_action=sorted(fitted.items()), support_completion=completion,
                terminal_seconds=time.perf_counter()-started)
    return dict(output), meta


def run(scenario):
    started = time.perf_counter()
    baseline, original = count_growth(scenario)
    counts, meta = finish(scenario, baseline, original)
    meta["total_seconds"] = time.perf_counter() - started
    return counts, meta
