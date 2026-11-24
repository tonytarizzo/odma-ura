"""DEC-35 count-growth option with a bounded ordinary SCL4096 scan of outstanding placements."""
from collections import Counter
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_cavity import cavity_query
from framework.polar_count_feedback import support_completion
from framework.polar_count_growth import run as count_growth
from framework.polar_full_pattern import CountBank


def finish(scenario, baseline, original):
    started = time.perf_counter()
    cancellation = Counter(dict(original["cancellation_action"]))
    remaining = scenario.k - cancellation.total()
    if not original["complete"] or not 0 <= remaining <= scenario.k:
        raise ValueError("Require a complete count-growth prefix and a valid fitted count")
    meta = {**original, "terminal_mode": "magnitude_deficit_scan4096", "terminal_queries": []}
    if remaining == 0:
        meta["terminal_seconds"] = time.perf_counter() - started
        return dict(baseline), meta

    model = scenario.model
    if type(model.code) is not PolarCode or model.code.list_size != 128:
        raise ValueError("Require the original ordinary128 polar encoder")
    bank = CountBank(scenario, original["bank"])
    residual = scenario.y - bank.matrix @ bank.vector(cancellation)
    magnitude = np.abs(residual[model.positions]).sum(axis=1)
    patterns = np.argsort(-magnitude, kind="stable")[:min(remaining + 1, 4)]
    code = PolarCode(model.b, model.code.n, model.code.crc_bits, 4096)
    candidates = []
    for index in patterns:
        pattern = int(index)
        llr, context = cavity_query(scenario, cancellation, residual, pattern)
        payloads, _ = code.decode_list(llr)
        ids = [model.word(pattern, payload) for payload in payloads]
        candidates.extend(ids)
        meta["terminal_queries"].append({"pattern": pattern, "ids": ids, "variance": context["variance"]})
    fresh, _ = bank.extend(candidates)
    fit, stats = bank.refit(cancellation) if fresh else (cancellation, None)
    if stats is not None and not stats["converged"]:
        raise ValueError("Terminal count refit failed to converge")
    output, completion = support_completion(scenario, fit, bank.words)
    meta.update(bank=sorted(bank.words), cancellation_action=sorted(fit.items()), support_completion=completion,
                terminal_new_ids=fresh, terminal_seconds=time.perf_counter() - started)
    return dict(output), meta


def run(scenario):
    """Return a multiplicity-aware count action and diagnostic metadata on the original balanced receiver contract."""
    started = time.perf_counter()
    baseline, original = count_growth(scenario)
    output, meta = finish(scenario, baseline, original)
    meta["total_seconds"] = time.perf_counter() - started
    return output, meta
