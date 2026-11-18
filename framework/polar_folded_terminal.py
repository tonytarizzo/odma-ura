"""One terminal wide-list query when the fitted count leaves one slot."""

from collections import Counter
import time

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode
from framework.polar_count_cavity import cavity_query
from framework.polar_count_feedback import support_completion
from framework.polar_full_pattern import CountBank
from framework.polar_tractable import independent


def signed_prefix_end(model):
    """Use a common physical mask for which every visible nonconstant sign is independent given its pattern."""
    if model.keys[0] != 0 or 0 in model.keys[1:] or np.any(np.diff(model.positions, axis=1) <= 0):
        raise ValueError("Require sorted placements and a unique deterministic first code coordinate")
    end = int(model.positions[:, 0].max())
    pivots = {}
    for i, key in enumerate(model.keys[1:], 1):
        if not independent(key, pivots):
            return min(end, int(model.positions[:, i].min()) - 1)
    return end


def folded_log_bf(model, residual, variance, end=None):
    """Pattern likelihood relative to pure noise for one IID slot, all magnitudes and signed resources through end."""
    if np.shape(residual) != (model.n,) or not np.isfinite(residual).all() or not np.isfinite(variance) or variance <= 0:
        raise ValueError("Require a finite physical residual and positive noise variance")
    safe_end = signed_prefix_end(model)
    end = safe_end if end is None else end
    if not -1 <= end <= safe_end:
        raise ValueError("Signed prefix exceeds the verified independent code projection")
    scores = np.empty(model.p)
    for start in range(0, model.p, 128):
        stop = min(start + 128, model.p)
        positions, amplitudes = model.positions[start:stop], model.amplitudes[start:stop]
        field = amplitudes * residual[positions] / variance
        terms = np.logaddexp(field, -field) - np.log(2.)
        terms[:, 0] = np.where(positions[:, 0] <= end, field[:, 0], terms[:, 0])
        scores[start:stop] = (terms - amplitudes**2 / (2 * variance)).sum(axis=1)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite coarsened likelihood")
    return scores, end


def terminal_query(scenario, cancellation, ids, mode):
    """Continue from an existing complete CountBank endpoint, without using labels or excluding known words."""
    started = time.perf_counter()
    if mode not in ("magnitude", "folded") or scenario.k - sum(cancellation.values()) != 1:
        raise ValueError("Require one fitted slot left and a declared pattern selector")
    model = scenario.model
    bank = CountBank(scenario, ids)
    counts = bank.vector(cancellation)
    raw = scenario.y - bank.matrix @ counts
    setup = time.perf_counter() - started
    tick = time.perf_counter()
    end = None
    top_probability = None
    if mode == "magnitude":
        scores = np.abs(raw[model.positions]).sum(axis=1)
    else:
        scores, end = folded_log_bf(model, raw, scenario.noise_var)
        top_probability = float(np.exp(np.max(scores) - logsumexp(scores)))
    pattern = int(np.argmax(scores))
    ranking = time.perf_counter() - tick
    tick = time.perf_counter()
    llr, _ = cavity_query(scenario, cancellation, raw, pattern)
    code = PolarCode(model.b, model.code.n, model.code.crc_bits, 4096)
    if type(model.code) is not PolarCode or not np.array_equal(code.information, model.code.information) or not np.array_equal(code.frozen, model.code.frozen):
        raise ValueError("Terminal list provider requires the original polar encoder and matching layout")
    payloads, metrics = code.decode_list(llr)
    found = [model.word(pattern, payload) for payload in payloads]
    search = time.perf_counter() - tick
    tick = time.perf_counter()
    fresh, _ = bank.extend(found)
    action, fit = bank.refit(cancellation) if fresh else (Counter(cancellation), None)
    output, completion = support_completion(scenario, action, bank.words)
    return output, dict(pattern=pattern, signed_prefix_end=end, model_top_probability=top_probability,
        query_ids=found, list_metrics=metrics.tolist(), new_ids=fresh, cancellation=sorted(action.items()),
        fit=fit, completion=completion, setup_seconds=setup, ranking_seconds=ranking, search_seconds=search,
        refit_completion_seconds=time.perf_counter()-tick, seconds=time.perf_counter()-started)
