"""Tag-conditioned placement ranking, serial polar SIC and two raw terminal repairs."""
import hashlib
import math
import time

import numpy as np
from scipy.special import expit

from framework.polar_serial_repair import SerialMoments, prefix as serial_prefix
from framework.polar_tractable import Work
from framework.polar_two_raw import finish


def _occupancy(scores, remaining):
    placements = len(scores)
    target = -placements * np.expm1(remaining * np.log1p(-1 / placements))
    weights = np.empty_like(scores)
    for group, column in enumerate(scores.T):
        low, high = -np.max(column) - 60., -np.min(column) + 60.
        for _ in range(50):
            offset = (low + high) * .5
            if expit(column + offset).sum() < target:
                low = offset
            else:
                high = offset
        weights[:, group] = expit(column + (low + high) * .5)
        if abs(weights[:, group].sum() - target) > 1e-7:
            raise ArithmeticError("Tagged placement occupancy did not reach the IID distinct-count mean")
    return weights


def update(moments):
    """Rank every placement on the tagged chip channel, then refresh held-out nuisance powers once."""
    ranks = np.empty(moments.P)
    scores = np.empty((moments.P, 8))
    for start in range(0, moments.P, 128):
        end = min(start + 128, moments.P)
        at, amplitude = moments.pos[start:end], moments.amp[start:end]
        group = at % 8
        weight = moments.pi[np.arange(start, end)[:, None], group]
        other = (1 - 1 / moments.P) / (moments.sums[group] - weight)
        own = 1 / moments.P - other * weight
        mean = other * moments.mean[at] + own * amplitude * moments.constant
        second = other * moments.energy[at] + own * amplitude**2
        variance = moments.noise + (moments.h - 1) * (second - mean**2)
        if np.any(variance < moments.noise - 1e-13):
            raise ArithmeticError("Invalid tagged nuisance variance in placement update")
        field = amplitude * (moments.raw[at] - (moments.h - 1) * mean) / variance
        terms = np.logaddexp(field, -field) - math.log(2.)
        terms[:, moments.constant] = field[:, moments.constant]
        terms -= amplitude**2 / (2 * variance)
        full = terms.sum(axis=1)
        ranks[start:end] = full
        for g in range(8):
            scores[start:end, g] = full - np.sum(terms, axis=1, where=group == g)

    moments.pi = _occupancy(scores, moments.h)
    moments.sums = moments.pi.sum(axis=0)
    moments.energy.fill(0.)
    moments.mean.fill(0.)
    for start in range(0, moments.P, 128):
        end = min(start + 128, moments.P)
        at, amplitude = moments.pos[start:end], moments.amp[start:end]
        w = moments.pi[np.arange(start, end)[:, None], at % 8]
        np.add.at(moments.energy, at.ravel(), (w * amplitude * amplitude).ravel())
        fixed = moments.constant
        np.add.at(moments.mean, at[:, fixed].ravel(), (w[:, fixed] * amplitude[:, fixed]).ravel())
    return ranks


def native(scenario):
    started = time.perf_counter()
    model, residual = scenario.model, scenario.y.copy()
    constant = np.array(model.keys, dtype=object) == 0
    work, accepted, discovered, rounds = Work(model), {}, set(), []
    calls = high_rounds = 0
    for step in range(30):
        h = scenario.k - len(accepted)
        if h == 0:
            break
        density = model.code.n / model.n
        high = h > 1 and (density == 1 or -(h - 1) * math.log1p(-density) >= math.log(2.))
        if high:
            high_rounds += 1
            moments = SerialMoments(model.positions, model.amplitudes, constant, model.n, residual, scenario.noise_var, h)
            queue = np.argsort(-update(moments), kind="stable")[:h + 5]
        else:
            queue = work.patterns(residual, h + 5)
        current = []
        for pattern in queue:
            at = model.positions[pattern]
            llr = moments.llr(pattern) if high else 2 * model.amplitudes[pattern] * residual[at] / (scenario.noise_var + (h - 1) / model.n)
            payloads, _ = model.code.decode_list(llr)
            calls += 1
            words = [model.word(pattern, u) for u in payloads]
            discovered.update(words)
            if words and words[0] not in accepted:
                word = words[0]
                accepted[word] = step
                current.append(word)
                if high:
                    _, values = model.symbols(word)
                    residual[at] -= values
                    moments.identify(int(pattern))
                if len(accepted) == scenario.k:
                    break
        rounds.append(len(current))
        if not current:
            break
        if not high:
            for word in current:
                pattern, values = model.symbols(word)
                residual[model.positions[pattern]] -= values
    return accepted, dict(calls=calls, rounds=rounds, held_rounds=high_rounds, seconds=time.perf_counter() - started,
                          valid_ids=sorted(discovered), residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest())


def prefix(scenario):
    """Return the observation-driven native, rollback and complete count-growth state."""
    output, meta = serial_prefix(scenario, provider=native)
    return output, {**meta, "native_demapper": "conditional_serial"}


def run(scenario):
    started = time.perf_counter()
    output, meta = prefix(scenario)
    output, meta = finish(scenario, output, meta)
    return output, {**meta, "total_seconds": time.perf_counter() - started}
