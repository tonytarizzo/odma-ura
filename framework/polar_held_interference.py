"""Native polar SIC with cross-fitted ODMA chip weights and DEC-35 count repair."""
import copy
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit

from baselines.polar import PolarCode
from framework.polar_count_feedback import continue_feedback, support_completion
from framework.polar_repetition_scl import RepetitionPolarCode
from framework.polar_terminal_query import finish
from framework.polar_tractable import PolarScenario, Work


class HeldResourceMoments:
    """Unknown-placement activity surrogates exclude the fold of the chip being decoded."""

    def __init__(self, pos, amp, constant, n, raw, noise, h):
        self.pos, self.amp, self.constant = pos, amp, constant
        self.raw, self.noise, self.h = raw, noise, h
        self.P = len(pos)
        if not np.any(~constant) or h < 2 or self.P < 2:
            raise ValueError("Require a multichip code and positive remaining interference")
        variance = noise + (h - 1) / n
        scores = np.empty((self.P, 8))
        for start in range(0, self.P, 128):
            block = slice(start, min(start + 128, self.P))
            group = pos[block] % 8
            field = amp[block] * raw[pos[block]] / variance
            terms = np.logaddexp(field, -field) - np.log(2.)
            terms[:, constant] = field[:, constant]
            terms -= amp[block] ** 2 / (2 * variance)
            full = terms.sum(axis=1)
            for g in range(8):
                scores[block, g] = full - np.sum(terms, axis=1, where=group == g)

        expected = -self.P * np.expm1(h * np.log1p(-1/self.P))
        self.pi = np.empty_like(scores)
        for g in range(8):
            score = scores[:, g].copy()
            lo, hi = -np.max(score) - 60., -np.min(score) + 60.
            for _ in range(50):
                mid = (lo + hi) * .5
                if expit(score + mid).sum() < expected:
                    lo = mid
                else:
                    hi = mid
            self.pi[:, g] = expit(score + (lo + hi) * .5)
            if abs(self.pi[:, g].sum() - expected) > 1e-7:
                raise ArithmeticError("Activity mean mismatch")
        self.sums = self.pi.sum(axis=0)
        self.energy, self.mean = np.zeros(n), np.zeros(n)
        for start in range(0, self.P, 128):
            end = min(start + 128, self.P)
            at, a = pos[start:end], amp[start:end]
            w = self.pi[np.arange(start, end)[:, None], at % 8]
            np.add.at(self.energy, at.ravel(), (w * a ** 2).ravel())
            np.add.at(self.mean, at[:, constant].ravel(), (w[:, constant] * a[:, constant]).ravel())

    def llr(self, p):
        at, amplitude = self.pos[p], self.amp[p]
        group = at % 8
        weight = self.pi[p, group]
        # A second user can choose the tagged placement. Reweight only the other placements.
        other = (1 - 1/self.P) / (self.sums[group] - weight)
        own = 1/self.P - other * weight
        mean = other * self.mean[at] + own * amplitude * self.constant
        second = other * self.energy[at] + own * amplitude * amplitude
        variance = self.noise + (self.h - 1) * (second - mean**2)
        if np.any(variance < self.noise - 1e-13):
            raise ArithmeticError("Invalid tagged chip variance")
        return 2 * amplitude * (self.raw[at] - (self.h - 1) * mean) / variance


def native(s):
    start = time.perf_counter()
    model, residual, accepted = s.model, s.y.copy(), {}
    work, rounds, valid_ids = Work(model), [], set()
    constant = np.array(model.keys, dtype=object) == 0
    calls, held_rounds = 0, 0
    for step in range(30):
        remaining = s.k - len(accepted)
        if remaining == 0:
            break
        density = model.code.n / model.n
        high = remaining > 1 and (density == 1 or -(remaining - 1) * math.log1p(-density) >= math.log(2.))
        moments = HeldResourceMoments(model.positions, model.amplitudes, constant, model.n, residual, s.noise_var, remaining) if high else None
        held_rounds += int(high)
        scalar = s.noise_var + (remaining - 1) / model.n
        current = []
        for p in work.patterns(residual, remaining + 5):
            llr = moments.llr(p) if high else 2 * model.amplitudes[p] * residual[model.positions[p]] / scalar
            payloads, _ = model.code.decode_list(llr)
            calls += 1
            valid = [model.word(p, payload) for payload in payloads]
            valid_ids.update(valid)
            if valid and valid[0] not in accepted:
                accepted[valid[0]] = step
                current.append(valid[0])
                if len(accepted) == s.k:
                    break
        rounds.append(len(current))
        if not current:
            break
        for w in current:
            p, x = model.symbols(w)
            residual[model.positions[p]] -= x
    return accepted, dict(calls=calls, rounds=rounds, held_rounds=held_rounds, seconds=time.perf_counter()-start,
                          valid_ids=sorted(valid_ids), residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest())


def prefix(scenario):
    """Complete the placement-weighted native pass, rollback and count-growth fit."""
    start = time.perf_counter()
    if not isinstance(scenario, PolarScenario) or type(scenario.model.code) is not PolarCode:
        raise ValueError("Require a procedural scenario with the original polar encoder")
    code = scenario.model.code
    if code.list_size != 128 or scenario.model.p < 32:
        raise ValueError("Require native width 128 and at least 32 placements for rollback")
    model = copy.copy(scenario.model)
    model.code = RepetitionPolarCode(code.payload_bits, code.n, code.crc_bits, 128)
    if not np.array_equal(code.information, model.code.information) or not np.array_equal(code.frozen, model.code.frozen):
        raise ValueError("Native repetition and original encoder layouts differ")
    s = PolarScenario(model, scenario.y, scenario.k, scenario.noise_var, scenario.sampling)
    legacy = str(Path(__file__).resolve().parents[1] / "jobs/037_polar_tractable")
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    from rollback_progressive import progressive

    accepted, first = native(s)
    initial, fresh, batches, seconds = progressive(s, list(accepted), first["valid_ids"], "ungated")
    initial_bank = set(first["valid_ids"]).union(fresh, initial)
    counts, bank, trace = continue_feedback(s, initial, initial_bank, memoize=True, progress_extra_sweep=True, count_growth=True)
    output, completion = support_completion(s, counts, bank)
    meta = dict(native=first, native_accepted=list(accepted), rollback_seconds=seconds, rollback_batches=batches,
                rollback_action=sorted(initial.items()), base_bank=sorted(initial_bank), bank=sorted(bank),
                cancellation_action=sorted(counts.items()), continuation=trace, support_completion=completion,
                complete_support=True, complete=trace["complete"])
    meta.update(native_demapper="held_resource", total_seconds=time.perf_counter()-start)
    return dict(output), meta


def run(scenario):
    """Complete balanced-encoder receiver, returning sparse full-message multiplicities."""
    start = time.perf_counter()
    output, meta = prefix(scenario)
    final, meta = finish(scenario, output, meta, "magnitude")
    meta.update(native_demapper="held_resource", total_seconds=time.perf_counter()-start)
    return dict(final), meta
