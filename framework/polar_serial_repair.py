"""High-load successive cancellation with conditional ODMA chip moments and fixed terminal repair."""
import copy
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_feedback import continue_feedback, support_completion
from framework.polar_held_interference import HeldResourceMoments, prefix as held_prefix
from framework.polar_rate_challenge import finish as challenge
from framework.polar_signed_restoration import finish as restore
from framework.polar_repetition_scl import RepetitionPolarCode
from framework.polar_terminal_scan import finish as scan
from framework.polar_tractable import PolarScenario, Work


class SerialMoments(HeldResourceMoments):
    """Remove the fitted placement's soft weight while retaining its IID repeat probability."""

    def __init__(self, *args):
        super().__init__(*args)
        self.identified = set()
        self.known_mean = np.zeros_like(self.mean)
        self.known_energy = np.zeros_like(self.energy)

    def identify(self, pattern):
        self.h -= 1
        if pattern in self.identified:
            return
        at, a = self.pos[pattern], self.amp[pattern]
        weight = self.pi[pattern, at % 8]
        self.energy[at] -= weight * a**2
        self.mean[at] -= weight * a * self.constant
        self.known_energy[at] += a**2
        self.known_mean[at] += a * self.constant
        self.sums -= self.pi[pattern]
        self.identified.add(pattern)

    def llr(self, pattern):
        if not self.identified:
            return super().llr(pattern)
        at, a = self.pos[pattern], self.amp[pattern]
        if self.h == 1:
            return 2 * a * self.raw[at] / self.noise
        group = at % 8
        extra = int(pattern not in self.identified)
        weight = extra * self.pi[pattern, group]
        restricted = len(self.identified) + extra
        other = np.zeros(len(at)) if restricted == self.P else (1 - restricted/self.P) / (self.sums[group] - weight)
        mean = (self.known_mean[at] + extra*a*self.constant)/self.P + other*(self.mean[at] - weight*a*self.constant)
        second = (self.known_energy[at] + extra*a**2)/self.P + other*(self.energy[at] - weight*a**2)
        variance = self.noise + (self.h - 1)*(second - mean**2)
        if np.any(variance < self.noise - 1e-13):
            raise ArithmeticError("Invalid conditional nuisance variance")
        return 2*a*(self.raw[at] - (self.h - 1)*mean)/variance


def native(scenario):
    started = time.perf_counter()
    model, residual = scenario.model, scenario.y.copy()
    constant = np.array(model.keys, dtype=object) == 0
    work, accepted, ids, rounds = Work(model), {}, set(), []
    calls = high_rounds = 0
    for step in range(30):
        h = scenario.k - len(accepted)
        if h == 0:
            break
        density = model.code.n / model.n
        high = h > 1 and (density == 1 or -(h-1)*math.log1p(-density) >= math.log(2.))
        high_rounds += int(high)
        moments = SerialMoments(model.positions, model.amplitudes, constant, model.n, residual, scenario.noise_var, h) if high else None
        current = []
        for p in work.patterns(residual, h + 5):
            at = model.positions[p]
            llr = moments.llr(p) if high else 2*model.amplitudes[p]*residual[at]/(scenario.noise_var + (h-1)/model.n)
            payloads, _ = model.code.decode_list(llr)
            calls += 1
            returned = [model.word(p, u) for u in payloads]
            ids.update(returned)
            if returned and returned[0] not in accepted:
                w = returned[0]
                accepted[w] = step
                current.append(w)
                if high:
                    _, values = model.symbols(w)
                    residual[at] -= values
                    moments.identify(int(p))
                if len(accepted) == scenario.k:
                    break
        rounds.append(len(current))
        if not current:
            break
        if not high:
            for w in current:
                p, values = model.symbols(w)
                residual[model.positions[p]] -= values
    return accepted, dict(calls=calls, rounds=rounds, held_rounds=high_rounds, seconds=time.perf_counter()-started,
                          valid_ids=sorted(ids), residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest())


def prefix(scenario, provider=native):
    started = time.perf_counter()
    if not isinstance(scenario, PolarScenario) or type(scenario.model.code) is not PolarCode or scenario.sampling != "iid":
        raise ValueError("Require the original procedural polar encoder and IID messages")
    code = scenario.model.code
    if code.list_size != 128 or scenario.model.p < 32:
        raise ValueError("Require native width 128 and at least 32 placements for rollback")
    model = copy.copy(scenario.model)
    model.code = RepetitionPolarCode(code.payload_bits, code.n, code.crc_bits, 128)
    if not np.array_equal(code.information, model.code.information) or not np.array_equal(code.frozen, model.code.frozen):
        raise ValueError("Incompatible native polar layout")
    s = PolarScenario(model, scenario.y, scenario.k, scenario.noise_var, scenario.sampling)
    legacy = str(Path(__file__).resolve().parents[1] / "jobs/037_polar_tractable")
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    from rollback_progressive import progressive
    accepted, first = provider(s)
    initial, fresh, batches, seconds = progressive(s, list(accepted), first["valid_ids"], "ungated")
    base_bank = set(first["valid_ids"]).union(fresh, initial)
    fitted, bank, trace = continue_feedback(s, initial, base_bank, memoize=True, progress_extra_sweep=True, count_growth=True)
    output, completion = support_completion(s, fitted, bank)
    meta = dict(native=first, native_accepted=list(accepted), rollback_seconds=seconds, rollback_batches=batches,
                rollback_action=sorted(initial.items()), base_bank=sorted(base_bank), bank=sorted(bank),
                cancellation_action=sorted(fitted.items()), continuation=trace, support_completion=completion,
                complete_support=True, complete=trace["complete"], total_seconds=time.perf_counter()-started)
    return dict(output), meta


def finish(scenario, output, meta, *, always=False):
    output, meta = scan(scenario, output, meta)
    return (restore if always else challenge)(scenario, output, meta)


def run(scenario, *, always=False):
    started = time.perf_counter()
    counts, meta = prefix(scenario)
    counts, meta = finish(scenario, counts, meta, always=always)
    return counts, {**meta, "native_demapper": "serial_held_resource", "total_seconds": time.perf_counter()-started}


def run_held(scenario, *, always=False):
    started = time.perf_counter()
    counts, meta = held_prefix(scenario)
    counts, meta = finish(scenario, counts, meta, always=always)
    return counts, {**meta, "native_demapper": "held_resource", "total_seconds": time.perf_counter()-started}
