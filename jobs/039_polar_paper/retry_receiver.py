"""Single-block CRC-legal likelihood transport within the ordinary count-growth receiver."""

from collections import Counter
import copy
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"jobs/037_polar_tractable")]
from baselines.polar import PolarCode, append_crc
from framework.polar_count_cavity import action_hash, cavity_query
from framework.polar_count_feedback import PayloadMemo, support_completion
from framework.polar_full_pattern import CountBank
from framework.polar_repetition_scl import RepetitionPolarCode
from framework.polar_tractable import PolarScenario
from rollback_frames import native
from rollback_progressive import progressive


class CaptureCode(PolarCode):
    """One ordinary information-set SCL pass, exposing the payloads of its final survivors."""

    def __init__(self, original):
        super().__init__(original.payload_bits, original.n, original.crc_bits, original.list_size)
        self.raw = PolarCode(self.k, self.n, 0, self.list_size)
        assert np.array_equal(self.raw.information, self.information)
        self.capture = True
        self.packed = None
        self.extra_seconds = 0.
        self.captured = 0

    def decode_list(self, llr):
        if not self.capture:
            return super().decode_list(llr)
        full, metrics = self.raw.decode_list(llr)
        tick = time.perf_counter()
        payload = full[:, :self.payload_bits]
        valid = np.all(append_crc(payload, self.crc_bits) == full, axis=1)
        self.packed = np.packbits(payload, axis=1)
        self.extra_seconds += time.perf_counter()-tick
        self.captured += 1
        return payload[valid], metrics[valid]


def priority(code, packed, amp, old, current, variance):
    """Integrate only over distinct old legal payloads. Use a sparse numerator displacement."""
    payloads = np.unpackbits(np.unique(packed, axis=0), axis=1, count=code.payload_bits)
    signs = 1-2*code.encode(payloads).astype(np.int8)
    f = amp*old/variance
    delta = amp*(current-old)/variance
    nz = np.flatnonzero(delta)
    previous = np.einsum("ij,j->i", signs, f, optimize=False)
    update = np.einsum("ij,j->i", signs[:, nz], delta[nz], optimize=False)
    anchor = (np.logaddexp(f,-f)-np.log(2.)-amp**2/(2*variance)).sum()
    return float(anchor + logsumexp(previous+update)-logsumexp(previous))


def requeue(original, known, unchanged, ranking):
    paid = [p for p in original if p not in unchanged]
    if len(paid) < 8:
        return None
    quota = sum(p in known for p in paid[:8])
    available = set(known)-unchanged
    if not quota or len(available) <= quota:
        return None
    if ranking is None:
        return quota
    if set(ranking) != available or len(ranking) != len(available):
        raise ValueError("Require each changed earlier pattern once")
    selected = ranking[:quota]
    rest = [p for p in original if p in available and p not in selected]
    pool = iter(selected+rest)
    revised = [next(pool) if p in available else p for p in original]
    assert len(set(revised)) == len(revised) == len(original)
    assert all((a in known,a in unchanged) == (b in known,b in unchanged) for a,b in zip(original,revised))
    return revised


def choose(s, raw, action, queue, history, memo, pool, code):
    var = s.noise_var+max(0, s.k-action.total()-1)/s.model.n
    known = set(history)
    unchanged = set()
    for p in known:
        amp, pos = s.model.amplitudes[p], s.model.positions[p]
        key = (2*amp*raw[pos]/var).tobytes()
        if key in memo.cache:
            unchanged.add(p)
    quota = requeue(queue, known, unchanged, None)
    if quota is None:
        return None
    old = {}
    for p in known-unchanged:
        key, r0 = history[p]
        pos, amp = s.model.positions[p], s.model.amplitudes[p]
        old[p] = priority(code, pool[key][0], amp, r0, raw[pos], var)
    revised = requeue(queue, known, unchanged, sorted(old, key=lambda p:(-old[p],p)))
    return revised, dict(quota=quota, original=queue, revised=revised, changed=revised!=queue, scored=len(old),
        paid_before=sum(p not in unchanged for p in queue), prior_changed_slots=sum(p in old for p in queue),
        selected_exploration=[p for p in queue if p not in known])


def continuation(s, initial, initial_bank, policy):
    bank = CountBank(s, initial_bank)
    action, fitted = bank.refit(initial)
    assert fitted["converged"]
    code = CaptureCode(PolarCode(s.model.b, s.model.code.n, s.model.code.crc_bits, 128))
    code.capture = policy
    memo = PayloadMemo(code)
    history, pools, trace, sweeps, selection = {}, {}, [], [], None
    peak = capture_updates = 0
    packed_bytes = local_bytes = duplicate_key_bytes = 0
    capture_ops = selection_seconds = 0.
    for sweep in range(s.k+4):
        if sweep == 3 and not sweeps[-1]["moves"]:
            stop = "third_sweep_no_count_moves"
            break
        old_total, old_size = action.total(), len(bank.words)
        magnitude = np.abs(bank.last_raw_residual[s.model.positions]).sum(axis=1)
        queue = np.argsort(-magnitude, kind="stable")[:s.k+5].tolist()
        if policy and selection is None and sweep and s.k > old_total:
            t0 = time.perf_counter()
            picked = choose(s, bank.last_raw_residual, action, queue, history, memo, pools, code)
            selection_seconds += time.perf_counter()-t0
            if picked is not None:
                queue, selection = picked
                selection.update(sweep=sweep, at_query=len(trace), h=s.k-old_total)
                history.clear()
                pools.clear()
                code.capture = False
                code.packed = None
        moves = 0
        for p in queue:
            before = action_hash(action)
            llr, _ = cavity_query(s, action, bank.last_raw_residual, p)
            payloads, _ = memo.decode_list(llr)
            if code.capture:
                t0 = time.perf_counter()
                key = llr.tobytes()
                if not memo.last_hit:
                    pools[key] = (code.packed, key)
                    packed_bytes += code.packed.nbytes
                    duplicate_key_bytes += len(key)
                if p not in history:
                    local_bytes += llr.nbytes
                history[p] = (pools[key][1],bank.last_raw_residual[s.model.positions[p]].copy())
                peak = max(peak,packed_bytes+local_bytes+duplicate_key_bytes)
                capture_ops += time.perf_counter()-t0
                capture_updates += 1
            words = [s.model.word(p,v) for v in payloads]
            fresh, _ = bank.extend(words)
            trace.append([p,before,hashlib.sha256(llr.tobytes()).hexdigest(),words,memo.last_hit])
            if fresh:
                action, fit = bank.refit(action)
                assert fit["converged"]
                moves += len(fit["moves"])
        sweeps.append(dict(moves=moves, end_count=action.total(), bank_size=len(bank.words)))
        if len(bank.words) == old_size:
            stop = "full_sweep_no_new_ids"
            break
        if sweep >= 3 and action.total() <= old_total:
            stop = "no_endpoint_count_growth"
            break
    else:
        raise AssertionError("Growth bound exhausted")
    return action, set(bank.words), dict(trace=trace,sweeps=sweeps,stop=stop,actual_scl_calls=memo.calls,
        cache_hits=memo.hits,selection=selection,selection_seconds=selection_seconds,capture_ops_seconds=capture_ops,
        capture_readout_seconds=code.extra_seconds,captured_queries=code.captured,history_updates=capture_updates,
        extra_stored_data_bytes=peak, packed_data_bytes=packed_bytes, local_data_bytes=local_bytes, duplicate_key_bytes=duplicate_key_bytes)


def run(scenario, *, policy=True):
    tick = time.perf_counter()
    base = scenario.model.code
    if type(base) is not PolarCode or base.list_size != 128 or scenario.sampling != "iid":
        raise ValueError("Require the original IID SCL128 scenario")
    m = copy.copy(scenario.model)
    m.code = RepetitionPolarCode(base.payload_bits, base.n, base.crc_bits, base.list_size)
    s = PolarScenario(m,scenario.y,scenario.k,scenario.noise_var,scenario.sampling)
    accepted, native_meta = native(s)
    initial, fresh, batches, rollback_seconds = progressive(s,list(accepted),native_meta["valid_ids"],"ungated")
    initial_bank = set(native_meta["valid_ids"]).union(initial,fresh)
    action, bank, details = continuation(s,initial,initial_bank,policy)
    output, completion = support_completion(s,action,bank)
    return dict(output), dict(cancellation_action=sorted(action.items()), bank=sorted(bank), rollback_action=sorted(initial.items()),
        base_bank=sorted(initial_bank), native=native_meta, rollback_seconds=rollback_seconds, rollback_batches=batches,
        continuation=details,support_completion=completion,complete=True,total_seconds=time.perf_counter()-tick)
