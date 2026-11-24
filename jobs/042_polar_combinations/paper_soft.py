"""Fixed-placement scalar transfer of Li et al. (2026), Eqs. (25)–(29) and Algorithm 2 lines 16–24."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from paper_native import SEEDS, paper_curve, rollback_frames
from baselines.polar import append_crc, polar_transform
from framework.polar_fast_scl import RateZeroPolarCode
from framework.polar_tractable import Work


class SoftSCL(RateZeroPolarCode):
    """Ordinary static-frozen SCL and the Yuan–Duffy–Médard approximate mass of its disjoint pruned subtrees."""

    def __init__(self, payload_bits, n, crc_bits, list_size):
        super().__init__(payload_bits, n, crc_bits, list_size)
        self.frozen_after = np.cumsum(self.frozen[::-1])[::-1] - self.frozen

    def decode_means(self, llr, *, crc_flag=False, terminal=None):
        llr = np.asarray(llr, dtype=float)
        if llr.shape != (self.n,) or not np.all(np.isfinite(llr)):
            raise ValueError("Expected one finite LLR per coded chip")
        if terminal not in (None,"bound","resolve") or (terminal and (crc_flag or self.frozen[-1])):
            raise ValueError("Terminal-sibling audit requires a free last input and a soft sign query")
        pruned_logs = []
        terminal_log = -np.inf

        def visit(alpha, metrics, start):
            nonlocal terminal_log
            paths, width = alpha.shape
            if self._rate_zero[(self.n + start) // width]:
                beta = np.zeros((paths, width), dtype=np.uint8)
                return beta, metrics + np.logaddexp(0., -alpha).sum(axis=1), np.arange(paths)
            if width == 1:
                both = (metrics[:, None] + np.logaddexp(0., -alpha * np.array([1., -1.]))).ravel()
                order = np.argsort(both, kind="stable")
                discarded = order[self.list_size:]
                last = terminal is not None and start==self.n-1
                if discarded.size:
                    suffix = self.frozen_after[start] * np.log(2.)
                    lost = np.logaddexp.reduce(-both[discarded]) - suffix
                    if last:
                        terminal_log = lost
                    if not (last and terminal=="resolve"):
                        pruned_logs.append(lost)
                keep = order if last and terminal=="resolve" else order[:self.list_size]
                return (keep % 2).astype(np.uint8)[:, None], both[keep], keep // 2
            half = width // 2
            a, b = alpha[:, :half], alpha[:, half:]
            alpha_left = np.logaddexp(0., a + b) - np.logaddexp(a, b)
            left, left_metrics, parents = visit(alpha_left, metrics, start)
            alpha_right = b[parents] + (1. - 2. * left) * a[parents]
            right, last_metrics, origins = visit(alpha_right, left_metrics, start + half)
            return np.concatenate((left[origins] ^ right, right), axis=1), last_metrics, parents[origins]

        codewords, metrics, _ = visit(llr[None, :], np.zeros(1), 0)
        seen_log = np.logaddexp.reduce(-metrics)
        lost_log = np.logaddexp.reduce(pruned_logs) if pruned_logs else -np.inf
        total_log = np.logaddexp(seen_log, lost_log)
        lost_share = np.exp(lost_log - total_log)
        # Eq. (26) in BPSK sign coordinates: E[x] = P(c=0) - P(c=1).
        mean = np.exp(-metrics - total_log) @ (1. - 2. * codewords) + lost_share * np.tanh(llr / 2.)
        if not np.all(np.isfinite(mean)) or np.any(abs(mean) > 1. + 1e-11):
            raise FloatingPointError("Invalid posterior sign means")
        result = np.clip(mean, -1., 1.), float(total_log), float(lost_share)
        if terminal:
            return (*result,float(np.exp(terminal_log-total_log)))
        if crc_flag:
            data = polar_transform(codewords)[:,self.information]
            expected = append_crc(data[:,:self.payload_bits],self.crc_bits)
            return (*result, bool(np.any(np.all(data == expected,axis=1))))
        return result


def cavity_llrs(residual, positions, amplitudes, means, noise_var):
    signed = amplitudes * means
    uncertainty = amplitudes**2 * (1. - means**2)
    aggregate = np.bincount(positions.ravel(), weights=signed.ravel(), minlength=len(residual))
    variance = np.bincount(positions.ravel(), weights=uncertainty.ravel(), minlength=len(residual))
    other_var = noise_var + variance[positions] - uncertainty
    if np.any(other_var <= 0.):
        raise FloatingPointError("Nonpositive real cavity variance")
    return 2. * amplitudes * (residual[positions] - aggregate[positions] + signed) / other_var


def hard_action(model, positions, means, native):
    polar_inputs = polar_transform((means <= 0.).astype(np.uint8))
    infos = polar_inputs[:, model.code.information]
    payloads = infos[:, :model.b]
    valid = np.all(polar_inputs[:, model.code.frozen] == 0, axis=1)
    valid &= np.all(append_crc(payloads, model.code.crc_bits) == infos, axis=1)
    candidates = [model.word(p, payload) for p, payload in zip(positions[valid], payloads[valid], strict=True)]
    return dict.fromkeys([*native, *candidates], 1), int(np.sum(valid))


def soft_stage(scenario, native):
    start = time.perf_counter()
    m = scenario.model
    deficit = scenario.k - len(native)
    if not 0 <= deficit <= m.p:
        raise ValueError("Fixed distinct candidate placements require 0 <= h <= P")
    if not deficit:
        return dict.fromkeys(native, 1), dict.fromkeys(native, 1), dict(patterns=[], rounds=[], soft_seconds=time.perf_counter()-start)
    residual = scenario.y.copy()
    for word in native:
        p, values = m.symbols(word)
        residual[m.positions[p]] -= values
    pos = Work(m).patterns(residual, deficit)
    positions, amplitude = m.positions[pos], m.amplitudes[pos]
    code = SoftSCL(m.b, m.code.n, m.code.crc_bits, 64)
    if not np.array_equal(code.frozen, m.code.frozen):
        raise ValueError("The scalar transfer must preserve the original static-frozen polar code")
    means = np.zeros_like(amplitude)
    rounds = []
    for step in range(5):
        llrs = cavity_llrs(residual, positions, amplitude, means, scenario.noise_var)
        rows = [code.decode_means(llr) for llr in llrs]
        means = np.stack([row[0] for row in rows])
        action, passed = hard_action(m, pos, means, native)
        rounds.append(dict(scl64_calls=deficit, crc_frozen_pass=passed, listed_share=[1.-v[2] for v in rows],
                           seconds=time.perf_counter()-start))
        if step == 0:
            first = action
        if passed == deficit:
            break
    meta = dict(patterns=pos.tolist(), rounds=rounds, soft_seconds=time.perf_counter()-start)
    return first, action, meta


def run(scenario, *, full_user=False):
    """Decode on inferred scalar placements, with an optional native SIC prefix."""
    native, meta = ({}, None) if full_user else rollback_frames.native(scenario)
    _, action, soft = soft_stage(scenario, native)
    return action, dict(native=meta, soft=soft)


def collect(seed, folder, full_user=False):
    start = time.perf_counter()
    m, _ = rollback_frames.geometry(seed)
    setup = time.perf_counter()-start
    s, labels = rollback_frames.make_frame(m, seed, 0)
    base = json.loads((folder/f"native_{seed}.json").read_text())
    native = [int(word) for word, _ in base["native_order"]]
    if paper_curve.digest(s.y) != base["y_sha256"]:
        raise ValueError("Independently regenerated observation differs")
    if full_user:
        native = []
    first, final, meta = soft_stage(s, native)
    missing = Counter(labels) - Counter(native)
    live_patterns = set(meta["patterns"])
    observed = sum(c for word,c in missing.items() if m.split(word)[0] in live_patterns)
    overlap = 0
    if live_patterns:
        selected_positions = m.positions[list(live_patterns)].ravel()
        occupation = np.bincount(selected_positions, minlength=m.n)
        overlap = int(np.sum(occupation*(occupation-1)//2))
    return dict(seed=seed, y_sha256=base["y_sha256"], native=paper_curve.score(labels, dict.fromkeys(native, 1)),
                first=paper_curve.score(labels, first),
                final=paper_curve.score(labels, final), first_word_additions=sorted(set(first)-set(native)),
                final_word_additions=sorted(set(final)-set(native)), proposed_true_missing_copies=observed,
                proposed_distinct_missing_placements=len(live_patterns.intersection(m.split(w)[0] for w in missing)),
                selected_pair_overlap_resources=overlap, native_s=0. if full_user else base["seconds"],
                geometry_s=setup if full_user else base["geometry_s"],
                regeneration_geometry_s=setup, **meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--full-user", action="store_true", help="Use the fixed first eight per cohort from y with no native support")
    args = parser.parse_args()
    if args.full_user:
        args.seeds = [seed for seed in args.seeds if seed in (*SEEDS[:8], *SEEDS[16:24])]
    for seed in args.seeds:
        prefix = "full_soft" if args.full_user else "soft"
        path = args.out/f"{prefix}_{seed}.json"
        if path.exists():
            continue
        row = collect(seed, args.out, args.full_user)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(row) + "\n")
        temp.replace(path)
        print(f"soft {seed} complete", flush=True)
