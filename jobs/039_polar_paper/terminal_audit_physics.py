"""Independent CRC, physical count objective, and local-optimum checks used by the terminal audit.

The calculations are the verified independent Job 037 development-audit helpers, distributed here
without their archived study paths. Source placement geometry retains the established generator.
"""

from collections import Counter
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'jobs/037_polar_tractable'))
from rollback_frames import geometry

COUNTS, ERRORS = Counter(), Counter()


def close(left, right, name, tolerance=2e-7):
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape, (name, left.shape, right.shape)
    value = float(np.max(np.abs(left - right))) if left.size else 0.
    assert np.isfinite(value) and value <= tolerance, (name, value, tolerance)
    ERRORS[name] = max(ERRORS[name], value)


def payloads(ids):
    raw = b"".join((int(word) & ((1 << 87) - 1)).to_bytes(11, "big") for word in ids)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 11), axis=1)[:, 1:]


def encode(source, information):
    """Independent bit-register CRC16 followed by explicitly indexed polar butterflies."""
    source = np.asarray(source, dtype=np.uint8)
    register = np.zeros(len(source), dtype=np.uint32)
    for bit in source.T:
        feedback = (register >> 15) ^ bit
        register = ((register << 1) & 65535) ^ (feedback * 0x1021)
    crc = ((register[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.uint8)
    output = np.zeros((len(source), 512), dtype=np.uint8)
    output[:, information] = np.concatenate((source, crc), axis=1)
    for bit in range(9):
        left = np.array([j for j in range(512) if not j & (1 << bit)])
        output[:, left] ^= output[:, left | (1 << bit)]
    return output


def regenerate(seed):
    model, balance = geometry(seed)
    rng = np.random.default_rng([671129, seed, 0])
    messages = rng.integers(2, size=(100, 100), dtype=np.uint8)
    labels = [int.from_bytes(np.packbits(row).tobytes(), "big") >> 4 for row in messages]
    original_noise = rng.normal(0., np.sqrt(1 / (200 * 10 ** .05)), 30000)
    codewords = encode(messages[:, 13:], model.code.information)
    values = model.amplitudes[[word >> 87 for word in labels]] * (1.0 - 2.0 * codewords)
    y = original_noise.copy()
    for word, signal in zip(labels, values):
        y[model.positions[word >> 87]] += signal
    extracted_noise = y.copy()
    for word, signal in zip(labels, values):
        extracted_noise[model.positions[word >> 87]] -= signal
    close(extracted_noise, original_noise, "noise_extraction", 2e-15)
    COUNTS["independently_regenerated_frames"] += 1
    return model, labels, y, extracted_noise, values


class PhysicalBank:
    """Independent CRC/polar encoding, physical sums and direct neighboring objective values."""

    def __init__(self, model, y, words):
        self.model, self.y, self.words = model, y, sorted(words)
        self.index = {w: j for j, w in enumerate(self.words)}
        self.codewords = encode(payloads(self.words), model.code.information)
        self.patterns = np.array([w >> 87 for w in self.words], dtype=int)
        self.values = model.amplitudes[self.patterns] * (1. - 2. * self.codewords)
        self.positions = model.positions[self.patterns]
        self.matrix = csc_matrix((self.values.ravel(), self.positions.ravel(), np.arange(len(words)+1)*512),
                                 shape=(30000, len(words)))
        self.gram = (self.matrix.T @ self.matrix).toarray()
        self.norms = np.diag(self.gram)
        close(self.norms, np.ones(len(words)), 'unit_atom_energies', 2e-10)
        generator = encode(np.eye(87, dtype=np.uint8), model.code.information)
        constant = np.flatnonzero(np.all(generator == 0, axis=0))
        mean = np.zeros(30000)
        for pattern in range(8192):
            mean[model.positions[pattern, constant]] += model.amplitudes[pattern, constant] / 8192
        close(mean, model.mean, 'independent_source_mean', 1e-15)
        self.source_mean = mean
        self.variance = 1 / (200 * 10 ** .05)
        COUNTS['complete_bank_words_independently_encoded'] += len(words)

    def vector(self, action):
        assert set(action) <= set(self.words) and action.total() <= 100
        return np.array([action[w] for w in self.words], dtype=np.int64)

    def raw(self, action):
        return self.y - self.matrix @ self.vector(action)

    def outside(self, bank):
        indices = np.array([self.index[w] for w in sorted(bank)], dtype=int)
        outside = (1 << 100) - len(bank)
        total = np.asarray(self.matrix[:, indices].sum(axis=1)).ravel()
        mean = self.source_mean + (len(bank)*self.source_mean-total)/outside
        return indices, mean, (1-float(mean @ mean))/30000, math.log(outside)

    def objective(self, action, raw, mean, kappa, log_outside):
        h = 100-action.total()
        variance = self.variance+h*kappa
        residual = raw-h*mean
        prior = h*log_outside-math.lgamma(h+1)-sum(math.lgamma(a+1) for a in action.values())
        return -.5*30000*math.log(variance)-float(residual @ residual)/(2*variance)+prior

    def best_gain(self, action, bank, raw, mean, kappa, log_outside):
        """Evaluate direct candidate F values using independently encoded quadratic norms."""
        indices = np.array([self.index[w] for w in sorted(bank)], dtype=int)
        counts = np.array([action[w] for w in sorted(bank)])
        h = 100-action.total()
        residual = raw-h*mean
        square, norm_mean = float(residual @ residual), float(mean @ mean)
        dots = np.asarray(self.matrix.T @ residual).ravel()[indices]
        mean_dots = np.asarray(self.matrix.T @ mean).ravel()[indices]
        norms = self.norms[indices]
        centered_dot, centered_norm = dots-float(residual @ mean), norms-2*mean_dots+norm_mean
        prior = h*log_outside-math.lgamma(h+1)-sum(math.lgamma(int(a)+1) for a in counts)
        current = self.objective(action, raw, mean, kappa, log_outside)
        best = 0.
        if h and len(indices):
            variance = self.variance+(h-1)*kappa
            new_square = square-2*centered_dot+centered_norm
            new_prior = prior+math.log(h)-np.log(counts+1)-log_outside
            values = -.5*30000*math.log(variance)-new_square/(2*variance)+new_prior-current
            best = max(best, float(max(values)))
        active = np.flatnonzero(counts)
        if len(active):
            variance = self.variance+(h+1)*kappa
            new_square = square+2*centered_dot[active]+centered_norm[active]
            new_prior = prior+np.log(counts[active])+log_outside-math.log(h+1)
            values = -.5*30000*math.log(variance)-new_square/(2*variance)+new_prior-current
            best = max(best, float(max(values)))
        variance = self.variance+h*kappa
        for old in active:
            difference_norm = norms+norms[old]-2*self.gram[indices[old], indices]
            new_square = square-2*(dots-dots[old])+difference_norm
            new_prior = prior+math.log(int(counts[old]))-np.log(counts+1)
            values = -.5*30000*math.log(variance)-new_square/(2*variance)+new_prior-current
            values[old] = -np.inf
            best = max(best, float(max(values)))
        COUNTS['independent_local_maximum_checks'] += 1
        return best

    def refit(self, action, bank, saved):
        action = action.copy()
        _, mean, kappa, log_outside = self.outside(bank)
        raw = self.raw(action)
        value = self.objective(action, raw, mean, kappa, log_outside)
        close(value, saved['initial_objective'], 'initial_refit_objective')
        close(float(np.linalg.norm(mean)), saved['outside_mean_norm'], 'outside_mean_norm', 1e-15)
        close(kappa, saved['outside_trace_variance_per_coordinate'], 'outside_trace', 1e-15)
        assert saved['bank_size'] == len(bank) and saved['max_moves'] == 2000
        evaluated, gram_rows, max_cached, cache = 0, 0, 0, set()
        for j in range(len(saved['moves'])+1):
            active = set(action)
            evaluated += (len(bank) if action.total()<100 else 0) + len(active)*len(bank)
            cache.intersection_update(active)
            gram_rows += len(active-cache)
            cache.update(active)
            max_cached = max(max_cached, len(cache))
            best = self.best_gain(action, bank, raw, mean, kappa, log_outside)
            if j == len(saved['moves']):
                close(best, saved['next_positive_gain'], 'terminal_best_gain')
                assert saved['converged'] == (saved['next_positive_gain'] <= 1e-8)
                assert saved['move_cap_hit'] == (not saved['converged'])
                if not saved['converged']:
                    assert len(saved['moves']) == 2000
                break
            move = saved['moves'][j]
            close(best, move['gain'], 'greedy_maximum_gain')
            assert move['gain'] > 1e-8
            old, new, kind = move['old'], move['new'], move['kind']
            assert kind in ('add','drop','swap')
            assert (old is None) == (kind=='add') and (new is None) == (kind=='drop')
            if old is not None:
                assert action[old] > 0 and old in bank
                action[old] -= 1
                if action[old] == 0: del action[old]
            if new is not None:
                assert new in bank and new != old
                action[new] += 1
            assert action.total() <= 100 and 100-action.total() == move['outside_count']
            raw = self.raw(action)
            next_value = self.objective(action, raw, mean, kappa, log_outside)
            close(next_value-value, move['gain'], 'direct_physical_objective_increment')
            assert next_value >= value-2e-7
            value = next_value
            COUNTS['recorded_count_moves_checked'] += 1
        close(value, saved['final_objective'], 'final_refit_objective')
        assert saved['outside_count'] == 100-action.total()
        assert saved['candidate_gain_evaluations'] == evaluated
        assert saved['gram_rows_constructed'] == gram_rows and saved['largest_cached_gram_rows'] == max_cached
        COUNTS['count_refits_checked'] += 1
        return action


def action_from_pairs(values):
    assert values == sorted(values) and len(values) == len(dict(values))
    assert all(type(w) is int and 0 <= w < 2**100 and type(c) is int and c > 0 for w,c in values)
    action = Counter(dict(values))
    assert action.total() <= 100
    return action


def measurements(action, bank, labels):
    truth = Counter(labels)
    matched = (action & truth).total()
    absent = sorted(set(truth)-set(bank))
    return dict(matched_copies=matched, misses=100-matched, count_pupe=(100-matched)/100,
        false_copies=action.total()-matched, output_copies=action.total(), bank_size=len(bank),
        bank_absent_true_ids=absent, multiplicity_capable_oracle_bank_miss_floor=sum(truth[w] for w in absent))


def audit_completion(row,physical,action,bank):
    done=action_from_pairs(row['completed_action']);meta=row['completion_meta']
    assert all(done[w]>=c for w,c in action.items()) and done.total()<=100
    additions=done-action;absent=sorted(bank-set(action));remaining=100-action.total()
    assert set(additions)<=set(absent) and all(c==1 for c in additions.values())
    assert meta['additions']==sorted([w,c] for w,c in additions.items())
    assert meta['input_count']==action.total() and meta['output_count']==done.total()
    assert meta['bank_size']==len(bank) and meta['absent_bank_ids']==len(absent)
    method='no_capacity' if not remaining else 'all_fit'
    if len(absent)<=remaining:
        assert done==action+Counter(absent) and not meta['steps']
        assert meta['bank_construction_seconds']==meta['ranking_seconds']==0
    elif remaining:
        method='count_add_gain';current=action.copy()
        _,mean,kappa,log_outside=physical.outside(bank)
        eligible=set(absent)
        for step in meta['steps']:
            raw=physical.raw(current);before=physical.objective(current,raw,mean,kappa,log_outside)
            gains={}
            for word in sorted(eligible):
                proposed=current+Counter({word:1})
                gains[word]=physical.objective(proposed,physical.raw(proposed),mean,kappa,log_outside)-before
            chosen=step['word'];assert chosen in eligible
            close(gains[chosen],max(gains.values()),'completion_greedy_maximum',2e-7)
            close(gains[chosen],step['gain'],'completion_direct_F_increment',2e-7)
            current[chosen]+=1;eligible.remove(chosen)
            assert step['remaining_slots']==100-current.total()
            COUNTS['completion_ranked_additions_verified']+=1
        assert current==done and current.total()==100
    else:
        assert done==action and not meta['steps']
    assert meta['method']==method
    assert meta['seconds']<=row['completion_wall_seconds']
    assert meta['bank_construction_seconds']+meta['ranking_seconds']<=meta['seconds']
    assert row['completion_preserves_counts_and_cap']
    truth=Counter(row['labels'])
    gain=(truth&done).total()-(truth&action).total()
    assert gain>=0 and gain==sum(truth[w]>0 for w in additions)
    assert (done.total()-(truth&done).total())-(action.total()-(truth&action).total())==additions.total()-gain
    COUNTS['terminal_completions_verified']+=1
    return done,dict(method=method,added_copies=additions.total(),saved_copies=gain,added_false_copies=additions.total()-gain)
