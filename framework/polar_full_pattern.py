"""Full finite-pattern continuation with an IID count bucket and an explicit Gaussian outside approximation."""

from collections import Counter
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import gammaln

from baselines.polar import PolarCode


class CountBank:
    def __init__(self, scenario, ids):
        self.scenario, self.model = scenario, scenario.model
        self.atoms, self.words = {}, []
        self.construction_seconds = 0.
        self.extend(ids)

    def extend(self, ids):
        started = time.perf_counter()
        ids = list(ids)
        if any(not isinstance(word, (int, np.integer)) for word in ids):
            raise ValueError("Bank IDs must be integers")
        fresh = sorted(set(map(int, ids)).difference(self.atoms))
        for word in fresh:
            pattern, values = self.model.symbols(word)
            self.atoms[word] = (self.model.positions[pattern], values)
        if not fresh and hasattr(self, "matrix"):
            return [], 0.
        self.words = sorted(self.atoms)
        self.index = {word: j for j, word in enumerate(self.words)}
        count, length = len(self.words), self.model.code.n
        if count:
            indices = np.concatenate([self.atoms[word][0] for word in self.words])
            values = np.concatenate([self.atoms[word][1] for word in self.words])
        else:
            indices, values = np.empty(0, dtype=int), np.empty(0)
        self.matrix = csc_matrix((values, indices, np.arange(count + 1) * length), shape=(self.model.n, count))
        outside = (1 << self.model.B) - count
        if outside < 1:
            raise ValueError("This explicit outside-bucket construction requires a nonempty source complement")
        self.log_outside = math.log(outside)
        atom_sum = np.asarray(self.matrix.sum(axis=1)).ravel()
        self.mean = self.model.mean + (count * self.model.mean - atom_sum) / outside
        mean_norm = float(self.mean @ self.mean)
        if mean_norm > 1 + 1e-12:
            raise ValueError("Outside-source mean contradicts unit waveform energy")
        self.kappa = max(0., 1. - mean_norm) / self.model.n
        self.norms = np.asarray(self.matrix.power(2).sum(axis=0)).ravel()
        self.mean_dots = np.asarray(self.matrix.T @ self.mean).ravel()
        self.centered_norms = self.norms - 2 * self.mean_dots + mean_norm
        seconds = time.perf_counter() - started
        self.construction_seconds += seconds
        return fresh, seconds

    def vector(self, action):
        if any(not isinstance(word, (int, np.integer)) for word in action):
            raise ValueError("Action IDs must be integers")
        if any(not isinstance(count, (int, np.integer)) or count < 0 for count in action.values()):
            raise ValueError("Counts must be nonnegative integers")
        action = Counter({int(word): int(count) for word, count in action.items() if count})
        if any(word not in self.index for word in action) or action.total() > self.scenario.k:
            raise ValueError("Action must be supported on the complete bank with total at mostK")
        return np.array([action[word] for word in self.words], dtype=np.int64)

    def objective(self, counts):
        counts = np.asarray(counts)
        if counts.shape != (len(self.words),) or not np.isfinite(counts).all() or np.any(counts != np.floor(counts)):
            raise ValueError("Objective requires a finite integer bank-count vector")
        h = self.scenario.k - int(np.sum(counts))
        if h < 0 or np.any(np.asarray(counts) < 0):
            return -np.inf
        residual = self.scenario.y - self.matrix @ counts - h * self.mean
        variance = self.scenario.noise_var + h * self.kappa
        return float(-.5 * self.model.n * math.log(variance) - residual @ residual / (2 * variance)
                     + h * self.log_outside - gammaln(h + 1) - gammaln(np.asarray(counts) + 1).sum())

    def gains(self, counts, residual, gram_cache):
        h = self.scenario.k - int(counts.sum())
        variance = self.scenario.noise_var + h * self.kappa
        square = float(residual @ residual)
        dots = np.asarray(self.matrix.T @ residual).ravel()
        centered_dots = dots - float(residual @ self.mean)
        active = np.flatnonzero(counts)
        candidate_evaluations = 0
        best = (0., "none", -1, -1)
        if h and len(counts):
            smaller = self.scenario.noise_var + (h - 1) * self.kappa
            common = -.5 * self.model.n * math.log1p(-self.kappa / variance)
            common -= .5 * square * self.kappa / (variance * smaller)
            values = common + (centered_dots - .5 * self.centered_norms) / smaller
            values += math.log(h) - np.log(counts + 1) - self.log_outside
            j = int(np.argmax(values))
            if values[j] > best[0]:
                best = (float(values[j]), "add", -1, j)
            candidate_evaluations += len(counts)
        if len(active):
            larger = self.scenario.noise_var + (h + 1) * self.kappa
            common = -.5 * self.model.n * math.log1p(self.kappa / variance)
            common += .5 * square * self.kappa / (variance * larger)
            values = common - (centered_dots[active] + .5 * self.centered_norms[active]) / larger
            values += np.log(counts[active]) + self.log_outside - math.log(h + 1)
            j = int(np.argmax(values))
            if values[j] > best[0]:
                best = (float(values[j]), "drop", int(active[j]), -1)
            candidate_evaluations += len(active)
        for old in list(gram_cache):
            if not counts[old]:
                del gram_cache[old]
        gram_rows = 0
        for old in active:
            old = int(old)
            if old not in gram_cache:
                gram_cache[old] = (self.matrix.T @ self.matrix[:, old]).toarray().ravel()
                gram_rows += 1
            norm_difference = self.norms + self.norms[old] - 2 * gram_cache[old]
            values = (dots - dots[old] - .5 * norm_difference) / variance
            values += math.log(int(counts[old])) - np.log(counts + 1)
            values[old] = -np.inf
            if len(values):
                new = int(np.argmax(values))
                if values[new] > best[0]:
                    best = (float(values[new]), "swap", old, new)
            candidate_evaluations += max(0, len(counts) - 1)
        return best, {"candidate_gain_evaluations": candidate_evaluations, "gram_rows_constructed": gram_rows}

    def refit(self, initial, max_moves=2000):
        started = time.perf_counter()
        if not isinstance(max_moves, int) or max_moves < 0:
            raise ValueError("Require a nonnegative integer move cap")
        counts = self.vector(initial)
        h = self.scenario.k - int(counts.sum())
        residual = self.scenario.y - self.matrix @ counts - h * self.mean
        initial_objective = self.objective(counts)
        history, gram_cache, evaluations, gram_rows, max_cached_rows = [], {}, 0, 0, 0
        converged, next_gain = False, None
        for step in range(max_moves + 1):
            best, work = self.gains(counts, residual, gram_cache)
            evaluations += work["candidate_gain_evaluations"]
            gram_rows += work["gram_rows_constructed"]
            max_cached_rows = max(max_cached_rows, len(gram_cache))
            gain, kind, old, new = best
            next_gain = gain
            if gain <= 1e-8:
                converged = True
                break
            if step == max_moves:
                break
            before = self.objective(counts)
            if old >= 0:
                word = self.words[old]
                positions, values = self.atoms[word]
                residual[positions] += values
                counts[old] -= 1
                if kind == "drop":
                    residual -= self.mean
            if new >= 0:
                word = self.words[new]
                positions, values = self.atoms[word]
                residual[positions] -= values
                counts[new] += 1
                if kind == "add":
                    residual += self.mean
            after = self.objective(counts)
            if abs(after - before - gain) > 1e-7 * (1 + abs(gain)) or after < before - 1e-8:
                raise AssertionError("Count move does not agree with the common objective")
            history.append({"kind": kind, "old": self.words[old] if old >= 0 else None,
                            "new": self.words[new] if new >= 0 else None, "gain": gain,
                            "outside_count": self.scenario.k - int(counts.sum())})
        self.last_raw_residual = self.scenario.y - self.matrix @ counts
        action = Counter({word: int(counts[j]) for j, word in enumerate(self.words) if counts[j]})
        return action, {"converged": converged, "move_cap_hit": not converged, "max_moves": max_moves,
                        "moves": history, "next_positive_gain": next_gain, "bank_size": len(self.words),
                        "initial_objective": initial_objective, "final_objective": self.objective(counts),
                        "outside_count": self.scenario.k - action.total(), "outside_mean_norm": float(np.linalg.norm(self.mean)),
                        "outside_trace_variance_per_coordinate": self.kappa,
                        "candidate_gain_evaluations": evaluations, "gram_rows_constructed": gram_rows,
                        "largest_cached_gram_rows": max_cached_rows, "seconds": time.perf_counter() - started,
                        "scope": "Fixed-bank local count MAP ascent; outside Gaussian approximation, not count-PUPE Bayes action"}


def continue_full_patterns(scenario, initial, ids, *, list_size=128, max_passes=3, max_moves=2000, progress=None):
    started = time.perf_counter()
    if list_size != 128 or max_passes != 3 or max_moves != 2000:
        raise ValueError("This phase freezes SCL128, three full passes and2000 moves per refit")
    t0 = time.perf_counter()
    bank = CountBank(scenario, set(ids).union(initial))
    initial_bank_seconds = time.perf_counter() - t0
    action, first_refit = bank.refit(initial, max_moves=max_moves)
    refit_only = action.copy()
    t0 = time.perf_counter()
    code = PolarCode(scenario.model.b, scenario.model.code.n, scenario.model.code.crc_bits, list_size)
    query_code_setup_seconds = time.perf_counter() - t0
    traces, refits, passes = [], [], []
    scl_seconds = llr_seconds = ranking_seconds = 0.
    complete, stop = first_refit["converged"], "initial_move_cap" if not first_refit["converged"] else "pass_cap"
    for pass_index in range(max_passes if complete else 0):
        t0 = time.perf_counter()
        score = np.abs(bank.last_raw_residual[scenario.model.positions]).sum(1)
        order = np.argsort(-score, kind="stable")
        ranking_seconds += time.perf_counter() - t0
        start_queries, start_ids, pass_moves = len(traces), len(bank.words), 0
        for query_index, pattern_value in enumerate(order):
            pattern = int(pattern_value)
            positions = scenario.model.positions[pattern]
            t0 = time.perf_counter()
            local = bank.last_raw_residual[positions].copy()
            h = scenario.k - action.total()
            occupied = [(word, count) for word, count in action.items() if word >> scenario.model.b == pattern]
            contexts = [("raw", local, h)]
            if occupied:
                restored = local.copy()
                for word, count in occupied:
                    restored += count * bank.atoms[word][1]
                contexts.append(("restore_pattern", restored, h + sum(count for _, count in occupied)))
            llr_seconds += time.perf_counter() - t0
            discovered = set()
            for name, observation, pending in contexts:
                t0 = time.perf_counter()
                variance = scenario.noise_var + max(pending - 1, 0) / scenario.model.n
                llr = 2 * scenario.model.amplitudes[pattern] * observation / variance
                llr_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                payloads, _ = code.decode_list(llr)
                seconds = time.perf_counter() - t0
                scl_seconds += seconds
                found = [scenario.model.word(pattern, payload) for payload in payloads]
                discovered.update(found)
                traces.append({"pass": pass_index, "pattern": pattern, "context": name, "pending": pending,
                               "variance": variance, "ids": found, "scl_seconds": seconds})
            fresh, _ = bank.extend(discovered)
            if fresh:
                action, fitted = bank.refit(action, max_moves=max_moves)
                pass_moves += len(fitted["moves"])
                refits.append({"pass": pass_index, "pattern": pattern, "new_ids": fresh, "meta": fitted})
                if not fitted["converged"]:
                    complete, stop = False, "count_move_cap"
                    break
            if progress is not None and ((query_index + 1) % 512 == 0 or query_index + 1 == scenario.model.p):
                progress({"pass": pass_index, "patterns_completed": query_index + 1, "queries": len(traces),
                          "bank_size": len(bank.words), "output_count": action.total(),
                          "elapsed_seconds": time.perf_counter() - started})
        new_ids = len(bank.words) - start_ids
        passes.append({"pass": pass_index, "queries": len(traces) - start_queries, "new_ids": new_ids,
                       "moves": pass_moves, "output_count": action.total(), "bank_size": len(bank.words),
                       "complete_pattern_coverage": complete})
        if not complete:
            break
        if not new_ids:
            stop = "full_pass_no_new_ids"
            break
    meta = {"initial_refit_action": sorted(refit_only.items()), "initial_refit": first_refit,
            "passes": passes, "queries": traces, "refits": refits, "complete": complete, "stop": stop,
            "pass_cap_with_new_ids": stop == "pass_cap" and bool(passes and passes[-1]["new_ids"]),
            "scl_seconds": scl_seconds, "llr_seconds": llr_seconds, "ranking_seconds": ranking_seconds,
            "query_code_setup_seconds": query_code_setup_seconds,
            "bank_construction_seconds": bank.construction_seconds,
            "initial_bank_seconds": initial_bank_seconds,
            "count_refit_seconds": first_refit["seconds"] + sum(row["meta"]["seconds"] for row in refits),
            "seconds": time.perf_counter() - started, "final_bank_size": len(bank.words),
            "bank_sparse_bytes": bank.matrix.data.nbytes + bank.matrix.indices.nbytes + bank.matrix.indptr.nbytes,
            "scope": "All-pattern bounded continuation; no bank truncation, no oracle acceptance, all multiplicities legal"}
    return action, set(bank.words), meta


def run(scenario, **params):
    started = time.perf_counter()
    job_path = str(Path(__file__).resolve().parents[1] / "jobs/037_polar_tractable")
    if job_path not in sys.path:
        sys.path.insert(0, job_path)
    from rollback_frames import native
    from rollback_progressive import progressive
    accepted, native_meta = native(scenario)
    initial, fresh, batches, rollback_seconds = progressive(scenario, list(accepted), native_meta["valid_ids"], "ungated")
    base_bank = set(native_meta["valid_ids"]).union(fresh, initial)
    action, bank, meta = continue_full_patterns(scenario, initial, base_bank, **params)
    return dict(action), {"native": native_meta, "rollback_seconds": rollback_seconds, "rollback_batches": batches,
                          "rollback_action": sorted(initial.items()), "base_bank": sorted(base_bank),
                          "bank": sorted(bank), "continuation": meta, "total_seconds": time.perf_counter() - started}
