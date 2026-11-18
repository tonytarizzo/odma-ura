"""Unchanged raw count feedback with exact query reuse and a separate terminal support-completion decision."""

from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_cavity import action_hash, cavity_query
from framework.polar_full_pattern import CountBank


class PayloadMemo:
    """One fixed code and one receiver frame; byte identity, not a rounded or hashed channel surrogate."""

    def __init__(self, code, enabled=True):
        self.code, self.enabled, self.cache = code, bool(enabled), {}
        self.calls = self.hits = self.data_bytes = self.entry_object_bytes = 0
        self.scl_seconds = self.decode_seconds = 0.
        self.last_hit, self.last_scl_seconds = False, 0.

    def decode_list(self, llr):
        started = time.perf_counter()
        llr = np.asarray(llr, dtype=np.float64)
        if llr.shape != (self.code.n,) or not np.isfinite(llr).all():
            raise ValueError("Expected one finite LLR per coded bit")
        key = llr.tobytes() if self.enabled else None
        self.last_hit = self.enabled and key in self.cache
        self.last_scl_seconds = 0.
        if self.last_hit:
            self.hits += 1
            payloads, metrics = self.cache[key]
            result = payloads.copy(), metrics.copy()
        else:
            t0 = time.perf_counter()
            result = self.code.decode_list(llr)
            self.last_scl_seconds = time.perf_counter() - t0
            self.scl_seconds += self.last_scl_seconds
            self.calls += 1
            if self.enabled:
                saved = result[0].copy(), result[1].copy()
                self.cache[key] = saved
                data = len(key) + sum(array.nbytes for array in saved)
                self.data_bytes += data
                self.entry_object_bytes += sys.getsizeof(key) + sys.getsizeof(saved)
                self.entry_object_bytes += sum(sys.getsizeof(array) for array in saved) - data
        self.decode_seconds += time.perf_counter() - started
        return result

    def stats(self):
        objects = self.entry_object_bytes + sys.getsizeof(self.cache)
        return {"actual_scl_calls": self.calls, "cache_hits": self.hits, "cache_entries": len(self.cache),
                "cache_data_bytes": self.data_bytes, "cache_object_bytes": objects,
                "cache_bytes": self.data_bytes + objects, "scl_seconds": self.scl_seconds,
                "decode_seconds": self.decode_seconds, "cache_seconds": self.decode_seconds - self.scl_seconds}


def continue_feedback(scenario, initial, ids, *, memoize=True, progress=None, code_factory=PolarCode, progress_extra_sweep=False,
                      count_growth=False):
    started = time.perf_counter()
    if not isinstance(progress_extra_sweep, bool):
        raise ValueError("Require a Boolean progress_extra_sweep option")
    if not isinstance(count_growth, bool) or (count_growth and not progress_extra_sweep):
        raise ValueError("Boolean count_growth requires progress_extra_sweep=True when enabled")
    model = scenario.model
    if not isinstance(model.code, PolarCode):
        raise ValueError("This continuation is for the original polar encoder")
    t0 = time.perf_counter()
    bank = CountBank(scenario, set(ids).union(initial))
    initial_bank_seconds = time.perf_counter() - t0
    action, first_refit = bank.refit(initial, max_moves=2000)
    refit_only = action.copy()
    t0 = time.perf_counter()
    code = code_factory(model.b, model.code.n, model.code.crc_bits, 128)
    if (not isinstance(code, PolarCode) or (code.payload_bits, code.n, code.crc_bits, code.list_size)
            != (model.b, model.code.n, model.code.crc_bits, 128)
            or not np.array_equal(code.information, model.code.information)
            or not np.array_equal(code.frozen, model.code.frozen)):
        raise ValueError("Query code does not match the scenario encoder and fixed SCL128 budget")
    decoder = PayloadMemo(code, memoize)
    query_code_setup_seconds = time.perf_counter() - t0
    query_budget = int(scenario.k) + 5
    queries, refits, sweeps = [], [], []
    llr_seconds = ranking_seconds = bank_update_seconds = audit_seconds = conversion_seconds = 0.
    complete = first_refit["converged"]
    stop = "sweep_cap" if complete else "initial_move_cap"
    action_path, extra_initial_action = [], None
    growth_trace, fourth_endpoint = [], None
    maximum_sweeps = int(scenario.k) + 4 if count_growth else (4 if progress_extra_sweep else 3)
    for sweep in range(maximum_sweeps if complete else 0):
        if sweep == 3:
            if not sweeps[-1]["moves"]:
                stop = "third_sweep_no_count_moves"
                break
            t0 = time.perf_counter()
            extra_initial_action = sorted(action.items())
            audit_seconds += time.perf_counter() - t0
        if count_growth and sweep >= 4:
            assert fourth_endpoint is not None
            assert sweep - 3 <= scenario.k - fourth_endpoint + 1
            assert action.total() >= fourth_endpoint + sweep - 4
        start_count = action.total() if count_growth else None
        start_queries, start_ids, start_refits = len(queries), len(bank.words), len(refits)
        t0 = time.perf_counter()
        score = np.abs(bank.last_raw_residual[model.positions]).sum(1)
        patterns = np.argsort(-score, kind="stable")[:min(model.p, query_budget)].tolist()
        ranking_seconds += time.perf_counter() - t0
        for pattern in patterns:
            t0 = time.perf_counter()
            before_hash = action_hash(action)
            audit_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            llr, context = cavity_query(scenario, action, bank.last_raw_residual, pattern)
            preparation_seconds = time.perf_counter() - t0
            llr_seconds += preparation_seconds
            t0 = time.perf_counter()
            llr_hash = hashlib.sha256(llr.tobytes()).hexdigest()
            audit_seconds += time.perf_counter() - t0
            before_decode = decoder.decode_seconds
            payloads, _ = decoder.decode_list(llr)
            decode_seconds = decoder.decode_seconds - before_decode
            t0 = time.perf_counter()
            found = [model.word(pattern, payload) for payload in payloads]
            conversion_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            fresh, _ = bank.extend(found)
            update_seconds = time.perf_counter() - t0
            bank_update_seconds += update_seconds
            refit_index = None
            if fresh:
                if progress_extra_sweep:
                    t0 = time.perf_counter()
                    before_action = sorted(action.items())
                    audit_seconds += time.perf_counter() - t0
                action, fitted = bank.refit(action, max_moves=2000)
                refit_index = len(refits)
                refits.append({"sweep": sweep, "query": len(queries), "new_ids": fresh, "meta": fitted})
                if progress_extra_sweep and fitted["moves"]:
                    t0 = time.perf_counter()
                    action_path.append({"query": len(queries), "sweep": sweep, "refit_index": refit_index,
                                        "before": before_action, "after": sorted(action.items()),
                                        "count_moves": len(fitted["moves"])})
                    audit_seconds += time.perf_counter() - t0
                if not fitted["converged"]:
                    complete, stop = False, "count_move_cap"
            queries.append({"query": len(queries), "sweep": sweep, "stage": "ranked_patterns", "pattern": pattern,
                            "list_size": 128, **context, "pre_action_sha256": before_hash, "llr_sha256": llr_hash,
                            "ids": found, "new_ids": fresh, "refit_index": refit_index,
                            "llr_seconds": preparation_seconds, "scl_seconds": decoder.last_scl_seconds,
                            "decode_seconds": decode_seconds, "cache_hit": decoder.last_hit,
                            "bank_update_seconds": update_seconds})
            if progress is not None and len(queries) % 25 == 0:
                progress({"mode": "memo_raw128" if memoize else "raw128", "sweep": sweep, "queries": len(queries),
                          "actual_scl_calls": decoder.calls, "cache_hits": decoder.hits, "bank_size": len(bank.words),
                          "output_count": action.total(), "elapsed_seconds": time.perf_counter() - started})
            if not complete:
                break
        new_ids = len(bank.words) - start_ids
        sweeps.append({"sweep": sweep, "id_queue": [], "withdrawn_ids_skipped": [], "ranked_patterns": patterns,
                       "queries": len(queries) - start_queries, "new_ids": new_ids,
                       "moves": sum(len(row["meta"]["moves"]) for row in refits[start_refits:]),
                       "output_count": action.total(), "bank_size": len(bank.words), "complete_sweep": complete})
        if progress is not None:
            progress({"mode": "memo_raw128" if memoize else "raw128", "sweep": sweep, "sweep_complete": complete,
                      "queries": len(queries), "new_ids": new_ids, "output_count": action.total(),
                      "actual_scl_calls": decoder.calls, "cache_hits": decoder.hits,
                      "elapsed_seconds": time.perf_counter() - started})
        if count_growth and sweep >= 3:
            t0 = time.perf_counter()
            end_count = action.total()
            grew = end_count > start_count
            if sweep == 3 and complete and grew:
                fourth_endpoint = end_count
            reason = (stop if not complete else "full_sweep_no_new_ids" if not new_ids
                      else "strict_endpoint_growth" if grew else "no_endpoint_count_growth")
            growth_trace.append({"sweep": sweep, "start_count": start_count, "end_count": end_count,
                                 "grew": grew, "complete_sweep": complete, "new_ids": new_ids,
                                 "continue": complete and bool(new_ids) and grew, "reason": reason})
            audit_seconds += time.perf_counter() - t0
        if not complete:
            break
        if not new_ids:
            stop = "full_sweep_no_new_ids"
            break
        if count_growth and sweep >= 3 and not grew:
            stop = "no_endpoint_count_growth"
            break
    if count_growth:
        assert len(sweeps) <= scenario.k + 4
        if fourth_endpoint is not None:
            assert max(0, len(sweeps) - 4) <= scenario.k - fourth_endpoint + 1
        assert not (complete and stop == "sweep_cap"), "Integer growth bound exhausted without a stopping condition"
    seconds = time.perf_counter() - started
    count_refit_seconds = first_refit["seconds"] + sum(row["meta"]["seconds"] for row in refits)
    charged = (initial_bank_seconds + query_code_setup_seconds + count_refit_seconds + decoder.decode_seconds + llr_seconds
               + ranking_seconds + bank_update_seconds + audit_seconds + conversion_seconds)
    meta = {"mode": "raw128", "memoized": bool(memoize), "list_size": 128, "query_budget_per_sweep": query_budget,
            "initial_refit_action": sorted(refit_only.items()), "initial_refit": first_refit,
            "sweeps": sweeps, "queries": queries, "refits": refits, "complete": complete, "stop": stop,
            "sweep_cap_with_new_ids": stop == "sweep_cap" and bool(sweeps and sweeps[-1]["new_ids"]),
            **decoder.stats(), "llr_seconds": llr_seconds, "ranking_seconds": ranking_seconds,
            "query_code_setup_seconds": query_code_setup_seconds, "initial_bank_seconds": initial_bank_seconds,
            "bank_update_seconds": bank_update_seconds, "bank_construction_seconds": bank.construction_seconds,
            "count_refit_seconds": count_refit_seconds, "audit_seconds": audit_seconds,
            "candidate_conversion_seconds": conversion_seconds, "other_seconds": seconds - charged,
            "seconds": seconds, "final_bank_size": len(bank.words),
            "bank_sparse_bytes": bank.matrix.data.nbytes + bank.matrix.indices.nbytes + bank.matrix.indptr.nbytes,
            "scope": "Unchanged raw128 logical trajectory; exact fixed-code payload reuse on identical LLR bytes",
            "completion_scope": "Declared sweeps and count refits completed; not a global discovery or MAP certificate"}
    if progress_extra_sweep:
        meta.update(progress_extra_sweep=True, maximum_sweeps=4, extra_sweep_triggered=extra_initial_action is not None,
                    third_sweep_count_moves=sweeps[2]["moves"] if len(sweeps) >= 3 else 0,
                    extra_sweep_initial_action=extra_initial_action, count_action_path=action_path,
                    scope="One fourth fixed-queue sweep only after actual third-sweep count moves; same bank and exact cache")
    if count_growth:
        meta.update(count_growth=True, maximum_sweeps=maximum_sweeps, count_growth_trace=growth_trace,
                    fourth_endpoint_count=fourth_endpoint, later_sweeps=max(0, len(sweeps) - 4),
                    later_sweep_bound=int(scenario.k) - fourth_endpoint + 1 if fourth_endpoint is not None else 0,
                    scope="Unchanged first-four policy; later fixed queues follow strict endpoint cancellation-count growth, "
                          "with priority for existing no-new-ID and incomplete-refit stops. Completion remains terminal.")
    return action, set(bank.words), meta


def support_completion(scenario, action, ids):
    """Terminal first-copy additions only; no completed action is returned to cancellation/discovery."""
    started = time.perf_counter()
    if any(not isinstance(count, (int, np.integer)) or count < 0 for count in action.values()):
        raise ValueError("Require nonnegative integer counts")
    words = set(ids).union(action)
    if any(not isinstance(word, (int, np.integer)) for word in words):
        raise ValueError("Message IDs must be integers")
    for word in words:
        scenario.model.split(word)
    output = Counter({int(word): int(count) for word, count in action.items() if count})
    remaining = scenario.k - output.total()
    if remaining < 0:
        raise ValueError("Input action exceeds known K")
    absent = sorted(words - set(output))
    additions, steps = [], []
    bank_seconds = ranking_seconds = 0.
    method = "no_capacity" if not remaining else "all_fit"
    if len(absent) <= remaining:
        additions = absent
        output.update(additions)
    elif remaining:
        method = "count_add_gain"
        t0 = time.perf_counter()
        bank = CountBank(scenario, words)
        bank_seconds = time.perf_counter() - t0
        counts = bank.vector(output)
        residual = scenario.y - bank.matrix @ counts - remaining * bank.mean
        eligible = [bank.index[word] for word in absent]
        t0 = time.perf_counter()
        while remaining and eligible:
            variance = scenario.noise_var + remaining * bank.kappa
            smaller = scenario.noise_var + (remaining - 1) * bank.kappa
            common = -.5 * scenario.model.n * math.log1p(-bank.kappa / variance)
            common -= .5 * float(residual @ residual) * bank.kappa / (variance * smaller)
            common += math.log(remaining) - bank.log_outside
            dots = np.asarray(bank.matrix.T @ residual).ravel() - float(residual @ bank.mean)
            gains = common + (dots - .5 * bank.centered_norms) / smaller - np.log(counts + 1)
            selected = eligible[int(np.argmax(gains[eligible]))]
            word, gain = bank.words[selected], float(gains[selected])
            before = bank.objective(counts)
            counts[selected] += 1
            after = bank.objective(counts)
            if abs(after - before - gain) > 1e-7 * (1 + abs(gain)):
                raise AssertionError("Completion increment does not match CountBank objective")
            positions, values = bank.atoms[word]
            residual[positions] -= values
            residual += bank.mean
            remaining -= 1
            eligible.remove(selected)
            additions.append(word)
            output[word] += 1
            steps.append({"word": word, "gain": gain, "remaining_slots": remaining})
        ranking_seconds = time.perf_counter() - t0
    return output, {"method": method, "additions": sorted((word, 1) for word in additions), "steps": steps,
                    "input_count": sum(action.values()), "output_count": output.total(), "bank_size": len(words),
                    "absent_bank_ids": len(absent), "bank_construction_seconds": bank_seconds,
                    "ranking_seconds": ranking_seconds, "seconds": time.perf_counter() - started,
                    "scope": "Pointwise nonworsening count misses by terminal additions; false copies may increase. "
                             "First-copy completion preserves existing multiplicities; gain ranking uses the Gaussian-outside CountBank objective."}


def run(scenario, *, memoize=True, complete_support=False, query_code_factory=PolarCode, progress_extra_sweep=False,
        count_growth=False):
    started = time.perf_counter()
    job_path = str(Path(__file__).resolve().parents[1] / "jobs/037_polar_tractable")
    if job_path not in sys.path:
        sys.path.insert(0, job_path)
    from rollback_frames import native
    from rollback_progressive import progressive
    accepted, native_meta = native(scenario)
    initial, fresh, batches, rollback_seconds = progressive(scenario, list(accepted), native_meta["valid_ids"], "ungated",
                                                          code_factory=query_code_factory)
    base_bank = set(native_meta["valid_ids"]).union(fresh, initial)
    cancellation, bank, meta = continue_feedback(scenario, initial, base_bank, memoize=memoize, code_factory=query_code_factory,
                                                progress_extra_sweep=progress_extra_sweep, count_growth=count_growth)
    action, completion = cancellation, None
    if complete_support:
        action, completion = support_completion(scenario, cancellation, bank)
    return dict(action), {"native": native_meta, "rollback_seconds": rollback_seconds, "rollback_batches": batches,
                          "rollback_action": sorted(initial.items()), "base_bank": sorted(base_bank), "bank": sorted(bank),
                          "cancellation_action": sorted(cancellation.items()), "continuation": meta,
                          "support_completion": completion, "complete_support": bool(complete_support),
                          "complete": meta["complete"], "total_seconds": time.perf_counter() - started}
