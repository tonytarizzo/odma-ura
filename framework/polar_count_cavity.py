"""Bounded accepted-copy cavity proposals with one shared, unanchored physical count objective."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_full_pattern import CountBank


MODES = ("all_ids", "ranked_restore", "raw128", "raw512")


def action_hash(action):
    encoded = json.dumps(sorted((int(w), int(c)) for w, c in action.items() if c), separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def cavity_query(scenario, action, residual, pattern, *, word=None, restore_pattern=False):
    """Restore proposed copies without accepting them; the residual and action are unchanged."""
    model = scenario.model
    if not isinstance(pattern, (int, np.integer)) or not 0 <= pattern < model.p:
        raise ValueError("Query pattern must be a valid integer")
    if word is not None and restore_pattern:
        raise ValueError("Choose a one-copy or whole-pattern cavity, not both")
    if any(not isinstance(c, (int, np.integer)) or c < 0 for c in action.values()):
        raise ValueError("Require nonnegative integer accepted counts")
    h = scenario.k - sum(action.values())
    if h < 0:
        raise ValueError("Accepted counts exceed known K")
    if word is not None:
        if not isinstance(word, (int, np.integer)) or action.get(word, 0) < 1 or model.split(word)[0] != pattern:
            raise ValueError("A one-copy cavity requires a current ID on the queried pattern")
        restored, name = [(int(word), 1)], "restore_one"
    elif restore_pattern:
        restored = sorted((int(w), int(c)) for w, c in action.items() if c and model.split(w)[0] == pattern)
        name = "restore_pattern"
    else:
        restored, name = [], "raw"
    local = np.asarray(residual)[model.positions[pattern]].copy()
    for current, count in restored:
        _, values = model.symbols(current)
        local += count * values
    restored_count = sum(c for _, c in restored)
    pending = h + restored_count
    variance = scenario.noise_var + max(pending - 1, 0) / model.n
    llr = 2 * model.amplitudes[pattern] * local / variance
    return llr, {"context": name, "restored": restored, "restored_count": restored_count,
                 "outside_count": h, "pending": pending, "variance": variance}


def continue_cavity(scenario, initial, ids, *, mode="all_ids", max_sweeps=3, max_moves=2000, progress=None):
    started = time.perf_counter()
    if mode not in MODES or max_sweeps != 3 or max_moves != 2000:
        raise ValueError("Require a declared cavity mode, three sweeps and 2000 moves per refit")
    model = scenario.model
    if not isinstance(model.code, PolarCode):
        raise ValueError("This continuation is for the original polar encoder")
    t0 = time.perf_counter()
    bank = CountBank(scenario, set(ids).union(initial))
    initial_bank_seconds = time.perf_counter() - t0
    action, first_refit = bank.refit(initial, max_moves=max_moves)
    refit_only = action.copy()
    t0 = time.perf_counter()
    list_size = 512 if mode == "raw512" else 128
    code = PolarCode(model.b, model.code.n, model.code.crc_bits, list_size)
    if not np.array_equal(code.information, model.code.information):
        raise ValueError("Query code does not match the scenario encoder")
    query_code_setup_seconds = time.perf_counter() - t0
    query_budget = 40 if mode == "raw512" else scenario.k + 5
    queries, refits, sweeps = [], [], []
    scl_seconds = llr_seconds = ranking_seconds = bank_update_seconds = audit_seconds = conversion_seconds = 0.
    complete = first_refit["converged"]
    stop = "sweep_cap" if complete else "initial_move_cap"

    def rank_patterns(limit):
        nonlocal ranking_seconds
        t0 = time.perf_counter()
        score = np.abs(bank.last_raw_residual[model.positions]).sum(1)
        selected = np.argsort(-score, kind="stable")[:min(model.p, limit)].tolist()
        ranking_seconds += time.perf_counter() - t0
        return selected

    def query(pattern, sweep, stage, word=None, restore_pattern=False):
        nonlocal action, complete, stop, scl_seconds, llr_seconds, bank_update_seconds, audit_seconds, conversion_seconds
        t0 = time.perf_counter()
        before_hash = action_hash(action)
        audit_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        llr, context = cavity_query(scenario, action, bank.last_raw_residual, pattern,
                                    word=word, restore_pattern=restore_pattern)
        preparation_seconds = time.perf_counter() - t0
        llr_seconds += preparation_seconds
        t0 = time.perf_counter()
        llr_hash = hashlib.sha256(llr.tobytes()).hexdigest()
        audit_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        payloads, _ = code.decode_list(llr)
        decoding_seconds = time.perf_counter() - t0
        scl_seconds += decoding_seconds
        t0 = time.perf_counter()
        found = [model.word(pattern, payload) for payload in payloads]
        conversion_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        fresh, _ = bank.extend(found)
        update_seconds = time.perf_counter() - t0
        bank_update_seconds += update_seconds
        refit_index = None
        if fresh:
            action, fitted = bank.refit(action, max_moves=max_moves)
            refit_index = len(refits)
            refits.append({"sweep": sweep, "query": len(queries), "new_ids": fresh, "meta": fitted})
            if not fitted["converged"]:
                complete, stop = False, "count_move_cap"
        queries.append({"query": len(queries), "sweep": sweep, "stage": stage, "pattern": pattern,
                        "list_size": list_size, **context, "pre_action_sha256": before_hash,
                        "llr_sha256": llr_hash, "ids": found, "new_ids": fresh, "refit_index": refit_index,
                        "llr_seconds": preparation_seconds, "scl_seconds": decoding_seconds,
                        "bank_update_seconds": update_seconds})
        if progress is not None and len(queries) % 25 == 0:
            progress({"mode": mode, "sweep": sweep, "queries": len(queries), "bank_size": len(bank.words),
                      "output_count": action.total(), "elapsed_seconds": time.perf_counter() - started})

    for sweep in range(max_sweeps if complete else 0):
        start_queries, start_ids, start_refits = len(queries), len(bank.words), len(refits)
        queue, skipped, patterns = [], [], []
        if mode == "all_ids":
            queue = sorted(action)
            for word in queue:
                if not action.get(word, 0):
                    skipped.append(word)
                    continue
                query(model.split(word)[0], sweep, "current_ids", word=word)
                if not complete:
                    break
            if complete:
                patterns = rank_patterns(query_budget - (len(queries) - start_queries))
                for pattern in patterns:
                    query(pattern, sweep, "raw_fill")
                    if not complete:
                        break
        else:
            patterns = rank_patterns(query_budget)
            for pattern in patterns:
                query(pattern, sweep, "ranked_patterns", restore_pattern=mode == "ranked_restore")
                if not complete:
                    break
        new_ids = len(bank.words) - start_ids
        sweeps.append({"sweep": sweep, "id_queue": queue, "withdrawn_ids_skipped": skipped,
                       "ranked_patterns": patterns, "queries": len(queries) - start_queries,
                       "new_ids": new_ids, "moves": sum(len(r["meta"]["moves"]) for r in refits[start_refits:]),
                       "output_count": action.total(), "bank_size": len(bank.words), "complete_sweep": complete})
        if progress is not None:
            progress({"mode": mode, "sweep": sweep, "sweep_complete": complete, "queries": len(queries),
                      "new_ids": new_ids, "output_count": action.total(),
                      "elapsed_seconds": time.perf_counter() - started})
        if not complete:
            break
        if not new_ids:
            stop = "full_sweep_no_new_ids"
            break
    seconds = time.perf_counter() - started
    count_refit_seconds = first_refit["seconds"] + sum(row["meta"]["seconds"] for row in refits)
    charged = (initial_bank_seconds + query_code_setup_seconds + count_refit_seconds + scl_seconds + llr_seconds
               + ranking_seconds + bank_update_seconds + audit_seconds + conversion_seconds)
    meta = {"mode": mode, "list_size": list_size, "query_budget_per_sweep": query_budget,
            "initial_refit_action": sorted(refit_only.items()), "initial_refit": first_refit,
            "sweeps": sweeps, "queries": queries, "refits": refits, "complete": complete, "stop": stop,
            "sweep_cap_with_new_ids": stop == "sweep_cap" and bool(sweeps and sweeps[-1]["new_ids"]),
            "scl_seconds": scl_seconds, "llr_seconds": llr_seconds, "ranking_seconds": ranking_seconds,
            "query_code_setup_seconds": query_code_setup_seconds, "initial_bank_seconds": initial_bank_seconds,
            "bank_update_seconds": bank_update_seconds, "bank_construction_seconds": bank.construction_seconds,
            "count_refit_seconds": count_refit_seconds, "audit_seconds": audit_seconds,
            "candidate_conversion_seconds": conversion_seconds, "other_seconds": seconds - charged,
            "seconds": seconds, "final_bank_size": len(bank.words),
            "bank_sparse_bytes": bank.matrix.data.nbytes + bank.matrix.indices.nbytes + bank.matrix.indptr.nbytes,
            "scope": "Bounded cavity proposals and fixed-bank local count MAP ascent; outside Gaussian approximation",
            "completion_scope": "Declared sweeps and count refits completed; not a global discovery or MAP certificate"}
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
    action, bank, meta = continue_cavity(scenario, initial, base_bank, **params)
    return dict(action), {"native": native_meta, "rollback_seconds": rollback_seconds, "rollback_batches": batches,
                          "rollback_action": sorted(initial.items()), "base_bank": sorted(base_bank),
                          "bank": sorted(bank), "continuation": meta, "total_seconds": time.perf_counter() - started}
