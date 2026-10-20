"""Tiny matched approximation diagnostic; exhaustive references are never inputs to the proposer."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, softmax

from baselines import message_bits
from baselines.polar import PolarCode, append_crc
from tests.polar_assumptions_test import count_forest, enumerated_surrogate
from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import PolarPosteriorTests, exact_posterior


def encode_words(base, energy, words):
    words = np.asarray(words, dtype=int)
    patterns, payloads = np.divmod(words, 1 << base.polar.payload_bits)
    signs = 1. - 2. * base.polar.encode(message_bits(payloads, base.polar.payload_bits))
    result = np.zeros((base.n, len(words)))
    result[base.positions[patterns], np.arange(len(words))[:, None]] = np.sqrt(energy[patterns]) * signs
    return result


def compact_moments(base, energy):
    """Uniform full-message mean and coordinate second moment, without enumerating payloads."""
    generator = base.polar.encode(np.eye(base.polar.payload_bits, dtype=np.uint8)).T
    constant = ~generator.any(1)
    mean, second = np.zeros(base.n), np.zeros(base.n)
    np.add.at(mean, base.positions.ravel(), (np.sqrt(energy) * constant).ravel() / base.num_patterns)
    np.add.at(second, base.positions.ravel(), energy.ravel() / base.num_patterns)
    return mean, second


def remaining_moments(mean, second, alphabet, signals, h, sampling):
    """Prior sum moments after a supplied prefix; dependence of a proposed next word is handled in reranking."""
    if sampling == "distinct":
        available = alphabet - len(signals)
        if not 0 <= h <= available:
            raise ValueError("Infeasible distinct-source continuation")
        if available == 0:
            return np.zeros_like(mean), np.zeros_like(mean)
        mean = (alphabet * mean - np.sum(signals, axis=0)) / available if len(signals) else mean
        second = (alphabet * second - np.square(signals).sum(0)) / available if len(signals) else second
        factor = h * (available - h) / (available - 1) if available > 1 else 0.
    else:
        factor = h
    variance = second - mean ** 2
    if np.min(variance) < -1e-12:
        raise ArithmeticError("Invalid coordinate variance")
    return h * mean, factor * np.maximum(variance, 0.)


def crc_constrained_list(code, llr):
    """Prototype for this encoder's causal payload-then-CRC ordering; force CRC bits inside the SCL traversal."""
    llr = np.asarray(llr, dtype=float)
    if llr.shape != (code.n,) or not np.all(np.isfinite(llr)):
        raise ValueError("Expected a finite LLR vector")
    crc_index = {int(position): j for j, position in enumerate(code.information[code.payload_bits:])}
    payload_index = {int(position): j for j, position in enumerate(code.information[:code.payload_bits])}

    def visit(alpha, metrics, start, data):
        width, paths = alpha.shape[1], alpha.shape[0]
        if width == 1:
            if code.frozen[start] or start in crc_index:
                if code.frozen[start]:
                    bits = np.zeros((paths, 1), dtype=np.uint8)
                else:
                    bits = data[:, code.payload_bits + crc_index[start], None]
                cost = np.logaddexp(0., -(1. - 2. * bits[:, 0]) * alpha[:, 0])
                return bits, metrics + cost, np.arange(paths), data
            both = metrics[:, None] + np.logaddexp(0., -alpha * np.array([1., -1.]))
            chosen = np.argsort(both.ravel(), kind="stable")[:code.list_size]
            bits = (chosen % 2).astype(np.uint8)[:, None]
            data = data[chosen // 2].copy()
            data[:, payload_index[start]] = bits[:, 0]
            if payload_index[start] == code.payload_bits - 1:
                data = append_crc(data[:, :code.payload_bits], code.crc_bits)
            return bits, both.ravel()[chosen], chosen // 2, data
        half = width // 2
        a, b = alpha[:, :half], alpha[:, half:]
        left, left_metrics, origins, left_data = visit(np.logaddexp(0., a + b) - np.logaddexp(a, b), metrics, start, data)
        a, b = a[origins], b[origins]
        right, final_metrics, right_origins, final_data = visit(b + (1. - 2. * left) * a, left_metrics, start + half, left_data)
        beta = np.concatenate((left[right_origins] ^ right, right), axis=1)
        return beta, final_metrics, origins[right_origins], final_data

    _, metrics, _, data = visit(llr[None, :], np.zeros(1), 0, np.zeros((1, code.k), dtype=np.uint8))
    order = np.argsort(metrics, kind="stable")
    return data[order, :code.payload_bits], metrics[order]


def polar_beam(base, energy, y, k, sigma2, sampling, width, list_size, moment, constrained_crc=False):
    """A bounded proposal tree; its final list is not asserted to have posterior weights or coverage."""
    code = PolarCode(base.polar.payload_bits, base.polar.n, base.polar.crc_bits, list_size)
    mean, second = compact_moments(base, energy)
    alphabet = base.num_patterns * (1 << code.payload_bits)
    cache, beam = {}, [()]
    calls = empty = branches = scored = 0
    started = time.perf_counter()

    def signal(word):
        if word not in cache:
            cache[word] = encode_words(base, energy, [word])[:, 0]
        return cache[word]

    for depth in range(k):
        h, extended = k - depth - 1, {}
        for prefix in beam:
            signals = np.array([signal(w) for w in prefix]).reshape(-1, base.n)
            residual = y - signals.sum(0)
            shift, variance = remaining_moments(mean, second, alphabet, signals, h, sampling)
            diagonal = sigma2 + variance if moment else np.full(base.n, sigma2)
            target = residual - shift if moment else residual
            for pattern, positions in enumerate(base.positions):
                llr = 2 * np.sqrt(energy[pattern]) * target[positions] / diagonal[positions]
                payloads, _ = crc_constrained_list(code, llr) if constrained_crc else code.decode_list(llr)
                calls += 1
                empty += not len(payloads)
                for payload in payloads:
                    word = pattern * (1 << code.payload_bits) + int(payload @ (1 << np.arange(code.payload_bits - 1, -1, -1)))
                    if sampling == "distinct" and word in prefix:
                        continue
                    key = tuple(sorted((*prefix, word)))
                    branches += 1
                    if key in extended:
                        continue
                    scored += 1
                    new_signals = np.array([signal(w) for w in key])
                    shift, variance = remaining_moments(mean, second, alphabet, new_signals, h, sampling)
                    score_diagonal = sigma2 + variance if moment else np.full(base.n, sigma2)
                    error = y - new_signals.sum(0) - (shift if moment else 0.)
                    score = -.5 * ((error ** 2 / score_diagonal).sum() + np.log(score_diagonal).sum())
                    if sampling == "iid":
                        score -= gammaln(np.unique(key, return_counts=True)[1] + 1).sum()
                    extended[key] = score
        beam = sorted(extended, key=lambda key: (-extended[key], key))[:width]
        if not beam:
            break
    candidates = np.array(sorted(set(w for key in beam for w in key)), dtype=int)
    return candidates, {"polar_calls": calls, "empty_crc_calls": empty, "generated_extensions": branches,
                        "lookahead_scores": scored,
                        "encoded_words": len(cache), "complete_states": len(beam), "candidate_count": len(candidates),
                        "seconds": time.perf_counter() - started}


def least_squares_gauge(gram):
    """Remove the least-squares additive component of all off-diagonal pair coefficients in O(L^2)."""
    size = len(gram)
    if size == 1:
        return np.zeros(1)
    if size == 2:
        return np.full(2, gram[0, 1] / 2)
    rows = gram.sum(1) - gram.diagonal()
    return (rows - rows.sum() / (2 * (size - 1))) / (size - 2)


def maximum_forest(gram):
    size, parents, edges = len(gram), list(range(len(gram))), []

    def root(node):
        while parents[node] != node:
            node = parents[node]
        return node

    for u, v in sorted(itertools.combinations(range(size), 2), key=lambda edge: (-abs(gram[edge]), edge)):
        a, b = root(u), root(v)
        if a != b:
            parents[a] = b
            edges.append((u, v))
    return edges


def posterior_tails(states, posterior, k):
    return np.array([posterior @ (states >= j) for j in range(1, k + 1)]).T


def decision_risk(states, posterior, action, k):
    return float(1 - posterior @ np.minimum(states, action).sum(1) / k)


def block_options(gram, linear, slots, block, sampling):
    """Exact ordered-slot conditional; factorial count weights arise on aggregation, not within this law."""
    size = len(gram)
    options = np.array(list(itertools.product(range(size), repeat=len(block))))
    fixed = np.delete(slots, block)
    if sampling == "distinct":
        allowed = ~np.isin(options, fixed).any(1)
        if len(block) == 2:
            allowed &= options[:, 0] != options[:, 1]
        options = options[allowed]
    scores = (linear[options] - .5 * gram.diagonal()[options]).sum(1)
    for word in fixed:
        scores -= gram[options, word].sum(1)
    if len(block) == 2:
        scores -= gram[options[:, 0], options[:, 1]]
    return options, softmax(scores)


def sample_tails(phi, y, k, sigma2, sampling, block_size, budget, seed, chains=4):
    """Finite Gibbs runs with a common conditional-score budget; no iid or mixing guarantee."""
    size = phi.shape[1]
    gram, linear = phi.T @ phi / sigma2, phi.T @ y / sigma2
    blocks = list(itertools.combinations(range(k), block_size))
    cells = size ** block_size if sampling == "iid" else math.perm(size - k + block_size, block_size)
    updates = budget // cells
    if updates < 4:
        raise ValueError("Budget must permit at least four updates per chain")
    seen_counts, chain_tails, sample_count = set(), [], 0
    for chain in range(chains):
        rng = np.random.default_rng(seed + chain)
        slots = rng.choice(size, k, replace=sampling == "iid")
        tails, saved = np.zeros((size, k)), 0
        for step in range(updates):
            block = blocks[step % len(blocks)]
            options, probabilities = block_options(gram, linear, slots, block, sampling)
            slots[list(block)] = options[rng.choice(len(options), p=probabilities)]
            if step >= updates // 4:
                words, counts = np.unique(slots, return_counts=True)
                for word, count in zip(words, counts):
                    tails[word, :count] += 1
                seen_counts.add(tuple(sorted(slots)))
                saved += 1
        sample_count += saved
        chain_tails.append(tails / saved)
    return np.mean(chain_tails, axis=0), {"score_cells": chains * updates * cells,
        "option_cells": chains * updates * size ** block_size, "samples": sample_count,
        "updates_per_chain": updates, "unique_counts": len(seen_counts),
        "restart_tail_spread": float(np.ptp(chain_tails, axis=0).max())}


def audit(seeds, budget, proposal_names=None):
    PolarPosteriorTests.setUpClass()
    fixture = PolarPosteriorTests
    base, energy, phi, k = fixture.base, fixture.energy, fixture.phi, 3
    settings = [("physical_R4_J16", 4, 16, False, False), ("moment_R4_J4", 4, 4, True, False),
                ("moment_R4_J16", 4, 16, True, False), ("moment_R16_J16", 16, 16, True, False),
                ("crc_physical_R4_J2", 4, 2, False, True), ("crc_moment_R4_J2", 4, 2, True, True),
                ("crc_physical_R16_J2", 16, 2, False, True), ("crc_physical_R4_J4", 4, 4, False, True)]
    settings = [("full_alphabet_control", 0, 0, False, False), *settings]
    if proposal_names is not None:
        if set(proposal_names) - {setting[0] for setting in settings}:
            raise ValueError("Unknown proposal setting")
        settings = [setting for setting in settings if setting[0] in proposal_names]
    rows = []
    for sampling, sigma2, seed in itertools.product(("iid", "distinct"), (.05, .3, 1.), range(seeds)):
        rng = np.random.default_rng(9100 + seed)
        truth = np.bincount(rng.choice(16, k, replace=sampling == "iid"), minlength=16)
        y = phi @ truth + sigma2 ** .5 * rng.normal(size=12)
        states, p = exact_posterior(phi, y, k, sigma2, sampling)
        tails = posterior_tails(states, p, k)
        optimal = tail_action(tails, k)
        best_risk = decision_risk(states, p, optimal, k)
        for name, width, list_size, moment, constrained in settings:
            if width:
                candidate, proposal = polar_beam(base, energy, y, k, sigma2, sampling, width, list_size, moment, constrained)
            else:
                candidate, proposal = np.arange(16), {"candidate_count": 16}
            row = {"sampling": sampling, "sigma2": sigma2, "seed": seed, "proposal": name,
                   "bayes_risk": best_risk, "proposal_work": proposal, "candidate": candidate.tolist()}
            if not len(candidate) or (sampling == "distinct" and len(candidate) < k):
                row.update({"failure": "no_complete_configuration", "omitted_mass": 1., "truth_omitted_copies": k,
                            "abstention_regret": 1 - best_risk})
                rows.append(row)
                continue
            outside = np.setdiff1d(np.arange(16), candidate)
            keep = states[:, outside].sum(1) == 0
            mass = p[keep].sum()
            local_states, conditional = states[keep][:, candidate], p[keep] / mass
            local_phi = encode_words(base, energy, candidate)
            np.testing.assert_array_equal(local_phi, phi[:, candidate])
            conditional_tails = posterior_tails(local_states, conditional, k)

            def full_regret(action):
                expanded = np.zeros(16, dtype=int)
                expanded[candidate] = action
                return max(0., decision_risk(states, p, expanded, k) - best_risk)

            row.update({"omitted_mass": float(max(0., 1 - mass)), "truth_omitted_copies": int(truth[outside].sum()),
                        "conditional_states": len(local_states),
                        "conditional_regret": full_regret(tail_action(conditional_tails, k)),
                        "candidate_action_lower_bound": full_regret(tail_action(tails[candidate], k))})
            means = conditional @ local_states
            if len(candidate) >= k:
                distinct_action = np.zeros(len(candidate), dtype=int)
                distinct_action[np.argsort(-means, kind="stable")[:k]] = 1
                row["mean_distinct_regret"] = full_regret(distinct_action)
                row["excess_copy_bound"] = float(conditional @ np.maximum(local_states - 1, 0).sum(1) / k)
                row["mean_distinct_conditional_regret"] = max(0., decision_risk(local_states, conditional, distinct_action, k)
                    - decision_risk(local_states, conditional, tail_action(conditional_tails, k), k))
            row["count_map_regret"] = full_regret(local_states[np.argmax(conditional)])
            gram = local_phi.T @ local_phi / sigma2
            row["inference"] = []
            for use_gauge, use_tree in itertools.product((False, True), repeat=2):
                gauge = least_squares_gauge(gram) if use_gauge else np.zeros(len(candidate))
                residual = gram - gauge[:, None] - gauge[None, :]
                edges = maximum_forest(residual) if use_tree else []
                qs, q, _ = enumerated_surrogate(local_phi, y, k, sigma2, edges, sampling, gauge)
                np.testing.assert_array_equal(qs, local_states)
                probabilities, _ = count_forest(local_phi, y, k, sigma2, edges, sampling, gauge)
                qt = np.flip(np.cumsum(np.flip(probabilities[:, 1:], axis=1), axis=1), axis=1)
                np.testing.assert_allclose(qt, posterior_tails(qs, q, k), atol=2e-11)
                missing = [edge for edge in itertools.combinations(range(len(candidate)), 2) if edge not in edges]
                gamma = max((abs(residual[edge]) for edge in missing), default=0.)
                row["inference"].append({"method": ("gauged_" if use_gauge else "raw_") + ("forest" if use_tree else "diagonal"),
                    "tv": float(abs(q - conditional).sum() / 2), "regret": full_regret(tail_action(qt, k)),
                    "tv_bound": math.tanh(math.comb(k, 2) * gamma / 2), "edges": len(edges)})
            for block_size in (1, 2):
                estimated, work = sample_tails(local_phi, y, k, sigma2, sampling, block_size, budget, 12000 + seed * 10)
                row["inference"].append({"method": f"gibbs_block{block_size}", "regret": full_regret(tail_action(estimated, k)),
                    "max_conditional_tail_error": float(abs(estimated - conditional_tails).max()), **work})
            rows.append(row)
        print(f"completed {sampling} sigma2={sigma2} seed={seed}", flush=True)
    return {"contract": {"B": 4, "M": 16, "P": 4, "V": 4, "n": 12, "Nc": 8, "CRC": 4, "K": k,
            "bank_seed": 3217, "observation_seed": "9100+seed", "seeds_per_group": seeds,
            "gibbs_score_budget_per_chain": budget, "chains": 4, "burn_fraction": .25,
            "proposal_names": [setting[0] for setting in settings],
            "status": "tiny conditional diagnostic; exhaustive reference and full-alphabet control are not scalable"}, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--proposals", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.seeds, args.budget, args.proposals)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
