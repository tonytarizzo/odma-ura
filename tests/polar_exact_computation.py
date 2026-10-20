"""Finite exact-computation audit for Report 7; deliberately not a benchmark receiver."""

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp, softmax

from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode
from tests.polar_posterior_test import PolarPosteriorTests, binary_rank, count_states, exact_posterior


def certified_axes(positions, generator):
    """Axes in the affine variation span, proved using supports and positive amplitudes."""
    groups, inverse = np.unique(generator, axis=0, return_inverse=True)
    if np.flatnonzero(~generator.any(1)).tolist() != [0]:
        raise ValueError("This structural certificate assumes the selected code's sole constant symbol is local index zero")
    supports = [set(p[inverse == j]) for p in positions for j, g in enumerate(groups) if g.any()]
    known = set()
    while True:
        extra = set.union(set(), *(s - known for s in supports if len(s - known) == 1))
        if not extra:
            break
        known |= extra
    if any(p[0] in known for p in positions):
        known.update(positions[:, 0].tolist())
    while True:
        extra = set.union(set(), *(s - known for s in supports if len(s - known) == 1))
        if not extra:
            return known
        known |= extra


def frontier_widths(positions, order, n):
    incidence = np.zeros((len(positions), n), dtype=int)
    for p, support in enumerate(positions):
        incidence[p, support] = 1
    remaining, seen, widths = incidence.sum(0), np.zeros(n, dtype=bool), []
    for p in order:
        remaining -= incidence[p]
        seen |= incidence[p].astype(bool)
        widths.append(int(np.count_nonzero(seen & (remaining > 0))))
    return widths


def greedy_frontier_order(positions, n):
    incidence = np.zeros((len(positions), n), dtype=int)
    for p, support in enumerate(positions):
        incidence[p, support] = 1
    remaining, seen = incidence.sum(0), np.zeros(n, dtype=bool)
    available, order = list(range(len(positions))), []
    while available:
        p = min(available, key=lambda p: (np.count_nonzero((seen | (incidence[p] > 0))
                                                         & (remaining - incidence[p] > 0)), p))
        available.remove(p)
        order.append(p)
        remaining -= incidence[p]
        seen |= incidence[p].astype(bool)
    return order


def dyadic_matrix(phi):
    """Exact integer keys for the supplied binary floating point matrix; no tolerance-based merging."""
    ratios = [[float(x).as_integer_ratio() for x in row] for row in phi]
    denominator = max(d for row in ratios for _, d in row)
    integers = np.array([[a * (denominator // d) for a, d in row] for row in ratios], dtype=object)
    return integers, denominator


def frontier_decode(phi, groups, y, k, sigma2, sampling, order):
    """Resource elimination, then forward/backward count queries. Exponential finite reference only."""
    integers, denominator = dyadic_matrix(phi)
    n, m = phi.shape
    supports = [np.flatnonzero(np.any(phi[:, group] != 0, axis=1)) for group in groups]
    last = np.full(n, -1)
    for stage, p in enumerate(order):
        last[supports[p]] = stage
    local_banks = []
    for p in order:
        group, bank = groups[p], []
        for count in range(min(k, len(group)) + 1 if sampling == "distinct" else k + 1):
            merged = {}
            for a in count_states(len(group), count, sampling):
                signal = tuple(integers[:, group] @ a)
                weight = 1. if sampling == "distinct" else math.exp(-gammaln(a + 1).sum())
                if signal not in merged:
                    merged[signal] = [0., np.zeros((len(group), k + 1))]
                merged[signal][0] += weight
                merged[signal][1][np.arange(len(group)), a] += weight
            bank.extend((count, signal, weight, laws) for signal, (weight, laws) in merged.items())
        local_banks.append(bank)
    start = (0, ())
    initial = math.exp(-np.square(y[last < 0]).sum() / (2 * sigma2))
    forward, transitions = [{start: initial}], []
    old_boundary, seen, widths = np.array([], dtype=int), set(), []
    for stage, p in enumerate(order):
        seen.update(supports[p].tolist())
        boundary = np.array(sorted(r for r in seen if last[r] > stage), dtype=int)
        finished = np.flatnonzero(last == stage)
        future_capacity = sum(len(groups[t]) for t in order[stage + 1:]) if sampling == "distinct" else k
        if stage == len(order) - 1:
            future_capacity = 0
        updated, edges = defaultdict(float), []
        for before, mass in forward[-1].items():
            q, old_signal = before
            past = np.zeros(n, dtype=object)
            past[old_boundary] = old_signal
            for option, (count, local_signal, weight, _) in enumerate(local_banks[stage]):
                total = q + count
                if total > k or k - total > future_capacity:
                    continue
                signal = past + np.asarray(local_signal, dtype=object)
                residual = y[finished] - np.array([float(signal[r] / denominator) for r in finished])
                gaussian = math.exp(-residual @ residual / (2 * sigma2))
                after = (total, tuple(signal[boundary]))
                updated[after] += mass * weight * gaussian
                edges.append((before, after, option, gaussian))
        forward.append(dict(updated))
        transitions.append(edges)
        old_boundary = boundary
        widths.append(len(boundary))
    terminal = (k, ())
    evidence = forward[-1][terminal]
    backward, probabilities = {terminal: 1.}, np.zeros((m, k + 1))
    for stage in range(len(order) - 1, -1, -1):
        previous = defaultdict(float)
        for before, after, option, gaussian in transitions[stage]:
            _, _, weight, laws = local_banks[stage][option]
            outside = gaussian * backward.get(after, 0.)
            previous[before] += weight * outside
            probabilities[groups[order[stage]]] += forward[stage][before] * outside * laws
        backward = previous
    probabilities /= evidence
    tails = np.flip(np.cumsum(np.flip(probabilities[:, 1:], axis=1), axis=1), axis=1)
    action = tail_action(tails, k)
    return action, probabilities, evidence, {"states": [len(f) for f in forward], "frontier_widths": widths,
                                           "transitions": sum(len(t) for t in transitions),
                                           "backward_evidence": initial * backward[start]}


def tail_action(tails, k):
    action = np.zeros(len(tails), dtype=int)
    for _ in range(k):
        values = np.array([tails[w, action[w]] if action[w] < tails.shape[1] else -np.inf for w in range(len(tails))])
        action[np.argmax(values)] += 1
    return action


def completion_bound(phi, residual, h, sampling, mode, iterations=8):
    ball = max(0., np.linalg.norm(residual) - h) ** 2
    if mode == "ball" or h == 0:
        return ball, 0
    if sampling == "iid":
        low, high = h * phi.min(1), h * phi.max(1)
    else:
        ordered = np.sort(phi, axis=1)
        low, high = ordered[:, :h].sum(1), ordered[:, -h:].sum(1)
    bound = max(ball, np.square(np.maximum(low - residual, 0) + np.maximum(residual - high, 0)).sum())
    if mode == "box":
        return bound, 0
    if mode != "dual":
        raise ValueError(mode)
    point = h * phi.mean(1)
    calls = 0
    for _ in range(iterations):
        dual = residual - point
        scores = dual @ phi
        calls += 1
        vertex = h * phi[:, np.argmax(scores)] if sampling == "iid" else phi[:, np.argsort(scores)[-h:]].sum(1)
        bound = max(bound, 2 * dual @ residual - dual @ dual - 2 * dual @ vertex)
        direction = vertex - point
        norm2 = direction @ direction
        if norm2 == 0:
            break
        step = np.clip(dual @ direction / norm2, 0, 1)
        point += step * direction
    return bound, calls


def event_ranges(prefix, h, m, k, sampling):
    d = len(prefix)
    low, high = np.zeros(m, dtype=int), np.zeros(m, dtype=int)
    low[:d], high[:d] = prefix, prefix
    remaining = m - d
    if remaining:
        cap = h if sampling == "iid" else 1
        low[d:] = max(0, h - (remaining - 1) * cap)
        high[d:] = min(h, cap)
    thresholds = np.arange(1, k + 1)
    return (low[:, None] >= thresholds).ravel(), (high[:, None] >= thresholds).ravel()


def certificate_search(phi, y, k, sigma2, sampling, mode, certificate, iterations=8, bounder=None, negative_probability=None):
    """Best-upper-mass count-tree search. Floating bounds are checked against exhaustive reference in tests."""
    n, m = phi.shape
    seen = np.zeros(m * k)
    frontier = {}
    evidence, leaves, nodes, support_calls, score_evaluations, next_id = 0., 0, 0, 0, 0, 0
    if bounder is not None:
        support_calls, score_evaluations = bounder.setup_calls, bounder.setup_scores
    total_states = math.comb(m + k - 1, k) if sampling == "iid" else math.comb(m, k)

    def add(prefix, residual, h, log_beta):
        nonlocal evidence, leaves, nodes, support_calls, score_evaluations, next_id
        nodes += 1
        remaining = m - len(prefix)
        if h < 0 or (sampling == "distinct" and h > remaining) or (remaining == 0 and h):
            return
        forced = h == 0 or remaining == 1 or (sampling == "distinct" and h == remaining)
        if forced:
            suffix = np.zeros(remaining, dtype=int)
            if h:
                suffix[:] = 1 if sampling == "distinct" else h
            state = np.r_[prefix, suffix].astype(int)
            r = residual - phi[:, len(prefix):] @ suffix
            log_weight = log_beta - gammaln(suffix + 1).sum() - r @ r / (2 * sigma2)
            weight = math.exp(log_weight)
            evidence += weight
            seen[:] += weight * (state[:, None] >= np.arange(1, k + 1)).ravel()
            leaves += 1
            return
        log_mass = log_beta + (h * math.log(remaining) - math.lgamma(h + 1) if sampling == "iid"
                               else math.log(math.comb(remaining, h)))
        distance, calls = (completion_bound(phi[:, len(prefix):], residual, h, sampling, mode, iterations)
                           if bounder is None else bounder(len(prefix), residual, h))
        support_calls += calls
        score_evaluations += calls * remaining
        upper = math.exp(log_mass - distance / (2 * sigma2))
        guaranteed, possible = event_ranges(prefix, h, m, k, sampling)
        negative = None if negative_probability is None else negative_probability(prefix, h, m, k, sampling)
        frontier[next_id] = (prefix, residual, h, log_beta, upper, guaranteed, possible, negative)
        next_id += 1

    add((), y, k, 0.)
    while True:
        action = tail_action(seen.reshape(m, k), k)
        selected = np.array([w * k + action[w] - 1 for w in range(m) if action[w] > 0])
        next_copy = np.array([w * k + action[w] for w in range(m) if action[w] < k])
        if certificate == "interval":
            upper = seen.copy()
            for node in frontier.values():
                upper += node[4] * node[6]
            certified = seen[selected].min() >= upper[next_copy].max()
        elif certificate == "difference":
            difference = seen[selected, None] - seen[None, next_copy]
            for node in frontier.values():
                # For different messages, opposite count inequalities are jointly feasible iff individually feasible.
                negative = ((~node[5][selected, None]) & node[6][None, next_copy] if node[7] is None
                            else node[7][selected[:, None], next_copy[None, :]])
                difference -= node[4] * negative
            same_word = selected[:, None] // k == next_copy[None, :] // k
            difference[same_word] = 0.
            certified = bool(np.all(difference >= 0))
        else:
            raise ValueError(certificate)
        if certified:
            return action, {"leaves": leaves, "total_leaves": total_states, "nodes": nodes,
                            "frontier": len(frontier), "support_calls": support_calls, "score_evaluations": score_evaluations,
                            "evaluated_evidence": evidence, "remaining_upper": sum(node[4] for node in frontier.values())}
        key = max(frontier, key=lambda key: (frontier[key][4], -key))
        prefix, residual, h, log_beta, _, _, _, _ = frontier.pop(key)
        column = phi[:, len(prefix)]
        for count in range(min(h, 1) + 1 if sampling == "distinct" else h + 1):
            add(prefix + (count,), residual - count * column, h - count, log_beta - math.lgamma(count + 1))


def audit_banks():
    generator = PolarCode(6, 32, 8).encode(np.eye(6, dtype=np.uint8)).T
    rows = []
    for bits in (12, 14):
        for seed in (3201, 3202, 3203):
            positions = ODMAPolarBaseline(bits, 256, seed, prefix_bits=bits - 6, code_length=32, crc_bits=8).positions
            cuts = []
            for size in (4, 8, 12, 16):
                future = certified_axes(positions[size:], generator)
                occupancy = np.bincount(positions[:size].ravel(), minlength=256)
                ranks = [binary_rank(generator[(occupancy[p] == 1) & np.isin(p, list(future))]) for p in positions[:size]]
                cuts.append({"processed_patterns": size, "future_axes": len(future), "private_ranks": ranks,
                             "log2_reachable_lower_bound": sum(ranks)})
            natural = frontier_widths(positions, range(len(positions)), 256)
            greedy = frontier_widths(positions, greedy_frontier_order(positions, 256), 256)
            rows.append({"B": bits, "seed": seed, "cuts": cuts, "natural_peak_frontier": max(natural),
                         "greedy_peak_frontier": max(greedy)})
    return rows


def audit_tiny():
    PolarPosteriorTests.setUpClass()
    f = PolarPosteriorTests
    groups = [list(range(4 * p, 4 * p + 4)) for p in range(4)]
    schedule_rows, search_rows, connected_rows = [], [], []
    for sampling in ("iid", "distinct"):
        states, posterior = exact_posterior(f.phi, f.y, 3, f.sigma2, sampling)
        expected = np.array([posterior @ (states == j) for j in range(4)]).T
        beta = np.ones(len(states)) if sampling == "distinct" else np.exp(-gammaln(states + 1).sum(1))
        evidence = beta @ np.exp(-np.square(f.y - states @ f.phi.T).sum(1) / (2 * f.sigma2))
        for order in itertools.permutations(range(4)):
            _, probabilities, z, meta = frontier_decode(f.phi, groups, f.y, 3, f.sigma2, sampling, order)
            np.testing.assert_allclose(probabilities, expected, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose([z, meta["backward_evidence"]], evidence, rtol=1e-12)
            schedule_rows.append({"sampling": sampling, "order": order, **meta})
        for sigma2 in (.05, .3, 1.):
            for seed in range(4):
                rng = np.random.default_rng(9100 + seed)
                truth = rng.choice(16, 3, replace=sampling == "iid")
                y = f.phi[:, truth].sum(1) + math.sqrt(sigma2) * rng.normal(size=12)
                order = np.argsort(-(f.phi.T @ y), kind="stable")
                phi = f.phi[:, order]
                states, posterior = exact_posterior(phi, y, 3, sigma2, sampling)
                tails = np.array([posterior @ (states >= j) for j in range(1, 4)]).T
                best = tail_action(tails, 3)
                optimal_gain = (tails * (np.arange(3) < best[:, None])).sum()
                nu = softmax(phi.T @ y / sigma2)
                pair = np.exp(-(phi.T @ phi) / sigma2)
                if sampling == "distinct":
                    np.fill_diagonal(pair, 0)
                interaction = pair - 1
                absolute = abs(interaction)
                epsilon, alpha = absolute.max(), (absolute @ nu).max()
                delta = 3 * alpha ** 2 * (1 + epsilon)
                partial_tails = []
                for w in range(16):
                    c1 = np.array([1 - nu[w], nu[w]])
                    degree = (np.arange(16) == w).astype(int)
                    c2 = np.bincount((degree[:, None] + degree[None, :]).ravel(),
                                     weights=(nu[:, None] * nu[None, :] * interaction).ravel(), minlength=3)
                    polynomial = np.convolve(np.convolve(c1, c1), c1) + 3 * np.convolve(c1, c2)
                    partial_tails.append(np.flip(np.cumsum(np.flip(polynomial[1:]))))
                partial_tails = np.asarray(partial_tails)
                proposed = tail_action(np.maximum(0, partial_tails - delta), 3)
                selected = [(w, proposed[w] - 1) for w in range(16) if proposed[w]]
                next_copy = [(w, proposed[w]) for w in range(16) if proposed[w] < 3]
                # A common absolute graph majorant bounds a signed indicator contrast by delta, not 2*delta.
                certified = all(partial_tails[a] - partial_tails[b] >= delta
                                for a in selected for b in next_copy if a[0] != b[0])
                beta = np.zeros(len(states)) if sampling == "distinct" else -gammaln(states + 1).sum(1)
                log_z = logsumexp(beta - np.square(y - states @ phi.T).sum(1) / (2 * sigma2))
                log_b3 = log_z + math.lgamma(4) + (y @ y + 3) / (2 * sigma2) - 3 * logsumexp(phi.T @ y / sigma2)
                connected_rows.append({"sampling": sampling, "sigma2": sigma2, "seed": seed,
                                       "delta_over_exact_normalizer": delta * math.exp(-log_b3), "certified": certified})
                if certified:
                    gain = (tails * (np.arange(3) < proposed[:, None])).sum()
                    if abs(gain - optimal_gain) > 1e-10:
                        raise AssertionError("Connected certificate disagrees with exhaustive Bayes inference")
                for mode in ("ball", "box", "dual"):
                    for certificate in ("interval", "difference"):
                        action, meta = certificate_search(phi, y, 3, sigma2, sampling, mode, certificate)
                        gain = (tails * (np.arange(3) < action[:, None])).sum()
                        if abs(optimal_gain - gain) > 1e-10:
                            raise AssertionError("A reported certificate disagrees with exhaustive Bayes inference")
                        search_rows.append({"sampling": sampling, "sigma2": sigma2, "seed": seed, "bound": mode,
                                            "certificate": certificate, **meta})
    return {"schedules": schedule_rows, "search": search_rows, "connected": connected_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("banks", "tiny", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    if args.part in ("banks", "all"):
        result["banks"] = audit_banks()
    if args.part in ("tiny", "all"):
        result["tiny"] = audit_tiny()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"Saved finite exact-computation audit to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
