"""Explicit positive and signed contraction variants, with declared support/state budgets."""

import itertools
import math

import numpy as np
from scipy.special import gammaln, logsumexp

from framework.polar_decoder_core import Scenario, allocate
from framework.polar_decoder_queries import interval_action, positive_integer, single_user, tilt_law
from framework.polar_decoder_search import beam_states, top_indices


def restrict(s, candidates):
    positive_integer(candidates, "candidates")
    m = s.phi.shape[1]
    size = min(m, max(candidates, s.k if s.sampling == "distinct" else 1))
    score = (s.y - (s.k - 1) * s.phi.mean(1)) @ s.phi
    ids = np.sort(top_indices(score, size))
    local = Scenario(s.phi[:, ids], s.y, s.k, s.noise_var, s.sampling)
    return local, ids, {"candidate_ids": ids.tolist(), "candidate_count": len(ids), "alphabet_size": m,
                        "support_restricted": len(ids) < m, "support_omitted_mass_bound": float(len(ids) < m),
                        "candidate_rule": "full_alphabet_prior_mean_residual_score"}


def expand(s, ids, local_tails, info):
    tails = np.zeros((s.phi.shape[1], s.k)); tails[ids] = local_tails
    return allocate(tails, s.k, s.sampling), {**info, "tails": tails}


def quantized(s, *, candidates=32, representatives=4, spacing=.25):
    positive_integer(representatives, "representatives")
    if spacing < 0 or not np.isfinite(spacing):
        raise ValueError("spacing must be finite and nonnegative")
    if s.k == 1:
        return single_user(s)
    local, ids, info = restrict(s, candidates)
    n, m = local.phi.shape
    bank = {0: [(np.zeros(n), 0., np.zeros((m, s.k)), 0.)]}
    generated = merged = capped = peak = 0
    for word in range(m):
        future = local.phi[:, word + 1:]
        available = future.shape[1]
        mean = future.mean(1) if available else np.zeros(n)
        if available > 1:
            vectors, singular, _ = np.linalg.svd(future - mean[:, None], full_matrices=False)
            rank = int(np.sum(singular > 1e-12 * singular[0])) if singular[0] else 0
            basis = vectors[:, :rank]
        else:
            basis = np.empty((n, 0))
        updated, keys = {}, {}
        for total, entries in bank.items():
            for signal, log_weight, marks, error in entries:
                maximum = min(1, s.k - total) if s.sampling == "distinct" else s.k - total
                for copies in range(maximum + 1):
                    used, h = total + copies, s.k - total - copies
                    if (not available and h) or (s.sampling == "distinct" and h > available):
                        continue
                    generated += 1
                    raw_signal = signal + copies * local.phi[:, word]
                    anchor = s.y - h * mean
                    residual = anchor - raw_signal
                    projected = (basis @ (basis.T @ residual) if h and not
                                 (s.sampling == "distinct" and h == available) else np.zeros(n))
                    representative = anchor - projected
                    weight = log_weight - gammaln(copies + 1) - np.square(residual - projected).sum() / (2 * s.noise_var)
                    marked = marks.copy(); marked[word, :copies] = 1.
                    key = (tuple(np.rint(projected / (spacing * math.sqrt(s.noise_var))).astype(np.int64))
                           if spacing else tuple(projected))
                    groups = updated.setdefault(used, [])
                    lookup = keys.setdefault(used, {})
                    if key in lookup:
                        target = lookup[key]
                    elif len(groups) < representatives:
                        lookup[key] = len(groups)
                        groups.append([representative, weight, marked, error])
                        continue
                    else:
                        target = int(np.argmin([np.square(representative - entry[0]).sum() for entry in groups]))
                        capped += 1
                    old_signal, old_weight, old_marked, old_error = groups[target]
                    delta = representative - old_signal
                    if h:
                        scores = delta @ future
                        if s.sampling == "iid":
                            low, high = h * scores.min(), h * scores.max()
                        else:
                            ordered = np.sort(scores)
                            low, high = ordered[:h].sum(), ordered[-h:].sum()
                    else:
                        low = high = 0.
                    correction = (delta @ (s.y - old_signal) - delta @ delta / 2 - (low + high) / 2) / s.noise_var
                    adjusted = weight + correction
                    combined = np.logaddexp(old_weight, adjusted)
                    groups[target] = [old_signal, combined,
                        math.exp(old_weight - combined) * old_marked + math.exp(adjusted - combined) * marked,
                        max(old_error, error + (high - low) / (2 * s.noise_var))]
                    merged += 1
        bank = updated
        peak = max(peak, sum(map(len, bank.values())))
    entries = bank[s.k]
    log_weights = np.array([entry[1] for entry in entries])
    tails = sum(p * entry[2] for p, entry in zip(np.exp(log_weights - logsumexp(log_weights)), entries))
    error = max(entry[3] for entry in entries)
    bound = min(1., 2 * (info["support_omitted_mass_bound"] + math.tanh(error / 2)))
    return expand(s, ids, tails, {**info, "target": "compatible_projected_count_states_with_positive_corrections",
        "generated_count_transitions": generated, "merged_states": merged, "forced_nearest_merges": capped,
        "peak_states": peak, "max_accumulated_log_error": float(error), "excess_risk_bound": bound,
        "certificate": "pathwise correction range; forward count marks retain identities; floating SVD projection"})


def local_states(size, k, distinct):
    rows = []
    choose = itertools.combinations if distinct else itertools.combinations_with_replacement
    for total in range(min(k, size) + 1 if distinct else k + 1):
        for words in choose(range(size), total):
            rows.append(np.bincount(words, minlength=size))
    return np.array(rows, dtype=np.int64)


def regions(s, *, region_size=1, max_local_states=20_000):
    positive_integer(region_size, "region_size"); positive_integer(max_local_states, "max_local_states")
    if s.k == 1:
        return single_user(s)
    n, m = s.phi.shape
    r = min(region_size, m)
    number = sum(math.comb(r, h) for h in range(min(r, s.k) + 1)) if s.sampling == "distinct" else math.comb(r + s.k, s.k)
    if number > max_local_states:
        raise ValueError(f"Region needs {number} local states, exceeding max_local_states")
    states = local_states(r, s.k, s.sampling == "distinct")
    occupancy = states.sum(1)
    groups = math.ceil(m / r)
    indices = np.arange(groups * r).reshape(groups, r)
    padded = indices >= m
    columns = s.phi[:, np.minimum(indices, m - 1)]
    grams = np.einsum("ngi,ngj->gij", columns, columns)
    linear = (s.y @ s.phi)[np.minimum(indices, m - 1)]
    scores = (linear @ states.T - .5 * np.einsum("si,gij,sj->gs", states, grams, states)) / s.noise_var
    scores -= gammaln(states + 1).sum(1)
    if padded.any():
        scores[-1, np.any(states[:, padded[-1]] != 0, axis=1)] = -np.inf
    extent = float(np.max(np.abs(linear)) / s.noise_var + s.k / s.noise_var + math.log(m * s.k + 1) + 30)
    lo, hi = -extent, extent
    for _ in range(64):
        fugacity = (lo + hi) / 2
        tilted = scores - fugacity * occupancy
        probability = np.exp(tilted - logsumexp(tilted, axis=1, keepdims=True))
        if np.sum(probability @ occupancy) > s.k:
            lo = fugacity
        else:
            hi = fugacity
    coefficients = np.stack([probability[:, occupancy == h].sum(1) for h in range(s.k + 1)], axis=1)
    identity = np.r_[1., np.zeros(s.k)]
    forward, backward = [identity], [identity]
    for coeff in coefficients:
        forward.append(np.convolve(forward[-1], coeff)[:s.k + 1])
    for coeff in coefficients[::-1]:
        backward.append(np.convolve(backward[-1], coeff)[:s.k + 1])
    evidence = forward[-1][s.k]
    if evidence <= 0 or not np.isfinite(evidence):
        raise FloatingPointError("Regional cardinality normalizer underflowed")
    tails = np.zeros((m, s.k))
    for group in range(groups):
        outside = np.convolve(forward[group], backward[groups - group - 1])[:s.k + 1]
        p = probability[group] * outside[s.k - occupancy] / evidence
        for index, word in enumerate(indices[group]):
            if word < m:
                tails[word] = [p[states[:, index] >= j].sum() for j in range(1, s.k + 1)]
    omitted_log_error = 0. if groups == 1 else s.k * (s.k - 1) / (2 * s.noise_var)
    return allocate(tails, s.k, s.sampling), {"tails": tails, "target": "exact_cardinality_positive_regional_model",
        "regions": groups, "region_size": r, "local_states": groups * len(states), "support_restricted": False,
        "excess_risk_bound": min(1., 2 * math.tanh(omitted_log_error / 2)),
        "certificate": "deleted cross-region edge range; all within-region states and total K retained",
        "cardinality_conditioning_probability": float(evidence)}


def log_polynomial_product(a, b, degree):
    result = np.full((len(a), degree + 1), -np.inf)
    for j in range(min(a.shape[1], degree + 1)):
        size = min(b.shape[1], degree + 1 - j)
        result[:, j:j + size] = np.logaddexp(result[:, j:j + size], a[:, j, None] + b[:, :size])
    return result


def connected(s, *, candidates=32, order=2, max_block_tuples=300_000):
    if s.sampling != "iid" or order not in (1, 2, 3):
        raise ValueError("Positive connected components support IID and orders 1, 2 or 3")
    positive_integer(max_block_tuples, "max_block_tuples")
    if s.k == 1:
        return single_user(s)
    local, ids, info = restrict(s, candidates)
    m, q = len(ids), min(order, s.k)
    if m**q > max_block_tuples:
        raise ValueError("Connected-block tuple budget exceeded; reduce candidates/order or increase the explicit cap")
    logp, optimization = tilt_law(local, iterations=40)
    optimization.pop("elbo")  # A restricted centering law is not the contraction's posterior approximation.
    center = local.phi @ np.exp(logp)
    centered = local.phi - center[:, None]
    pair = centered.T @ centered
    log_ratio = np.maximum(0., (pair.max() - pair) / s.noise_var)
    with np.errstate(divide="ignore"):
        logf = log_ratio + np.log(-np.expm1(-log_ratio))
    unary = (s.y - (s.k - 1) * center) @ local.phi / s.noise_var
    unary -= logsumexp(unary)
    coefficients = [None]
    visited = 0
    for size in range(1, q + 1):
        tuples = np.array(list(itertools.product(range(m), repeat=size)), dtype=np.int64)
        weight = unary[tuples].sum(1)
        if size == 2:
            weight += logf[tuples[:, 0], tuples[:, 1]]
        elif size == 3:
            a = logf[tuples[:, 0], tuples[:, 1]]
            b = logf[tuples[:, 0], tuples[:, 2]]
            c = logf[tuples[:, 1], tuples[:, 2]]
            weight += logsumexp(np.array([a + b, a + c, b + c, a + b + c]), axis=0)
        marked = np.full((m, size + 1), -np.inf)
        for word in range(m):
            counts = (tuples == word).sum(1)
            for count in range(size + 1):
                marked[word, count] = logsumexp(weight[counts == count])
        coefficients.append(marked); visited += len(tuples)
    polynomials = [np.zeros((m, 1))]
    for total in range(1, s.k + 1):
        polynomial = np.full((m, total + 1), -np.inf)
        for size in range(1, min(total, q) + 1):
            term = log_polynomial_product(coefficients[size], polynomials[total - size], total)
            polynomial = np.logaddexp(polynomial, math.log(math.comb(total - 1, size - 1)) + term)
        polynomials.append(polynomial)
    probability = np.exp(polynomials[-1] - logsumexp(polynomials[-1], axis=1, keepdims=True))
    tails = np.cumsum(probability[:, :0:-1], axis=1)[:, ::-1]
    exact = q == s.k and len(ids) == s.phi.shape[1]
    return expand(s, ids, tails, {**info, **optimization, "target": "positive_auxiliary_components_up_to_order_q",
        "order": q, "connected_block_tuples": visited, "excess_risk_bound": 0. if exact else 1.,
        "certificate": "full-order full-alphabet identity, otherwise trivial omitted-component/support bound",
        "reference": "fixed target-K centered unary and one global positive pair floor"})


def signed_rank(s, *, candidates=32, width=32, rank=8):
    positive_integer(width, "width"); positive_integer(rank, "rank")
    if s.k == 1:
        return single_user(s)
    local, ids, info = restrict(s, candidates)
    left_k = s.k // 2
    left = Scenario(local.phi, s.y * left_k / s.k, left_k, s.noise_var, s.sampling)
    right = Scenario(local.phi, s.y * (s.k - left_k) / s.k, s.k - left_k, s.noise_var, s.sampling)
    lw, ls, _, _, lc = beam_states(left, width)
    rw, rs, _, _, rc = beam_states(right, width)
    residual = s.y - ls[:, None] - rs[None]
    score = -np.square(residual).sum(2) / (2 * s.noise_var)
    legal = np.ones(score.shape, dtype=bool)
    if s.sampling == "distinct":
        legal = ~np.any(lw[:, None, :, None] == rw[None, :, None, :], axis=(2, 3))
    if not legal.any():
        # The declared split lists cover no legal global tuple; retain a legal observation-only incumbent.
        counts = np.zeros(s.phi.shape[1], dtype=np.int64)
        counts[ids[top_indices(s.y @ local.phi, s.k)]] = 1
        return counts, {**info, "target": "signed_split_list_contraction", "excess_risk_bound": 1.,
                        "empty_legal_split_list": True, "certificate": "no legal retained evidence; explicit legal incumbent"}
    log_prior = -sum(math.log(s.phi.shape[1] - j if s.sampling == "distinct" else s.phi.shape[1]) for j in range(s.k))
    shift = float(score[legal].max() + log_prior)
    table = np.zeros_like(score); table[legal] = np.exp(score[legal] + log_prior - shift)
    u, singular, v = np.linalg.svd(table, full_matrices=False)
    retained = min(rank, len(singular))
    approximation = (u[:, :retained] * singular[:retained]) @ v[:retained]
    approximation *= legal
    envelope = np.abs(table - approximation)
    evidence, approximate_evidence = float(table.sum()), float(approximation.sum())
    error_total = float(envelope.sum())
    lower, upper = np.zeros((len(ids), s.k)), np.ones((len(ids), s.k))
    for word in range(len(ids)):
        counts = (lw == word).sum(1)[:, None] + (rw == word).sum(1)[None]
        for j in range(1, s.k + 1):
            event = counts >= j
            numerator = float(approximation[event].sum()); error = float(envelope[event].sum())
            complement = approximate_evidence - numerator; other_error = max(0., error_total - error)
            lower[word, j - 1] = max(0., (numerator - error) / evidence, 1 - (complement + other_error) / evidence)
            upper[word, j - 1] = min(1., (numerator + error) / evidence, 1 - (complement - other_error) / evidence)
    counts, tails, conditional_bound = interval_action(local, lower, upper, s.y @ local.phi)
    prior_retained = min(1., int(legal.sum()) * math.exp(log_prior))
    log_missing = math.log1p(-prior_retained) if prior_retained < 1 else -np.inf
    log_evidence = shift + math.log(evidence)
    omitted = math.exp(log_missing - np.logaddexp(log_missing, log_evidence)) if np.isfinite(log_missing) else 0.
    result = np.zeros(s.phi.shape[1], dtype=np.int64); result[ids] = counts
    full_tails = np.zeros((s.phi.shape[1], s.k)); full_tails[ids] = tails
    return result, {**info, "tails": full_tails, "target": "signed_split_list_SVD_with_marked_absolute_envelope",
        "tail_estimate": "conditional_simultaneous_lower_bounds", "rank": retained, "table_shape": list(table.shape),
        "negative_approximate_entries": int(np.sum(approximation < 0)), "generated_half_children": lc + rc,
        "relative_envelope": error_total / evidence, "list_omitted_mass_bound": omitted,
        "excess_risk_bound": min(1., 2 * omitted + conditional_bound),
        "certificate": "actual entrywise signed residual and event/complement contractions; omitted list prior charged",
        "construction_caveat": "builds the full retained cut table and SVD; no construction speedup claimed"}
