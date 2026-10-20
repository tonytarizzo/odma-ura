"""Matched explicit-alphabet approximation screening; exhaustive labels are evaluation only."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp, softmax

from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.polar_energy_geometry import balanced_energy
from tests.polar_approximation_audit import encode_words, least_squares_gauge, posterior_tails, remaining_moments
from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import count_states, exact_posterior


def configuration_beam(phi, y, k, sigma2, sampling, width, moment):
    """Exact weights on retained complete count configurations; continuation ranking is only a heuristic."""
    n, m = phi.shape
    mean, second = phi.mean(1), np.square(phi).mean(1)
    beam, generated, scored, peak = [()], 0, 0, 0
    for depth in range(k):
        extended = set()
        for prefix in beam:
            for word in range(m):
                if sampling == "distinct" and word in prefix:
                    continue
                extended.add(tuple(sorted((*prefix, word))))
                generated += 1
        scores = {}
        for key in extended:
            signals = phi[:, key].T
            h = k - depth - 1
            shift, variance = remaining_moments(mean, second, m, signals, h, sampling)
            diagonal = sigma2 + variance if moment else np.full(n, sigma2)
            residual = y - signals.sum(0) - (shift if moment else 0.)
            score = -.5 * ((residual ** 2 / diagonal).sum() + np.log(diagonal).sum())
            if sampling == "iid":
                score -= gammaln(np.unique(key, return_counts=True)[1] + 1).sum()
            scores[key] = score
        scored += len(scores)
        peak = max(peak, len(scores))
        beam = sorted(scores, key=lambda key: (-scores[key], key))[:width]
    states = np.array([np.bincount(key, minlength=m) for key in beam])
    probabilities = softmax([scores[key] for key in beam])
    return posterior_tails(states, probabilities, k), states, {
        "generated_extensions": generated, "scored_prefixes": scored, "peak_extensions": peak,
        "retained_configurations": len(beam), "explicit_columns": m}


def log_convolve(left, right, k):
    return np.array([logsumexp(left[:h + 1] + right[h::-1]) for h in range(k + 1)])


def merged_recursion(phi, y, k, sigma2, sampling, spacing):
    """Sequential count recurrence with future projection, hashed grouping and explicit marked provenance."""
    n, m = phi.shape
    bank = {0: [(np.zeros(n), 0., np.zeros((m, k)), 0.)]}
    generated = merged = support_cells = peak = 0
    for word in range(m):
        future = phi[:, word + 1:]
        available = future.shape[1]
        mean = future.mean(1) if available else np.zeros(n)
        if available > 1:
            u, singular, _ = np.linalg.svd(future - mean[:, None], full_matrices=False)
            rank = int(np.sum(singular > 1e-12 * singular[0])) if singular[0] else 0
            projection = u[:, :rank] @ u[:, :rank].T
        else:
            projection = np.zeros((n, n))
        updated = {}
        for total, entries in bank.items():
            for signal, log_weight, markers, error in entries:
                for copies in range(min(1, k - total) + 1 if sampling == "distinct" else k - total + 1):
                    used, h = total + copies, k - total - copies
                    if (not available and h) or (sampling == "distinct" and h > available):
                        continue
                    generated += 1
                    raw_signal = signal + copies * phi[:, word]
                    anchor = y - h * mean
                    residual = anchor - raw_signal
                    projected = projection @ residual if h and not (sampling == "distinct" and h == available) else np.zeros(n)
                    representative = anchor - projected
                    weight = log_weight - gammaln(copies + 1) - np.square(residual - projected).sum() / (2 * sigma2)
                    marked = markers.copy()
                    marked[word, :copies] = 1.
                    key = tuple(np.rint(projected / (spacing * math.sqrt(sigma2))).astype(np.int64)) if spacing else tuple(projected)
                    groups = updated.setdefault(used, {})
                    if key not in groups:
                        groups[key] = [representative, weight, marked, error]
                        continue
                    old_signal, old_weight, old_marked, old_error = groups[key]
                    delta = representative - old_signal
                    scores = delta @ future
                    support_cells += available
                    if not h:
                        low = high = 0.
                    elif sampling == "iid":
                        low, high = h * scores.min(), h * scores.max()
                    else:
                        ordered = np.sort(scores)
                        low, high = ordered[:h].sum(), ordered[-h:].sum()
                    correction = (delta @ (y - old_signal) - delta @ delta / 2 - (low + high) / 2) / sigma2
                    adjusted_weight = weight + correction
                    combined = np.logaddexp(old_weight, adjusted_weight)
                    groups[key] = [old_signal, combined,
                        math.exp(old_weight - combined) * old_marked + math.exp(adjusted_weight - combined) * marked,
                        max(old_error, error + (high - low) / (2 * sigma2))]
                    merged += 1
        bank = {total: list(groups.values()) for total, groups in updated.items()}
        peak = max(peak, sum(map(len, bank.values())))
    entries = bank[k]
    log_weights = np.array([entry[1] for entry in entries])
    tails = sum(weight * entry[2] for weight, entry in zip(softmax(log_weights), entries))
    epsilon = max(entry[3] for entry in entries)
    return tails, {"generated_count_transitions": generated, "merged_states": merged, "peak_states": peak,
        "support_scores": support_cells, "max_accumulated_log_error": epsilon,
        "algebraic_tv_bound": math.tanh(epsilon / 2), "stored_tail_entries_at_peak": peak * m * k,
        "projection": "floating SVD, tolerance 1e-12; no outward-rounded certificate"}


def region_tails(gram, linear, groups, k, sampling):
    """Keep within-group count interactions and exact total K; all cross-group interactions are omitted."""
    tables, coefficients, local_states = [], [], 0
    for group in groups:
        counts, scores = [], []
        for h in range(min(k, len(group)) + 1 if sampling == "distinct" else k + 1):
            states = count_states(len(group), h, sampling)
            local = gram[np.ix_(group, group)]
            score = states @ linear[group] - np.einsum("bi,ij,bj->b", states, local, states) / 2
            if sampling == "iid":
                score -= gammaln(states + 1).sum(1)
            counts.append(states)
            scores.append(score)
        states, score = np.concatenate(counts), np.concatenate(scores)
        totals = states.sum(1)
        coeff = np.array([logsumexp(score[totals == h]) for h in range(k + 1)])
        tables.append((states, score, totals))
        coefficients.append(coeff)
        local_states += len(states)
    identity = np.r_[0., np.full(k, -np.inf)]
    forward, backward = [identity], [identity]
    for coeff in coefficients:
        forward.append(log_convolve(forward[-1], coeff, k))
    for coeff in coefficients[::-1]:
        backward.append(log_convolve(backward[-1], coeff, k))
    evidence = forward[-1][k]
    tails = np.zeros((len(gram), k))
    for j, group in enumerate(groups):
        outside = log_convolve(forward[j], backward[len(groups) - j - 1], k)
        states, score, totals = tables[j]
        p = np.exp(score + outside[k - totals] - evidence)
        tails[group] = posterior_tails(states, p, k)
    return tails, {"local_count_states": local_states, "groups": groups,
        "polynomial_convolution_terms": 3 * len(groups) * (k + 1) * (k + 2) // 2}


def select_regions(weight, size):
    """Greedily grow disjoint regions, charging all inspected coefficient entries."""
    available, groups, inspected = set(range(len(weight))), [], 0
    while available:
        candidates = sorted(available)
        scores = weight[np.ix_(candidates, candidates)].sum(1)
        inspected += len(candidates) ** 2
        group = [candidates[int(np.argmax(scores))]]
        available.remove(group[0])
        while len(group) < size and available:
            candidates = sorted(available)
            scores = weight[np.ix_(candidates, group)].sum(1)
            inspected += len(candidates) * len(group)
            chosen = candidates[int(np.argmax(scores))]
            group.append(chosen)
            available.remove(chosen)
        groups.append(sorted(group))
    return groups, inspected


def polynomial_product(left, right, k):
    result = np.zeros_like(left)
    for j in range(k + 1):
        result[:, j:] += left[:, j, None] * right[:, :k + 1 - j]
    return result


def connected_tails(gram, linear, k, sampling, maximum_order):
    """Graph-free marked cumulants; invalid signed output is returned explicitly, never clipped."""
    m = len(gram)
    logq = linear - gram.diagonal() / 2
    logq -= logsumexp(logq)
    identity = np.zeros((m, k + 1), dtype=np.longdouble)
    identity[:, 0] = 1
    moments, connected, visited = [identity], [np.zeros_like(identity)], 0
    for h in range(1, maximum_order + 1):
        states = count_states(m, h, sampling)
        score = states @ logq - (np.einsum("bi,ij,bj->b", states, gram, states) - states @ gram.diagonal()) / 2
        score += gammaln(h + 1) - gammaln(states + 1).sum(1)
        weights = np.exp(score.astype(np.longdouble))
        value = np.array([[weights[states[:, w] == j].sum() for j in range(k + 1)] for w in range(m)])
        cumulant = value.copy()
        for a in range(1, h):
            cumulant -= math.comb(h - 1, a - 1) * polynomial_product(connected[a], moments[h - a], k)
        moments.append(value)
        connected.append(cumulant)
        visited += len(states)
    if maximum_order == k:
        tails, status = checked_marginals(moments[k], k, sampling)
        return tails, {"status": status, "small_group_count_states": visited,
            "normalizer": float(moments[k][0].sum()), "exact_endpoint": True,
            "note": "Use the already computed positive full-order moment; avoid cancellation from inversion and reassembly"}
    partition = [identity]
    for h in range(1, k + 1):
        value = np.zeros_like(identity)
        for a in range(1, min(h, maximum_order) + 1):
            value += math.comb(h - 1, a - 1) * polynomial_product(connected[a], partition[h - a], k)
        partition.append(value)
    numerator = partition[k]
    tails, status = checked_marginals(numerator, k, sampling)
    return tails, {"status": status, "small_group_count_states": visited,
        "normalizer": float(numerator[0].sum()), "signed_construction": True}


def checked_marginals(numerator, k, sampling="iid"):
    normalizers = numerator.sum(1)
    if not np.all(np.isfinite(numerator)) or np.any(normalizers <= 0):
        return None, "nonpositive_or_nonfinite_evidence"
    q = numerator / normalizers[:, None]
    if np.max(abs(normalizers / normalizers[0] - 1)) > 1e-7:
        return None, "inconsistent_marked_evidence"
    if q.min() < -1e-9 or q.max() > 1 + 1e-9:
        return None, "negative_or_excess_marginal"
    if sampling == "distinct" and np.max(abs(q[:, 2:]), initial=0.) > 1e-9:
        return None, "distinct_source_violation"
    tails = np.flip(np.cumsum(np.flip(q[:, 1:], axis=1), axis=1), axis=1).astype(float)
    if abs(tails.sum() - k) > 1e-6:
        return None, "inconsistent_total_count"
    return tails, "valid_marginals"


def monomial_features(coordinates, degree):
    dimension, columns, signs = coordinates.shape[1], [], []
    for h in range(degree + 1):
        for indices in itertools.combinations_with_replacement(range(dimension), h):
            alpha = np.bincount(indices, minlength=dimension)
            denominator = math.sqrt(math.prod(math.factorial(int(a)) for a in alpha))
            columns.append(np.prod(coordinates ** alpha, axis=1) / denominator)
            signs.append((-1.) ** h)
    return np.stack(columns, axis=1), np.array(signs)


def feature_three_tails(q, features, signs, sampling, precision=0):
    """Exact K3 contraction for the supplied positive pair kernel U diag(signs) U.T, including exclusions."""
    if precision:
        import mpmath as mp

        with mp.workdps(precision):
            u, s, weights = mp.matrix(features.tolist()), mp.diag(signs.tolist()), list(map(mp.mpf, q))
            g = u.T * mp.diag(weights) * u
            def diagonal_product(left):
                return [sum(left[i, j] * u[i, j] for j in range(u.cols)) for i in range(u.rows)]

            diagonal, squares, cubes = diagonal_product(u * s), diagonal_product(u * s * g * s), diagonal_product(u * s * g * s * g * s)
            triple = [weights[i] ** 3 * diagonal[i] ** 3 for i in range(len(q))]
            repeated = [weights[i] ** 2 * diagonal[i] * squares[i] for i in range(len(q))]
            cube = [weights[i] * cubes[i] for i in range(len(q))]
            if sampling == "distinct":
                other = diagonal_product(u * s * (u.T * mp.diag([weights[i] ** 2 * diagonal[i] for i in range(len(q))]) * u) * s)
                cube = [cube[i] - 2 * repeated[i] - weights[i] * other[i] + 2 * triple[i] for i in range(len(q))]
                z = sum(cube)
                if not mp.isfinite(z) or z <= 0:
                    return None, "nonpositive_or_nonfinite_evidence"
                tails = [[3 * value / z, 0., 0.] for value in cube]
            else:
                z = sum(cube)
                if not mp.isfinite(z) or z <= 0:
                    return None, "nonpositive_or_nonfinite_evidence"
                tails = [[(3 * cube[i] - 3 * repeated[i] + triple[i]) / z,
                          (3 * repeated[i] - 2 * triple[i]) / z, triple[i] / z] for i in range(len(q))]
            tails = np.array(tails, dtype=float)
            probabilities = np.concatenate((1 - tails[:, :1], tails[:, :-1] - tails[:, 1:], tails[:, -1:]), axis=1)
            checked, status = checked_marginals(probabilities, 3, sampling)
            return checked, "high_precision_diagnostic" if checked is not None else status
    gram = features.T @ (q[:, None] * features)
    signed = signs[:, None] * gram * signs[None, :]
    diagonal = (features ** 2) @ signs
    square_sum = np.einsum("ir,rs,is->i", features, signed, features)
    cube_matrix = signed @ (gram * signs[None, :])
    cube_diagonal = q * np.einsum("ir,rs,is->i", features, cube_matrix, features)
    triple = q ** 3 * diagonal ** 3
    repeated_pair = q ** 2 * diagonal * square_sum
    normalizer = cube_diagonal.sum()
    if sampling == "distinct":
        marked_gram = features.T @ ((q ** 2 * diagonal)[:, None] * features)
        other_repeat = q * np.einsum("ir,rs,is->i", features,
                                    signs[:, None] * marked_gram * signs[None, :], features)
        cube_diagonal = cube_diagonal - 2 * repeated_pair - other_repeat + 2 * triple
        normalizer = cube_diagonal.sum()
        tails = np.zeros((len(q), 3))
        tails[:, 0] = 3 * cube_diagonal / normalizer
    else:
        exactly_two = 3 * (repeated_pair - triple) / normalizer
        exactly_three = triple / normalizer
        mean = 3 * cube_diagonal / normalizer
        tails = np.stack((mean - exactly_two - 2 * exactly_three, exactly_two + exactly_three, exactly_three), 1)
    if not np.isfinite(normalizer) or normalizer <= 0 or not np.all(np.isfinite(tails)):
        return None, "nonpositive_or_nonfinite_evidence"
    if tails.min() < -1e-8 or tails.max() > 1 + 1e-8 or np.max(np.diff(tails, axis=1)) > 1e-8:
        return None, "numerically_invalid_tails"
    if abs(tails.sum() - 3) > 1e-6:
        return None, "inconsistent_total_count"
    return tails, "valid_positive_model"


def feature_setup(phi, y, sigma2, dimensions, degree):
    """Centre exactly; a truncated coordinate projection and an even Taylor polynomial approximate pair terms only."""
    if degree % 2 or degree < 0:
        raise ValueError("Use a nonnegative even degree for a positive real pair kernel")
    centered = phi - phi.mean(1, keepdims=True)
    shifted = y - 3 * phi.mean(1)
    basis, singular, _ = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > 1e-12 * singular[0])) if singular[0] else 0
    dimension = min(rank, dimensions)
    coordinates = centered.T @ basis[:, :dimension] / math.sqrt(sigma2)
    features, signs = monomial_features(coordinates, degree)
    unary = (centered.T @ shifted - np.square(centered).sum(0) / 2) / sigma2
    q = softmax(unary)
    return q, features, signs, {"dimension": dimension, "exact_variation_rank": rank,
        "features": features.shape[1], "explicit_columns": phi.shape[1],
        "contraction_work_scale": phi.shape[1] * features.shape[1] ** 2 + features.shape[1] ** 3}


def frequency_numerators(phi, y, k, sampling, omega):
    """Per-frequency marked coefficients; real parts implement the antithetic Gaussian Fourier identity."""
    m = phi.shape[1]
    z, phase = np.exp(-1j * omega @ phi) / m, np.exp(1j * omega @ y)
    if sampling == "iid":
        total = z.sum(1)
        numerator = np.stack([math.comb(k, j) * z ** j * (total[:, None] - z) ** (k - j)
                              for j in range(k + 1)], axis=2)
        return (phase[:, None, None] * numerator).real
    coefficient = np.zeros((len(omega), k + 1), dtype=complex)
    coefficient[:, 0] = 1
    for word in range(m):
        for h in range(k, 0, -1):
            coefficient[:, h] += z[:, word] * coefficient[:, h - 1]
    excluded = np.ones_like(z)
    for h in range(1, k):
        excluded = coefficient[:, h, None] - z * excluded
    selected = z * excluded
    numerator = np.zeros((len(omega), m, k + 1), dtype=complex)
    numerator[:, :, 1] = selected
    numerator[:, :, 0] = coefficient[:, k, None] - selected
    numerator *= m ** k / math.comb(m, k)
    return (phase[:, None, None] * numerator).real


def compare(tails, exact, best_gain, k, meta, seconds):
    row = {**meta, "seconds": seconds}
    if tails is None:
        row.update({"failure": True, "regret": best_gain / k, "max_tail_error": None})
    else:
        action = tail_action(tails, k)
        gain = sum(exact[w, :copies].sum() for w, copies in enumerate(action))
        row.update({"failure": False, "regret": float(max(0., (best_gain - gain) / k)),
                    "max_tail_error": float(abs(tails - exact).max()), "action": action.tolist()})
    return row


def audit(ks, placements, seeds, sigmas, families, feature_precision=0):
    rows = []
    for placement in placements:
        base = ODMAPolarBaseline(4, 12, placement, prefix_bits=2, code_length=8, crc_bits=4)
        energy, balance = balanced_energy(base.positions, 12)
        phi = encode_words(base, energy, np.arange(16))
        for k, sampling, sigma2, seed in itertools.product(ks, ("iid", "distinct"), sigmas, seeds):
            rng = np.random.default_rng(9100 + seed)
            truth = np.bincount(rng.choice(16, k, replace=sampling == "iid"), minlength=16)
            y = phi @ truth + math.sqrt(sigma2) * rng.normal(size=12)
            states, p = exact_posterior(phi, y, k, sigma2, sampling)
            exact = posterior_tails(states, p, k)
            best = tail_action(exact, k)
            best_gain = sum(exact[w, :copies].sum() for w, copies in enumerate(best))
            gram, linear = phi.T @ phi / sigma2, phi.T @ y / sigma2
            gauge = least_squares_gauge(gram)
            gauged, adjusted = gram - gauge[:, None] - gauge[None, :], linear - k * gauge
            record = {"placement": placement, "K": k, "sampling": sampling, "sigma2": sigma2, "seed": seed,
                "M": 16, "exact_count_states": len(states), "bayes_risk": float(1 - best_gain / k),
                "balance": balance, "results": []}

            def save(name, tails, meta, started):
                record["results"].append(compare(tails, exact, best_gain, k, {"method": name, **meta}, time.perf_counter() - started))

            if "beam" in families:
                index = {tuple(state): mass for state, mass in zip(states, p)}
                for moment, width in itertools.product((False, True), (4, 16, 64, 256)):
                    started = time.perf_counter()
                    tails, retained, meta = configuration_beam(phi, y, k, sigma2, sampling, width, moment)
                    meta["omitted_configuration_mass"] = float(max(0., 1 - sum(index[tuple(s)] for s in retained)))
                    save(f"beam_{'moment' if moment else 'physical'}_{width}", tails, meta, started)
            if "regions" in families:
                proxy, proxy_work = region_tails(gauged, adjusted, [[w] for w in range(16)], k, sampling)
                variance = np.maximum(0., proxy @ (2 * np.arange(1, k + 1) - 1) - proxy.sum(1) ** 2)
                for selector, size in itertools.product(("geometry", "variance"), (1, 2, 4, 8, 16)):
                    started = time.perf_counter()
                    weight = abs(gauged).copy()
                    np.fill_diagonal(weight, 0.)
                    if selector == "variance":
                        weight *= np.sqrt(variance[:, None] * variance[None, :])
                    groups, inspected = select_regions(weight, size)
                    tails, meta = region_tails(gauged, adjusted, groups, k, sampling)
                    meta.update({"selector_entries_inspected": inspected,
                                 "selector_proxy_states": proxy_work["local_count_states"] if selector == "variance" else 0})
                    save(f"regions_{selector}_{size}", tails, meta, started)
            if "merge" in families:
                for spacing in (0., 1., 2., 4.):
                    started = time.perf_counter()
                    tails, meta = merged_recursion(phi, y, k, sigma2, sampling, spacing)
                    save(f"merge_{spacing:g}", tails, meta, started)
            if "connected" in families:
                for reference, order in itertools.product(("raw", "gauged"), range(1, k + 1)):
                    started = time.perf_counter()
                    g, l = (gram, linear) if reference == "raw" else (gauged, adjusted)
                    tails, meta = connected_tails(g, l, k, sampling, order)
                    save(f"connected_{reference}_{order}", tails, meta, started)
            if "features" in families and k == 3:
                for dimension, degree in itertools.product((1, 2, 4, 12), (2, 4)):
                    rank_bound = math.comb(dimension + degree, degree)
                    if rank_bound > 256:
                        record["results"].append({"method": f"features_d{dimension}_s{degree}",
                                                  "cost_gate": "more_than_256_features", "feature_upper_bound": rank_bound})
                        continue
                    started = time.perf_counter()
                    q, features, signs, meta = feature_setup(phi, y, sigma2, dimension, degree)
                    tails, status = feature_three_tails(q, features, signs, sampling, feature_precision)
                    save(f"features_d{dimension}_s{degree}", tails,
                         {**meta, "status": status, "precision_digits": feature_precision}, started)
                    kernel = (features * signs) @ features.T
                    if np.any(kernel <= 0):
                        raise FloatingPointError("Even-polynomial pair evaluation lost positivity")
                    log_kernel = np.log(kernel)
                    score = states @ np.log(q) + (np.einsum("bi,ij,bj->b", states, log_kernel, states)
                                                 - states @ log_kernel.diagonal()) / 2 - gammaln(states + 1).sum(1)
                    surrogate = posterior_tails(states, softmax(score), k)
                    label = compare(surrogate, exact, best_gain, k, {}, 0.)
                    record["results"][-1].update({"surrogate_reference_regret": label["regret"],
                        "surrogate_reference_tail_error": label["max_tail_error"],
                        "contraction_tail_disagreement": float(abs(tails - surrogate).max()) if tails is not None else None})
            if "spectral" in families:
                started = time.perf_counter()
                omega = np.random.default_rng(15000 + seed).normal(size=(4096, 12)) / math.sqrt(sigma2)
                cumulative = np.cumsum(frequency_numerators(phi, y, k, sampling, omega), axis=0)
                for frequencies in (64, 256, 1024, 4096):
                    numerator = cumulative[frequencies - 1] / frequencies
                    tails, status = checked_marginals(numerator, k, sampling)
                    save(f"spectral_{frequencies}", tails, {"status": status, "signed_construction": True,
                         "normalized_evidence_estimate": float(numerator[0].sum()),
                         "frequency_column_phases": frequencies * 16, "nested_batch_shared_seconds": True}, started)
            rows.append(record)
            print(f"bank={placement} K={k} {sampling} sigma2={sigma2} seed={seed}", flush=True)
    return {"contract": {"status": "explicit-alphabet branch screening, not scalable discovery", "B": 4,
        "n": 12, "Nc": 8, "P": 4, "CRC": 4, "families": families, "feature_precision": feature_precision,
        "invalid_output": "abstention, PUPE 1"}, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--placements", nargs="+", type=int, default=[3217, 3219])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[.05, .3, 1.])
    parser.add_argument("--families", nargs="+", default=["beam", "merge", "regions", "connected", "features", "spectral"])
    parser.add_argument("--feature-precision", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.ks, args.placements, args.seeds, args.sigmas, args.families, args.feature_precision)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
