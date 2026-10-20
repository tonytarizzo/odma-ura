"""Count-query, prefix, Fourier and auxiliary-field variants from Report 10.

All alphabet traversals are explicit. Certificates are real-arithmetic bounds;
floating-point evaluation is not an interval-arithmetic implementation.
"""

import math

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import binom, chi2

from framework.polar_decoder_core import allocate, count_tails


def positive_integer(value, name):
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def categorical(logp, rng):
    cdf = np.cumsum(np.exp(logp), axis=-1)
    cdf[..., -1] = 1.
    return np.sum(cdf <= rng.random(cdf.shape[:-1] + (1,)), axis=-1)


def single_user(s):
    score = s.y @ s.phi / s.noise_var
    tails = np.exp(score - logsumexp(score))[:, None]
    return allocate(tails, 1, s.sampling), {"tails": tails, "excess_risk_bound": 0., "special_case": "exact_K1"}


def replicate_error(log_mean, log_cap, samples, queries, failure):
    log_b = log_cap + .5 * math.log(math.log(2 * (queries + 1) / failure) / (2 * samples))
    if log_b >= log_mean:
        return 1.
    ratio = math.exp(log_b - log_mean)
    return min(1., 2 * ratio / (1 - ratio))


def geometry(phi):
    mean = phi.mean(1)
    centered = phi - mean[:, None]
    values, vectors = np.linalg.eigh(centered @ centered.T / phi.shape[1])
    if values.min() < -1e-10:
        raise FloatingPointError("Nonpositive codebook covariance")
    return {"phi": phi, "mean": mean, "values": np.maximum(values, 0.), "vectors": vectors, "centered": vectors.T @ centered}


def tilt_law(s, iterations=100, tolerance=1e-9, multiplier=None):
    coefficient = s.k - 1 if multiplier is None else multiplier
    m = s.phi.shape[1]

    def fun(theta):
        score = theta @ s.phi
        logz = logsumexp(score)
        mean = s.phi @ np.exp(score - logz)
        return (s.noise_var * (theta @ theta) / 2 + coefficient * (logz - math.log(m)) - s.y @ theta,
                s.noise_var * theta + coefficient * mean - s.y)

    result = minimize(fun, s.y / (s.noise_var + coefficient / len(s.y)), jac=True, method="L-BFGS-B",
                      options={"maxiter": iterations, "gtol": tolerance, "ftol": tolerance})
    scores = result.x @ s.phi
    logp = scores - logsumexp(scores)
    p = np.exp(logp)
    mean = s.phi @ p
    elbo = -(np.square(s.y - s.k * mean).sum() + s.k * (1 - mean @ mean)) / (2 * s.noise_var)
    elbo -= s.k * np.dot(p, logp + math.log(m))
    return logp, {"elbo": float(elbo), "optimizer_converged": bool(result.success), "iterations": int(result.nit)}


def gaussian_counts(s, *, geometry_cache=None):
    if s.k == 1:
        return single_user(s)
    n, m = s.phi.shape
    if s.sampling == "distinct" and s.k == m:
        tails = np.zeros((m, s.k)); tails[:, 0] = 1.
        return np.ones(m, dtype=np.int64), {"tails": tails, "excess_risk_bound": 0., "special_case": "known_full_distinct_alphabet"}
    if m == 1:
        return np.array([s.k]), {"tails": np.ones((1, s.k)), "excess_risk_bound": 0.}
    geo = geometry_cache if geometry_cache is not None else geometry(s.phi)
    if geo["phi"] is not s.phi:
        raise ValueError("Moment cache belongs to another encoder")
    p, delta = 1 / m, geo["centered"]
    shifted = geo["vectors"].T @ (s.y - s.k * geo["mean"])
    log_mass = np.full((m, s.k + 1), -np.inf)
    js = range(s.k + 1) if s.sampling == "iid" else (0, 1)
    for j in js:
        h = s.k - j
        # The distinct closure matches the finite-population covariance, not the IID covariance.
        scale = h if s.sampling == "iid" else (h * (m - 1 - h) / (m - 2) if m > 2 else 0.)
        diagonal = s.noise_var + scale * geo["values"] / (1 - p)
        correction = scale * p / (1 - p)**2
        inverse = 1 / diagonal
        denominator = 1 - correction * np.sum(delta**2 * inverse[:, None], axis=0)
        if np.any(denominator <= 0):
            raise FloatingPointError("Excluded covariance downdate lost positive definiteness")
        residual = shifted[:, None] - (j - h * p / (1 - p)) * delta
        cross = np.sum(residual * delta * inverse[:, None], axis=0)
        quadratic = np.sum(residual**2 * inverse[:, None], axis=0) + correction * cross**2 / denominator
        logdet = np.log(diagonal / s.noise_var).sum() + np.log(denominator)
        prior = (gammaln(s.k + 1) - gammaln(j + 1) - gammaln(h + 1) + j * math.log(p) + h * math.log1p(-p)
                 if s.sampling == "iid" else math.log(s.k / m if j else 1 - s.k / m))
        log_mass[:, j] = prior - .5 * (quadratic + logdet)
    probability = np.exp(log_mass - logsumexp(log_mass, axis=1, keepdims=True))
    tails = np.cumsum(probability[:, :0:-1], axis=1)[:, ::-1]
    return allocate(tails, s.k, s.sampling), {
        "tails": tails, "target": "excluded_word_Gaussian_count_queries", "excess_risk_bound": 1.,
        "normalizer_log_range": float(np.ptp(logsumexp(log_mass, axis=1))),
        "certificate": "trivial modeling-error bound; separate word normalizers do not define a joint posterior",
        "source_closure": "IID" if s.sampling == "iid" else "finite_population_mean_and_covariance"}


def prefix(s, *, splits=128, tolerance=1e-8, max_iterations=100):
    positive_integer(splits, "splits")
    if s.sampling != "iid":
        raise ValueError("This prefix variant uses the IID product query provider")
    if s.phi.shape[1] == 1:
        return np.array([s.k]), {"excess_risk_bound": 0., "special_case": "singleton_alphabet"}
    logp, info = tilt_law(s, max_iterations)
    p, m = np.exp(logp), s.phi.shape[1]
    cumulative = np.r_[0., np.cumsum(p)]
    cells, visited = [(0, m)], 1
    leaves = set()

    def query(cell):
        mass = max(0., min(1., cumulative[cell[1]] - cumulative[cell[0]]))
        return binom.sf(np.arange(s.k), s.k, mass)

    def split(index):
        nonlocal visited
        lo, hi = cells.pop(index)
        mid = (lo + hi) // 2
        cells.extend(((lo, mid), (mid, hi)))
        for a, b in cells[-2:]:
            if b - a == 1:
                leaves.add(a)
        visited += 2

    best = int(np.argmax(p))
    while best not in leaves:
        split(next(i for i, (a, b) in enumerate(cells) if a <= best < b))
    refinements = 0
    while True:
        upper = np.array([query(c) for c in cells])
        optimistic = allocate(upper, s.k)
        upper_gain = sum(upper[i, :h].sum() for i, h in enumerate(optimistic))
        ids = np.array(sorted(leaves))
        leaf_tails = binom.sf(np.arange(s.k)[None], s.k, p[ids, None])
        local = allocate(leaf_tails, s.k)
        lower_gain = sum(leaf_tails[i, :h].sum() for i, h in enumerate(local))
        gap = max(0., float((upper_gain - lower_gain) / s.k))
        choices = [i for i, (a, b) in enumerate(cells) if b - a > 1 and optimistic[i]]
        if not choices or gap <= tolerance or refinements >= splits:
            break
        chosen = max(choices, key=lambda i: (upper[i, :optimistic[i]].sum(), -cells[i][0]))
        split(chosen)
        refinements += 1
    counts = np.zeros(m, dtype=np.int64); counts[ids] = local
    var_error = min(1., math.sqrt(max(0., -info["elbo"]) / 2))
    return counts, {**info, "target": "product_provider_with_message_prefix_readout", "visited_prefixes": visited,
                    "resolved_words": len(leaves), "product_decision_gap": gap,
                    "excess_risk_bound": min(1., gap + 2 * var_error),
                    "certificate": "prefix gain gap plus product-law Pinsker; explicit alphabet partition/tilt cost"}


def interval_action(s, lower, upper, priority=None):
    lower = np.maximum.accumulate(np.clip(lower, 0., 1.)[:, ::-1], axis=1)[:, ::-1]
    upper = np.minimum.accumulate(np.clip(upper, 0., 1.), axis=1)
    if np.any(lower > upper + 1e-9):
        raise FloatingPointError("Inconsistent count-tail intervals")
    upper = np.maximum(upper, lower)
    counts = np.zeros(len(lower), dtype=np.int64)
    priority = np.zeros(len(lower)) if priority is None else priority
    for _ in range(s.k):
        scores = lower[np.arange(len(lower)), np.minimum(counts, s.k - 1)].copy()
        scores[counts >= (1 if s.sampling == "distinct" else s.k)] = -np.inf
        choices = np.flatnonzero(scores == scores.max())
        word = choices[np.argmax(priority[choices])]
        counts[word] += 1
    optimistic = allocate(upper, s.k, s.sampling)
    lo = sum(lower[w, :c].sum() for w, c in enumerate(counts))
    hi = sum(upper[w, :c].sum() for w, c in enumerate(optimistic))
    return counts, lower, min(1., max(0., float((hi - lo) / s.k)))


def fourier(s, *, frequencies=512, batch_size=64, failure_probability=.05, seed=0):
    positive_integer(frequencies, "frequencies"); positive_integer(batch_size, "batch_size")
    if s.sampling != "iid" or not 0 < failure_probability < 1:
        raise ValueError("Fourier powers require IID and a failure probability in (0,1)")
    if s.k == 1:
        return single_user(s)
    m = s.phi.shape[1]
    if m == 1:
        return np.array([s.k]), {"tails": np.ones((1, s.k)), "excess_risk_bound": 0., "special_case": "singleton_alphabet"}
    rng = np.random.default_rng(seed)
    masses = np.zeros((m, s.k + 1)); evidence = 0.
    coefficients = np.exp(binom.logpmf(np.arange(s.k + 1), s.k, 1 / m))
    for start in range(0, frequencies, batch_size):
        size = min(batch_size, frequencies - start)
        omega = rng.normal(size=(size, len(s.y))) / math.sqrt(s.noise_var)
        phase = np.exp(-1j * (omega @ s.phi))
        chi = phase.mean(1)
        outside = (m * chi[:, None] - phase) / (m - 1)
        observation = np.exp(1j * (omega @ s.y))
        evidence += np.real(observation * chi**s.k).sum()
        for j in range(s.k + 1):
            masses[:, j] += coefficients[j] * np.real(observation[:, None] * phase**j * outside**(s.k - j)).sum(0)
    masses /= frequencies; evidence /= frequencies
    numerator = np.cumsum(masses[:, :0:-1], axis=1)[:, ::-1]
    complement = evidence - numerator
    error = math.sqrt(2 * math.log(2 * (2 * m * s.k + 1) / failure_probability) / frequencies)
    nl, nu = np.maximum(0., numerator - error), np.maximum(0., numerator + error)
    cl, cu = np.maximum(0., complement - error), np.maximum(0., complement + error)
    lower = np.divide(nl, nl + cu, out=np.zeros_like(nl), where=nl + cu > 0)
    upper = np.divide(nu, nu + cl, out=np.ones_like(nu), where=nu + cl > 0)
    counts, lower, bound = interval_action(s, lower, upper, s.y @ s.phi)
    return counts, {"tails": lower, "target": "signed_continuous_Fourier_intervals", "frequencies": frequencies,
                    "tail_upper": upper, "max_word_evidence_discrepancy": float(np.max(np.abs(masses.sum(1) - evidence))),
                    "evidence_estimate": float(evidence), "absolute_query_error_bound": error,
                    "positive_evidence_certified": bool(evidence > error), "tail_estimate": "simultaneous_lower_bounds",
                    "excess_risk_bound": bound, "failure_probability": failure_probability,
                    "uninformative_tie_rule": "one_user_score; no normalization of negative mass"}


def fields(s, *, samples=512, field_rank=2, defensive=.1, failure_probability=.05, seed=0, geometry_cache=None):
    positive_integer(samples, "samples")
    if not isinstance(field_rank, (int, np.integer)) or field_rank < 0 or not 0 < defensive <= 1:
        raise ValueError("Require a nonnegative field rank and defensive fraction in (0,1]")
    if s.sampling != "iid" or not 0 < failure_probability < 1:
        raise ValueError("This field proposal supports IID and a failure probability in (0,1)")
    if s.k == 1:
        return single_user(s)
    n, m = s.phi.shape
    rank = min(field_rank, n)
    geo = geometry_cache if geometry_cache is not None else geometry(s.phi)
    if geo["phi"] is not s.phi:
        raise ValueError("Moment cache belongs to another encoder")
    projection = geo["vectors"][:, -rank:].T @ s.phi if rank else np.empty((0, m))
    contrast = helmert(s.k, full=False).T
    dimension = (s.k - 1) * rank
    shift_bound = s.k / math.sqrt(s.noise_var)
    radius = shift_bound + math.sqrt(dimension) + math.sqrt(2 * math.log(2 / failure_probability))
    log_domain_mass = float(chi2.logcdf(radius**2, dimension)) if dimension else 0.
    omitted = float(chi2.sf((radius - shift_bound)**2, dimension)) if dimension else 0.
    rng = np.random.default_rng(seed)
    log_guide, info = tilt_law(s, iterations=40)
    info.pop("elbo")
    tuples, log_weights = [], []
    attempts = 0
    for start in range(0, samples, 32):
        size = min(32, samples - start)
        u = rng.normal(size=(size, s.k - 1, rank)); attempts += size
        if dimension:
            bad = np.linalg.norm(u.reshape(size, -1), axis=1) > radius
            while bad.any():
                attempts += int(bad.sum()); u[bad] = rng.normal(size=(int(bad.sum()), s.k - 1, rank))
                bad = np.linalg.norm(u.reshape(size, -1), axis=1) > radius
        field = np.einsum("ki,sir->skr", contrast, u) * math.sqrt(s.k / s.noise_var)
        words = np.empty((size, s.k), dtype=np.int64)
        logq = np.zeros(size)
        for j in range(s.k):
            score = log_guide[None] + field[:, j] @ projection
            score -= logsumexp(score, axis=1, keepdims=True)
            proposal = (np.full_like(score, -math.log(m)) if defensive == 1 else
                        np.logaddexp(math.log(defensive) - math.log(m), math.log1p(-defensive) + score))
            words[:, j] = categorical(proposal, rng)
            logq += proposal[np.arange(size), words[:, j]]
        signals = s.phi[:, words].sum(2).T
        selected = projection[:, words].transpose(1, 2, 0)
        center = np.einsum("ki,skr->sir", contrast, selected) * math.sqrt(s.k / s.noise_var)
        # Omega(u,W)=pi(W) g(y-s_W) phi(u-B^T x_W), avoiding cancellation of large D and c_y terms.
        log_ratio = np.sum(u * center - center**2 / 2, axis=(1, 2))
        weight = (-np.square(s.y - signals).sum(1) / (2 * s.noise_var) - s.k * math.log(m)
                  + log_ratio + log_domain_mass - logq)
        tuples.append(words); log_weights.append(weight)
    tuples, log_weights = np.concatenate(tuples), np.concatenate(log_weights)
    weights = np.exp(log_weights - logsumexp(log_weights))
    tails = count_tails(tuples, weights, m)
    log_cap = log_domain_mass - s.k * math.log(defensive)
    if dimension:
        log_cap += radius * shift_bound - shift_bound**2 / 2
    log_mean = float(logsumexp(log_weights) - math.log(samples))
    error = replicate_error(log_mean, log_cap, samples, m * s.k, failure_probability / 2)
    return allocate(tails, s.k), {**info, "tails": tails, "target": "exact_joint_field_target_restricted_to_ball",
                                 "field_rank": rank, "field_dimension": dimension, "field_draw_attempts": attempts,
                                 "samples": samples, "effective_sample_size": float(1 / (weights @ weights)),
                                 "field_omitted_mass_bound": omitted, "log_mean_weight": log_mean,
                                 "excess_risk_bound": min(1., 2 * (omitted + (1 - omitted) * error)),
                                 "failure_probability": failure_probability / 2,
                                 "certificate": "bounded joint importance weights plus posterior Gaussian-ball remainder"}
