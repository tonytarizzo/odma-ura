"""Bounded configuration search, exact-target chains and independent particle batches."""

import math

import numpy as np
from scipy.special import logsumexp

from framework.polar_decoder_core import _guide, allocate, count_tails
from framework.polar_decoder_queries import categorical, positive_integer, replicate_error, single_user, tilt_law


def top_indices(values, count):
    flat = values.ravel()
    count = min(count, int(np.isfinite(flat).sum()))
    if count == 0:
        raise ValueError("No legal configuration children")
    threshold = np.partition(flat, len(flat) - count)[len(flat) - count]
    chosen = np.r_[np.flatnonzero(flat > threshold), np.flatnonzero(flat == threshold)[:count - np.sum(flat > threshold)]]
    return chosen[np.lexsort((chosen, -flat[chosen]))]


def beam_states(s, width):
    positive_integer(width, "width")
    n, m = s.phi.shape
    mean = s.phi.mean(1)
    word_mean = mean @ s.phi
    tuples = np.empty((1, 0), dtype=np.int64)
    signals = np.zeros((1, n))
    log_prior = 0.; log_frontier = -np.inf; generated = 0
    for depth in range(s.k):
        residual = s.y - signals
        squared = np.maximum(0., np.square(residual).sum(1)[:, None] + 1 - 2 * (residual @ s.phi))
        remaining = s.k - depth - 1
        guide_square = squared - 2 * remaining * ((residual @ mean)[:, None] - word_mean) + remaining**2 * (mean @ mean)
        scores = -guide_square / (2 * (s.noise_var + remaining / n))
        upper = -np.maximum(0., np.sqrt(squared) - remaining)**2 / (2 * s.noise_var)
        if s.sampling == "distinct" and depth:
            scores[np.arange(len(tuples))[:, None], tuples] = -np.inf
            upper[np.arange(len(tuples))[:, None], tuples] = -np.inf
        generated += int(np.isfinite(scores).sum())
        selected = top_indices(scores, width)
        log_prior -= math.log(m - depth if s.sampling == "distinct" else m)
        upper.ravel()[selected] = -np.inf
        log_frontier = float(np.logaddexp(log_frontier, log_prior + logsumexp(upper)))
        parent, word = np.divmod(selected, m)
        tuples = np.column_stack((tuples[parent], word))
        signals = signals[parent] + s.phi[:, word].T
    likelihood = -np.square(s.y - signals).sum(1) / (2 * s.noise_var)
    return tuples, signals, likelihood + log_prior, log_frontier, generated


def configuration(s, *, width=32):
    if s.k == 1:
        return single_user(s)
    tuples, _, log_weight, frontier, generated = beam_states(s, width)
    evidence = float(logsumexp(log_weight))
    weights = np.exp(log_weight - evidence)
    tails = count_tails(tuples, weights, s.phi.shape[1])
    omitted = float(math.exp(frontier - np.logaddexp(frontier, evidence))) if np.isfinite(frontier) else 0.
    return allocate(tails, s.k, s.sampling), {
        "tails": tails, "target": "physical_posterior_on_retained_ordered_configurations", "width": width,
        "retained_configurations": len(tuples), "generated_children": generated, "omitted_mass_bound": omitted,
        "log_retained_evidence": evidence, "log_frontier_upper": float(frontier) if np.isfinite(frontier) else None,
        "excess_risk_bound": min(1., 2 * omitted), "certificate": "disjoint pruned-prefix frontier with unit-norm completion bound"}


def product_log_density(logp, words, distinct):
    if not distinct:
        return logp[words].sum(1)
    remaining = np.repeat(logp[None], len(words), axis=0)
    result = np.zeros(len(words))
    for j in range(words.shape[1]):
        result += remaining[np.arange(len(words)), words[:, j]] - logsumexp(remaining, axis=1)
        remaining[np.arange(len(words)), words[:, j]] = -np.inf
    return result


def product_draw(logp, count, k, distinct, defensive, rng):
    m = len(logp)
    prior = rng.random(count) < defensive
    logits = np.repeat(logp[None], count, axis=0)
    logits[prior] = 0.
    words = np.empty((count, k), dtype=np.int64)
    for j in range(k):
        words[:, j] = categorical(logits - logsumexp(logits, axis=1, keepdims=True), rng)
        if distinct:
            logits[np.arange(count), words[:, j]] = -np.inf
    return words


def mixture_log_density(logp, words, distinct, defensive):
    m, k = len(logp), words.shape[1]
    prior = -sum(math.log(m - j if distinct else m) for j in range(k))
    if defensive == 1:
        return np.full(len(words), prior)
    return np.logaddexp(math.log(defensive) + prior,
                       math.log1p(-defensive) + product_log_density(logp, words, distinct))


def chains(s, *, samples=128, burn_in=128, thinning=1, chains=4, global_probability=.1, defensive=.1, block_size=1, seed=0):
    for name, value in (("samples", samples), ("burn_in", burn_in), ("thinning", thinning), ("chains", chains)):
        positive_integer(value, name)
    if block_size not in (1, 2, 3) or not 0 < global_probability <= 1 or not 0 < defensive <= 1:
        raise ValueError("Invalid block/global/defensive schedule")
    if s.k == 1:
        return single_user(s)
    m, n = s.phi.shape[1], len(s.y)
    rng = np.random.default_rng(seed)
    logp, info = tilt_law(s, iterations=40)
    info.pop("elbo")  # This IID proposal objective is not a bound for the chain's count-tail approximation.
    distinct = s.sampling == "distinct"
    words = product_draw(logp, chains, s.k, distinct, defensive, rng)
    signals = s.phi[:, words].sum(2).T
    likelihood = -np.square(s.y - signals).sum(1) / (2 * s.noise_var)
    records = []; global_tried = global_accepted = local_changed = 0
    size = min(block_size, s.k)
    transitions = burn_in + samples * thinning
    for step in range(transitions):
        global_ids = np.flatnonzero(rng.random(chains) < global_probability)
        if len(global_ids):
            proposed = product_draw(logp, len(global_ids), s.k, distinct, defensive, rng)
            sums = s.phi[:, proposed].sum(2).T
            log_l = -np.square(s.y - sums).sum(1) / (2 * s.noise_var)
            ratio = (log_l - likelihood[global_ids] + mixture_log_density(logp, words[global_ids], distinct, defensive)
                     - mixture_log_density(logp, proposed, distinct, defensive))
            accept = np.log(rng.random(len(global_ids))) < np.minimum(0., ratio)
            ids = global_ids[accept]
            words[ids], signals[ids], likelihood[ids] = proposed[accept], sums[accept], log_l[accept]
            global_tried += len(global_ids); global_accepted += int(accept.sum())
        local_ids = np.setdiff1d(np.arange(chains), global_ids)
        if size == 1 and len(local_ids):
            slots = rng.integers(s.k, size=len(local_ids))
            old = words[local_ids, slots]
            residual = s.y - signals[local_ids] + s.phi[:, old].T
            score = residual @ s.phi / s.noise_var
            if distinct:
                for row, (chain, slot) in enumerate(zip(local_ids, slots)):
                    score[row, np.delete(words[chain], slot)] = -np.inf
            new = categorical(score - logsumexp(score, axis=1, keepdims=True), rng)
            local_changed += int(np.sum(new != old))
            signals[local_ids] += s.phi[:, new].T - s.phi[:, old].T
            words[local_ids, slots] = new
            likelihood[local_ids] = -np.square(s.y - signals[local_ids]).sum(1) / (2 * s.noise_var)
        elif len(local_ids):
            # Uniform legal block proposals are symmetric given the outside configuration.
            for chain in local_ids:
                slots = rng.choice(s.k, size=size, replace=False)
                outside = np.delete(words[chain], slots)
                allowed = np.setdiff1d(np.arange(m), outside) if distinct else np.arange(m)
                proposed = rng.choice(allowed, size=size, replace=not distinct)
                new_sum = signals[chain] - s.phi[:, words[chain, slots]].sum(1) + s.phi[:, proposed].sum(1)
                log_l = -np.square(s.y - new_sum).sum() / (2 * s.noise_var)
                if math.log(rng.random()) < min(0., log_l - likelihood[chain]):
                    local_changed += int(np.any(words[chain, slots] != proposed))
                    words[chain, slots], signals[chain], likelihood[chain] = proposed, new_sum, log_l
        if step >= burn_in and (step - burn_in + 1) % thinning == 0:
            records.append(words.copy())
    tuples = np.concatenate(records)
    tails = count_tails(tuples, np.full(len(tuples), 1 / len(tuples)), m)
    log_z_lower = -(np.linalg.norm(s.y) + s.k)**2 / (2 * s.noise_var)
    log_kappa = math.log(global_probability * defensive) + log_z_lower
    kappa = math.exp(log_kappa) if log_kappa > -745 else 0.
    mixing_failure = min(1., len(tuples) * math.exp(-kappa * min(burn_in + thinning, thinning)))
    return allocate(tails, s.k, s.sampling), {**info, "tails": tails, "target": "exact_invariant_posterior_chain",
        "recorded_states": len(tuples), "transitions": transitions * chains, "block_size": size,
        "global_attempts": global_tried, "global_accepts": global_accepted, "local_changes": local_changed,
        "mixing_failure_bound": mixing_failure, "log_minorization_lower": float(log_kappa), "excess_risk_bound": 1.,
        "certificate": "declared finite schedule; correlated states are not independent and mixing bound may be vacuous"}


def particles(s, *, batches=4, particles=64, defensive=.1, guide_log_floor=-80., failure_probability=.05, seed=0):
    positive_integer(batches, "batches"); positive_integer(particles, "particles")
    if not 0 < defensive <= 1 or guide_log_floor >= 0 or not 0 < failure_probability < 1:
        raise ValueError("Invalid particle proposal/guide/confidence parameters")
    n, m = s.phi.shape
    rng = np.random.default_rng(seed)
    mean = s.phi.mean(1)
    records, all_weights, batch_evidence = [], [], []
    for _ in range(batches):
        words = np.empty((particles, 0), dtype=np.int64)
        residual = np.repeat(s.y[None], particles, axis=0)
        old_guide, logw = np.zeros(particles), np.full(particles, -math.log(particles))
        evidence = 0.
        for j in range(s.k):
            parent = categorical(np.repeat(logw[None], particles, axis=0), rng)
            words, residual, old_guide = words[parent], residual[parent], old_guide[parent]
            guide = _guide(s, residual, words, s.k - j, mean)
            prior = np.full_like(guide, -math.log(m - j if s.sampling == "distinct" else m))
            if s.sampling == "distinct" and j:
                prior[np.arange(particles)[:, None], words] = -np.inf
            proposal = prior if defensive == 1 else np.logaddexp(math.log(defensive) + prior, math.log1p(-defensive) + guide)
            draw = categorical(proposal, rng)
            logq = proposal[np.arange(particles), draw]
            words = np.column_stack((words, draw))
            residual -= s.phi[:, draw].T
            h = s.k - j - 1
            variance = s.noise_var + h / n
            new_guide = -.5 * (np.square(residual - h * mean).sum(1) / variance + n * math.log(variance / s.noise_var))
            if h:
                new_guide = np.clip(new_guide, guide_log_floor, 0.)
            increment = -math.log(m - j if s.sampling == "distinct" else m) + new_guide - old_guide - logq
            total = logsumexp(increment)
            evidence += total - math.log(particles)
            logw, old_guide = increment - total, new_guide
        records.append(words); all_weights.append(evidence + logw); batch_evidence.append(evidence)
    tuples, log_weights = np.concatenate(records), np.concatenate(all_weights)
    weights = np.exp(log_weights - logsumexp(log_weights))
    tails = count_tails(tuples, weights, m)
    log_mean = float(logsumexp(batch_evidence) - math.log(batches))
    log_cap = -s.k * math.log(defensive) - (s.k - 1) * guide_log_floor
    error = replicate_error(log_mean, log_cap, batches, m * s.k, failure_probability)
    return allocate(tails, s.k, s.sampling), {
        "tails": tails, "target": "prefix_SMC_with_physical_terminal_weight", "batches": batches,
        "particles_per_batch": particles, "independent_confidence_units": batches, "log_mean_weight": log_mean,
        "batch_log_evidence": list(map(float, batch_evidence)), "terminal_unique_tuples": len(np.unique(tuples, axis=0)),
        "excess_risk_bound": min(1., 2 * error), "failure_probability": failure_probability,
        "certificate": "pooled unnormalized evidence and marks across independent particle batches"}
