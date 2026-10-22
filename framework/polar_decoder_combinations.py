"""IID combination receivers: multiset discovery, corrected mixtures and terminal posterior moves.

All likelihoods use physical AWGN. Product/Gaussian closures guide discovery only.
Ordered importance samples and unordered beam states have explicitly different priors.
"""

import math

import numpy as np
from scipy.special import gammaln, logsumexp

from framework.polar_decoder_core import _guide, allocate, count_tails
from framework.polar_decoder_queries import categorical, positive_integer, tilt_law
from framework.polar_decoder_search import top_indices


def count_log_orbit(words):
    """log(t! / prod c_w!) for sorted rows of t message IDs."""
    result = np.full(len(words), gammaln(words.shape[1] + 1))
    repeats = np.ones(len(words), dtype=int)
    for j in range(1, words.shape[1]):
        repeats = np.where(words[:, j] == words[:, j - 1], repeats + 1, 1)
        result -= np.log(repeats)
    return result


def multiset_states(s, width=32):
    positive_integer(width, 'width')
    if s.sampling != 'iid':
        raise ValueError('Combination receivers currently require IID messages')
    n, m = s.phi.shape
    words = np.empty((1, 0), dtype=np.int64)
    signals = np.zeros((1, n))
    log_prior = np.zeros(1)
    mean = s.phi.mean(1)
    generated = 0
    for depth in range(s.k):
        h = s.k - depth - 1
        residual = s.y - signals - h * mean
        squared = np.maximum(0., np.square(residual).sum(1)[:, None] + 1 - 2 * residual @ s.phi)
        scores = -squared / (2 * (s.noise_var + h / n)) + log_prior[:, None] + math.log((depth + 1) / m)
        for row, prefix in enumerate(words):
            ids, counts = np.unique(prefix, return_counts=True)
            scores[row, ids] -= np.log(counts + 1)
        # Each parent's children are distinct multisets, so its top width suffices for the global top width.
        children = np.stack([top_indices(score, min(width, m)) for score in scores])
        parent = np.repeat(np.arange(len(words)), children.shape[1])
        child = children.ravel()
        candidates = np.sort(np.column_stack((words[parent], child)), axis=1)
        unique, index = np.unique(candidates, axis=0, return_index=True)
        priorities = scores[parent[index], child[index]]
        chosen = top_indices(priorities, min(width, len(unique)))
        words = unique[chosen]
        signals = signals[parent[index[chosen]]] + s.phi[:, child[index[chosen]]].T
        log_prior = count_log_orbit(words) - (depth + 1) * math.log(m)
        generated += scores.size
    log_weight = log_prior - np.square(s.y - signals).sum(1) / (2 * s.noise_var)
    return words, log_weight, generated


def multiset_beam(s, *, width=32):
    words, log_weight, generated = multiset_states(s, width)
    evidence = float(logsumexp(log_weight))
    tails = count_tails(words, np.exp(log_weight - evidence), s.phi.shape[1])
    return allocate(tails, s.k), {'tails': tails, 'retained_multisets': len(words), 'generated_children': generated,
                                'log_retained_evidence': evidence, 'excess_risk_bound': 1.,
                                'target': 'physical_count_posterior_restricted_to_unique_multisets',
                                'certificate': 'full multinomial prior on retained multisets; omitted mass uncertified'}


def proposal_setup(s, *, defensive=.1, product_weight=.2, beam_weight=.2, width=32, max_iterations=100):
    if s.sampling != 'iid':
        raise ValueError('Combination receivers currently require IID messages')
    positive_integer(max_iterations, 'max_iterations')
    weights = np.array([defensive, 1 - defensive - product_weight - beam_weight, product_weight, beam_weight])
    if not np.isfinite(weights).all() or defensive <= 0 or np.any(weights < 0):
        raise ValueError('Require positive defensive weight and nonnegative mixture weights summing to one')
    logp, info = tilt_law(s, max_iterations) if product_weight else (None, {})
    info.pop('elbo', None)
    beam, probabilities, lookup = None, None, {}
    if beam_weight:
        beam, logw, generated = multiset_states(s, width)
        logprob = logw - logsumexp(logw)
        probabilities = np.exp(logprob)
        # Draw a multiset, then uniformly permute it: q(ordered tuple) = q(counts) / orbit size.
        lookup = {tuple(row):float(p - orbit) for row, p, orbit in zip(beam, logprob, count_log_orbit(beam))}
        info.update(beam_multisets=len(beam), beam_generated_children=generated)
    return {'weights': weights, 'logp': logp, 'beam': beam, 'beam_probabilities': probabilities,
            'beam_lookup': lookup, 'mean': s.phi.mean(1), 'info': info}


def mixture_density(s, words, proposal, log_guide=None):
    if log_guide is None:
        residual = np.repeat(s.y[None], len(words), axis=0)
        log_guide = np.zeros(len(words))
        for j in range(s.k):
            logp = _guide(s, residual, words[:, :j], s.k - j, proposal['mean'])
            log_guide += logp[np.arange(len(words)), words[:, j]]
            residual -= s.phi[:, words[:, j]].T
    weights = proposal['weights']
    terms = [np.full(len(words), math.log(weights[0]) - s.k * math.log(s.phi.shape[1]))]
    if weights[1]:
        terms.append(math.log(weights[1]) + log_guide)
    if weights[2]:
        terms.append(math.log(weights[2]) + proposal['logp'][words].sum(1))
    if weights[3]:
        beam_q = np.array([proposal['beam_lookup'].get(tuple(row), -np.inf) for row in np.sort(words, axis=1)])
        terms.append(math.log(weights[3]) + beam_q)
    return logsumexp(terms, axis=0)


def mixture_draw(s, count, proposal, rng):
    component = rng.choice(4, size=count, p=proposal['weights'])
    words = np.empty((count, s.k), dtype=np.int64)
    beam_rows = np.flatnonzero(component == 3)
    if len(beam_rows):
        chosen = rng.choice(len(proposal['beam']), size=len(beam_rows), p=proposal['beam_probabilities'])
        words[beam_rows] = rng.permuted(proposal['beam'][chosen], axis=1)
    residual = np.repeat(s.y[None], count, axis=0)
    log_guide = np.zeros(count)
    for j in range(s.k):
        guide = _guide(s, residual, words[:, :j], s.k - j, proposal['mean'])
        for which in (0, 1, 2):
            rows = np.flatnonzero(component == which)
            if not len(rows):
                continue
            if which == 0:
                words[rows, j] = rng.integers(s.phi.shape[1], size=len(rows))
            else:
                logits = guide[rows] if which == 1 else np.broadcast_to(proposal['logp'], (len(rows), s.phi.shape[1]))
                words[rows, j] = categorical(logits, rng)
        log_guide += guide[np.arange(count), words[:, j]]
        residual -= s.phi[:, words[:, j]].T
    return words, mixture_density(s, words, proposal, log_guide)


def block_proposal(s, residual, size, rng=None, words=None):
    """Normalized ordered block proposal given the outside configuration; evaluate or sample."""
    if s.sampling != 'iid':
        raise ValueError('Block proposals currently require IID messages')
    evaluate = words is not None
    if not evaluate:
        words = np.empty((len(residual), size), dtype=np.int64)
    residual = residual.copy()
    logq = np.zeros(len(residual))
    mean = s.phi.mean(1)
    for j in range(size):
        logits = _guide(s, residual, words[:, :j], size - j, mean)
        if not evaluate:
            words[:, j] = categorical(logits, rng)
        logq += logits[np.arange(len(words)), words[:, j]]
        residual -= s.phi[:, words[:, j]].T
    return words, logq


def terminal_moves(s, words, rng, moves=8, block_size=1, batch_size=128):
    if s.sampling != 'iid':
        raise ValueError('Terminal kernels currently require IID messages')
    if not isinstance(moves, (int, np.integer)) or moves < 0 or block_size not in (1, 2, 3):
        raise ValueError('Require nonnegative moves and block_size in 1,2,3')
    positive_integer(batch_size, 'batch_size')
    words = words.copy()
    changed = 0
    size = min(block_size, s.k)
    for start in range(0, len(words), batch_size):
        current = words[start:start + batch_size]
        signal = s.phi[:, current].sum(2).T
        rows = np.arange(len(current))
        for _ in range(moves):
            slots = np.argsort(rng.random((len(current), s.k)), axis=1)[:, :size]
            old = current[rows[:, None], slots].copy()
            outside_residual = s.y - signal + s.phi[:, old].sum(2).T
            if size == 1:
                logits = outside_residual @ s.phi / s.noise_var
                proposed = categorical(logits - logsumexp(logits, axis=1, keepdims=True), rng)[:, None]
                accept = np.ones(len(current), dtype=bool)
            else:
                proposed, log_new = block_proposal(s, outside_residual, size, rng=rng)
                _, log_old = block_proposal(s, outside_residual, size, words=old)
                new_residual = outside_residual - s.phi[:, proposed].sum(2).T
                old_residual = s.y - signal
                ratio = (np.square(old_residual).sum(1) - np.square(new_residual).sum(1)) / (2 * s.noise_var)
                ratio += log_old - log_new
                accept = np.log(rng.random(len(current))) < np.minimum(0., ratio)
            changed += int(np.sum(accept & np.any(proposed != old, axis=1)))
            ids = rows[accept]
            signal[ids] += s.phi[:, proposed[accept]].sum(2).T - s.phi[:, old[accept]].sum(2).T
            current[ids[:, None], slots[accept]] = proposed[accept]
    return words, {'terminal_moves_per_sample': moves, 'terminal_block_size': size,
                   'terminal_changed_blocks': changed, 'terminal_attempts': len(words) * moves}


def combined_importance(s, *, samples=1024, defensive=.1, product_weight=.2, beam_weight=.2, width=32,
                        max_iterations=100, batch_size=128, moves=0, block_size=1, seed=0):
    for key, value in [('samples', samples), ('batch_size', batch_size)]:
        positive_integer(value, key)
    proposal = proposal_setup(s, defensive=defensive, product_weight=product_weight, beam_weight=beam_weight,
                              width=width, max_iterations=max_iterations)
    rng = np.random.default_rng(seed)
    records, log_weights = [], []
    log_prior = -s.k * math.log(s.phi.shape[1])
    for start in range(0, samples, batch_size):
        words, logq = mixture_draw(s, min(batch_size, samples - start), proposal, rng)
        residual = s.y - s.phi[:, words].sum(2).T
        records.append(words)
        log_weights.append(log_prior - np.square(residual).sum(1) / (2 * s.noise_var) - logq)
    words, logw = np.concatenate(records), np.concatenate(log_weights)
    weights = np.exp(logw - logsumexp(logw))
    words, info = terminal_moves(s, words, rng, moves, block_size, batch_size)
    # Invariant moves transport each weighted atom. Reweighting moved states by their old proposal would be wrong.
    tails = count_tails(words, weights, s.phi.shape[1])
    return allocate(tails, s.k), {**proposal['info'], **info, 'tails': tails, 'samples': samples,
                                'mixture_weights_prior_guide_product_beam': proposal['weights'].tolist(),
                                'effective_sample_size': float(1 / np.square(weights).sum()),
                                'unique_multisets': len(np.unique(np.sort(words, axis=1), axis=0)),
                                'log_mean_weight': float(logsumexp(logw) - math.log(samples)),
                                'target': 'physical_posterior_mixture_IS_then_invariant_transport', 'excess_risk_bound': 1.,
                                'certificate': 'self-normalized finite-sample estimate; evidence recorded before terminal moves'}


def combined_particles(s, *, batches=4, particles=64, defensive=.1, product_weight=.25, max_iterations=100,
                       guide_log_floor=-1000., moves=0, block_size=1, batch_size=128, seed=0):
    if s.sampling != 'iid':
        raise ValueError('Combination receivers currently require IID messages')
    for key, value in [('batches', batches), ('particles', particles), ('max_iterations', max_iterations)]:
        positive_integer(value, key)
    if not 0 < defensive <= 1 or not 0 <= product_weight <= 1 - defensive or not np.isfinite(guide_log_floor) or guide_log_floor >= 0:
        raise ValueError('Invalid SMC proposal weights or continuation floor')
    rng = np.random.default_rng(seed)
    logp, info = tilt_law(s, max_iterations) if product_weight else (None, {})
    info.pop('elbo', None)
    n, m = s.phi.shape
    mean = s.phi.mean(1)
    records, all_weights, evidences = [], [], []
    guide_weight = 1 - defensive - product_weight
    for _ in range(batches):
        words = np.empty((particles, 0), dtype=np.int64)
        residual = np.repeat(s.y[None], particles, axis=0)
        old_guide, logw = np.zeros(particles), np.full(particles, -math.log(particles))
        evidence = 0.
        for j in range(s.k):
            parent = categorical(np.repeat(logw[None], particles, axis=0), rng)
            words, residual, old_guide = words[parent], residual[parent], old_guide[parent]
            proposal = np.full((particles, m), math.log(defensive) - math.log(m))
            if guide_weight:
                proposal = np.logaddexp(proposal, math.log(guide_weight) + _guide(s, residual, words, s.k - j, mean))
            if product_weight:
                proposal = np.logaddexp(proposal, math.log(product_weight) + logp)
            draw = categorical(proposal, rng)
            logq = proposal[np.arange(particles), draw]
            words = np.column_stack((words, draw))
            residual -= s.phi[:, draw].T
            h = s.k - j - 1
            variance = s.noise_var + h / n
            new_guide = -.5 * (np.square(residual - h * mean).sum(1) / variance + n * math.log(variance / s.noise_var))
            if h:
                new_guide = np.clip(new_guide, guide_log_floor, 0.)
            increment = -math.log(m) + new_guide - old_guide - logq
            total = logsumexp(increment)
            evidence += total - math.log(particles)
            logw, old_guide = increment - total, new_guide
        records.append(words); all_weights.append(evidence + logw); evidences.append(evidence)
    words, logw = np.concatenate(records), np.concatenate(all_weights)
    weights = np.exp(logw - logsumexp(logw))
    words, movement = terminal_moves(s, words, rng, moves, block_size, batch_size)
    tails = count_tails(words, weights, m)
    return allocate(tails, s.k), {**info, **movement, 'tails': tails, 'batches': batches, 'particles_per_batch': particles,
                                'independent_confidence_units': batches, 'batch_log_evidence': list(map(float, evidences)),
                                'log_mean_weight': float(logsumexp(evidences) - math.log(batches)),
                                'unique_multisets': len(np.unique(np.sort(words, axis=1), axis=0)),
                                'target': 'physical_terminal_SMC_then_invariant_transport', 'excess_risk_bound': 1.,
                                'certificate': 'pooled unnormalized batch evidence and marks; particles are dependent'}
