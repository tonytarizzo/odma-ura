"""Finite parity-cover identities and a placement-only rank census; no decoder or runtime benchmark."""

import itertools
import json

import numpy as np

from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode


def span(keys):
    values = {0}
    for key in keys:
        values |= {v ^ int(key) for v in values}
    return values


def basis(keys):
    pivots, selected = {}, []
    for key in keys:
        value = int(key)
        while value:
            bit = value.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = value
                selected.append(int(key))
                break
            value ^= pivots[bit]
    return selected


def affine_maps(keys, b):
    h = np.array([[(key >> j) & 1 for j in range(b)] for key in keys], dtype=np.uint8).reshape(len(keys), b)
    a = np.concatenate((h, np.eye(len(keys), dtype=np.uint8)), axis=1)
    pivots = []
    for col in range(b):
        candidates = np.flatnonzero(a[len(pivots):, col])
        if not len(candidates):
            continue
        row = len(pivots)
        other = row + candidates[0]
        a[[row, other]] = a[[other, row]]
        for j in range(len(keys)):
            if j != row and a[j, col]:
                a[j] ^= a[row]
        pivots.append(col)
    assert len(pivots) == len(keys)
    free = [j for j in range(b) if j not in pivots]
    right = np.zeros((b, len(keys)), dtype=np.uint8)
    null = np.zeros((b, len(free)), dtype=np.uint8)
    right[pivots] = a[:, b:]
    null[free, np.arange(len(free))] = 1
    null[pivots] = a[:, free]
    assert np.array_equal((h @ right) % 2, np.eye(len(keys), dtype=np.uint8))
    assert not np.any((h @ null) % 2)
    return right, null


def inner_envelope(coefficients, offset, null):
    b, q = null.shape
    grouped = {}
    for key, value in coefficients.items():
        row = np.array([(key >> j) & 1 for j in range(b)], dtype=np.uint8)
        restricted = int(((row @ null) % 2) @ (1 << np.arange(q)))
        grouped[restricted] = grouped.get(restricted, 0.) + value * (-1.) ** int(row @ offset % 2)
    h0 = grouped.pop(0, 0.)
    selected = set(basis(sorted(grouped, key=lambda key: (-abs(grouped[key]), key))))
    tail = sum(abs(value) for key, value in grouped.items() if key not in selected)
    log_z = h0 + (q - len(selected)) * np.log(2.) + sum(np.logaddexp(grouped[key], -grouped[key]) for key in selected)
    rows = np.array([[(z >> j) & 1 for j in range(q)] for z in range(1 << q)], dtype=np.uint8).reshape(1 << q, q)
    payloads = ((offset + rows @ null.T) % 2) @ (1 << np.arange(b))
    scores = np.array([h0 + sum(value * (-1.) ** ((key & z).bit_count() % 2) for key, value in grouped.items())
                       for z in range(1 << q)])
    reference = np.array([h0 + sum(grouped[key] * (-1.) ** ((key & z).bit_count() % 2) for key in selected)
                          for z in range(1 << q)])
    return payloads, np.exp(scores), np.exp(reference - log_z), np.exp(log_z - tail), np.exp(log_z + tail)


def physical_factors(generator, positions, energy, residual, sigma2):
    keys = [int(row @ (1 << np.arange(generator.shape[1]))) for row in generator]
    unary, interaction, constant = [], {}, 0.
    for sites, e in zip(positions, energy):
        values = {}
        for key, value in zip(keys, np.sqrt(e) * residual[sites] / sigma2):
            values[key] = values.get(key, 0.) + value
        unary.append(values)
    for k, l in itertools.combinations(range(len(positions)), 2):
        _, left, right = np.intersect1d(positions[k], positions[l], return_indices=True)
        for i, j in zip(left, right):
            g, h, value = keys[i], keys[j], np.sqrt(energy[k, i] * energy[l, j]) / sigma2
            if g == h == 0:
                constant -= value
            elif g == 0:
                unary[l][h] = unary[l].get(h, 0.) - value
            elif h == 0:
                unary[k][g] = unary[k].get(g, 0.) - value
            else:
                term = k, l, g, h
                interaction[term] = interaction.get(term, 0.) + value
    return unary, interaction, constant


def verify_cover(generator, positions, energy, residual, sigma2, covers):
    b, m, n = generator.shape[1], len(positions), len(residual)
    words = np.array([[(v >> j) & 1 for j in range(b)] for v in range(1 << b)], dtype=np.uint8)
    signs = 1. - 2. * ((words @ generator.T) % 2)
    waveforms = np.zeros((m, 1 << b, n))
    for k in range(m):
        waveforms[k][:, positions[k]] = signs * np.sqrt(energy[k])
    tuples = np.array(list(itertools.product(range(1 << b), repeat=m)))
    sums = sum(waveforms[k, tuples[:, k]] for k in range(m))
    score = (np.square(residual).sum() - np.square(residual - sums).sum(1) + m) / (2 * sigma2)
    f = np.exp(score)
    unary, interaction, constant = physical_factors(generator, positions, energy, residual, sigma2)
    index = {tuple(row): i for i, row in enumerate(tuples)}
    maps = [affine_maps(h, b) for h in covers]
    spaces = [span(h) for h in covers]
    tail = sum(abs(value) for (k, l, g, h), value in interaction.items() if g not in spaces[k] and h not in spaces[l])
    reference, envelope_density = np.zeros(len(f)), np.zeros(len(f))
    lower, upper, sectors = 0., 0., 0
    for alpha in itertools.product(*[range(1 << len(h)) for h in covers]):
        offsets = [right @ np.array([(a >> j) & 1 for j in range(right.shape[1])], dtype=np.uint8) % 2
                   for a, (right, _) in zip(alpha, maps)]
        integers = [int(offset @ (1 << np.arange(b))) for offset in offsets]
        fields, c = [dict(values) for values in unary], constant
        for (k, l, g, h), value in interaction.items():
            left, right = g in spaces[k], h in spaces[l]
            sg, sh = (-1.) ** ((g & integers[k]).bit_count() % 2), (-1.) ** ((h & integers[l]).bit_count() % 2)
            if left and right:
                c -= value * sg * sh
            elif left:
                fields[l][h] = fields[l].get(h, 0.) - value * sg
            elif right:
                fields[k][g] = fields[k].get(g, 0.) - value * sh
        inner = [inner_envelope(field, offset, null) for field, offset, (_, null) in zip(fields, offsets, maps)]
        sector_lower = np.exp(c) * np.prod([entry[3] for entry in inner])
        sector_upper = np.exp(c) * np.prod([entry[4] for entry in inner])
        lower += sector_lower
        upper += sector_upper
        sectors += 1
        for choices in itertools.product(*[range(len(entry[0])) for entry in inner]):
            key = tuple(int(entry[0][i]) for entry, i in zip(inner, choices))
            j = index[key]
            reference[j] = np.exp(c) * np.prod([entry[1][i] for entry, i in zip(inner, choices)])
            envelope_density[j] = sector_upper * np.prod([entry[2][i] for entry, i in zip(inner, choices)])
    proposal = envelope_density / upper
    acceptance = f / (np.exp(tail) * envelope_density)
    pi, pi_reference = f / f.sum(), reference / reference.sum()
    variation = float(abs(pi - pi_reference).sum() / 2)
    errors = [abs(proposal.sum() - 1), float(np.max(abs(proposal * acceptance - f / (np.exp(tail) * upper)))),
              abs(envelope_density.sum() / upper - 1)]
    if tail == 0:
        errors.append(float(np.max(abs(np.log(reference) - score))))
    assert np.all(abs(score - np.log(reference)) <= tail + 1e-11)
    assert np.max(acceptance) <= 1 + 1e-11
    assert variation <= np.tanh(tail / 2) + 1e-11
    assert np.exp(-tail) * reference.sum() <= f.sum() * (1 + 1e-11) <= np.exp(tail) * reference.sum() * (1 + 1e-11) ** 2
    assert f.sum() / (np.exp(tail) * upper) >= np.exp(-2 * tail) * lower / upper - 1e-11
    return errors, {"conditioned_rank": sum(map(len, covers)), "sectors": sectors, "omitted_log_bound": float(tail),
                    "tv_from_omission": variation, "corrected_acceptance": float(f.sum() / (np.exp(tail) * upper))}


def main():
    rng = np.random.default_rng(112810)
    generator = PolarCode(3, 16, crc_bits=4).encode(np.eye(3, dtype=np.uint8)).T
    positions = np.arange(48).reshape(3, 16)
    positions[1, [2, 3, 11, 12]] = positions[0, [3, 1, 11, 0]]
    positions[2, [0, 5, 10]] = positions[0, [4, 14, 12]]
    energy = rng.uniform(.5, 1.5, (3, 16))
    energy /= energy.sum(axis=1, keepdims=True)
    errors, examples = [], []
    for fixture in range(12):
        residual = rng.normal(0, .6, 48)
        for covers in ([[], [], []], [[1], [], []], [[1, 2], [], []], [[1, 2], [1], [2]]):
            values, summary = verify_cover(generator, positions, energy, residual, .8, covers)
            errors.extend(values)
            if fixture == 0:
                examples.append(summary)
    # A same-pattern control includes ordered duplicate payloads and full-rank restrictions.
    values, _ = verify_cover(generator, np.tile(np.arange(16), (2, 1)), np.tile(energy[0], (2, 1)), rng.normal(0, .3, 16),
                             .8, [[1, 2, 4], []])
    errors.extend(values)

    weights = {1: 9., 2: 9., 3: 9., 4: 10., 8: 10.}
    retained, current = [], {0}
    for _ in range(2):
        key = max((key for key in weights if key not in current),
                  key=lambda key: (sum(value for g, value in weights.items() if g in span(retained + [key])), -key))
        retained.append(key)
        current = span(retained)
    greedy = sum(value for key, value in weights.items() if key in current)
    optimum = max(sum(value for key, value in weights.items() if key in span(pair))
                  for pair in itertools.combinations(weights, 2))
    assert greedy == 20 and optimum == 27

    model = ODMAPolarBaseline(12, 256, 3201, prefix_bits=6, code_length=32, crc_bits=8)
    keys = model.polar.encode(np.eye(6, dtype=np.uint8)).T @ (1 << np.arange(6))
    ranks, old_entries = [], []
    for p, q in itertools.combinations(model.positions, 2):
        _, left, right = np.intersect1d(p, q, return_indices=True)
        interacting = (keys[left] != 0) & (keys[right] != 0)
        rp, rq = len(basis(keys[left[interacting]])), len(basis(keys[right[interacting]]))
        ranks.append(min(rp, rq))
        old_entries.append(1 << (rp + rq))
    census = {"B": 12, "b": 6, "d": 32, "n": 256, "seed": 3201, "unordered_distinct_pattern_pairs": len(ranks),
              "nonconstant_min_rank_histogram": {str(rank): ranks.count(rank) for rank in sorted(set(ranks))},
              "one_sided_sector_sum": sum(1 << rank for rank in ranks),
              "two_sided_nonconstant_table_entries": sum(old_entries),
              "scope": "placement/generator only; unary constant-row factors removed; no runtime or acceptance claim"}
    assert max(errors) < 1e-11
    print(json.dumps({"physical_star_fixtures": 12, "cover_choices_per_fixture": 4, "payload_tuples_per_fixture": 512,
                      "maximum_identity_error": max(errors), "first_fixture_covers": examples,
                      "span_greedy_counterexample": {"greedy_weight": greedy, "optimal_weight": optimum},
                      "placement_rank_census": census}, indent=2))


if __name__ == "__main__":
    main()
