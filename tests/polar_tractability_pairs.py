"""Tiny pair-channel algebra checks and limited generator rank controls; no receiver implementation."""

import itertools
import json

import numpy as np

from baselines.polar import PolarCode


def rank(rows):
    pivots = {}
    for row in rows:
        key = sum(int(x) << j for j, x in enumerate(row))
        while key:
            bit = key.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = key
                break
            key ^= pivots[bit]
    return len(pivots)


def main():
    rng = np.random.default_rng(11028)
    code = PolarCode(2, 8, crc_bits=4, list_size=16)
    payloads = np.array(list(itertools.product((0, 1), repeat=2)), dtype=np.uint8)
    coded = code.encode(payloads)
    signs = 1. - 2. * coded
    amplitude = np.sqrt(np.arange(1., 9.) / 36.)
    errors, examples = [], []
    for fixture in range(20):
        residual = rng.normal(size=8)
        sigma2 = .3
        theta = 2 * amplitude * residual / sigma2
        kappa = 2 * amplitude ** 2 / sigma2
        logcosh = np.logaddexp(theta, -theta) - np.log(2.)
        weights, guide, correction = [], [], []
        for difference in range(4):
            mask = 1 - coded[difference]
            log_joint = (signs * mask) @ theta - mask @ kappa
            other = np.array([v ^ difference for v in range(4)])
            physical = amplitude * (signs + signs[other])
            direct = -(np.square(residual - physical).sum(axis=1) - residual @ residual) / (2 * sigma2)
            errors.append(float(np.max(np.abs(direct - log_joint))))
            log_guide = mask @ (logcosh - kappa)
            factor = np.exp((signs * mask) @ theta - mask @ logcosh).mean()
            target_mass = np.exp(log_joint).sum()
            errors.append(abs(target_mass / (4 * np.exp(log_guide) * factor) - 1))
            weights.append(target_mass)
            guide.append(np.exp(log_guide))
            correction.append(factor)
        posterior = np.array(weights) / np.sum(weights)
        guide = np.array(guide) / np.sum(guide)
        if np.argmax(posterior) != np.argmax(guide) and not examples:
            examples.append({"fixture": fixture, "residual": residual.tolist(), "sigma2": sigma2,
                             "exact_difference_probabilities": posterior.tolist(), "guide_probabilities": guide.tolist(),
                             "code_consistency_factors": correction})

    # Exact F2^2 minus and plus kernels: two local observations, no frozen/CRC shortcut.
    a = rng.random(4) + .05
    b = rng.random(4) + .05
    joint = np.array([[a[u ^ v] * b[v] for v in range(4)] for u in range(4)])
    minus = np.array([sum(a[u ^ v] * b[v] for v in range(4)) / 4 for u in range(4)])
    errors.append(float(np.max(abs(joint.sum(axis=1) / 4 - minus))))
    for u in range(4):
        plus = np.array([a[u ^ v] * b[v] for v in range(4)])
        errors.append(float(np.max(abs(plus / plus.sum() - joint[u] / joint[u].sum()))))

    # A proposed same-pattern block list mixed with a full two-slot prior, checked on all 64 ordered pairs.
    m = 8
    phi = rng.normal(size=(5, m))
    phi /= np.linalg.norm(phi, axis=0)
    residual, sigma2 = rng.normal(size=5), .6
    pairs = np.array(list(itertools.product(range(m), repeat=2)))
    target = np.exp(-np.square(residual - phi[:, pairs[:, 0]].T - phi[:, pairs[:, 1]].T).sum(axis=1) / (2 * sigma2))
    target /= target.sum()
    same_pattern = pairs[:, 0] // 4 == pairs[:, 1] // 4
    retained = np.flatnonzero(same_pattern)[::3]
    proposal = np.full(len(pairs), .2 / m ** 2)
    proposal[retained] += .8 * target[retained] / target[retained].sum()
    transition = np.minimum(proposal[None, :], target[None, :] * (proposal / target)[:, None])
    transition[np.diag_indices(len(pairs))] += 1 - transition.sum(axis=1)
    errors.extend([abs(proposal.sum() - 1), float(np.max(abs(target @ transition - target))),
                   float(np.max(abs(target[:, None] * transition - target[None, :] * transition.T)))])

    controls = []
    for b, d, crc in ((6, 32, 8), (32, 128, 16), (88, 512, 16)):
        code = PolarCode(b, d, crc_bits=crc)
        generator = code.encode(np.eye(b, dtype=np.uint8)).T
        if b == 6:
            differences = np.array([[(v >> j) & 1 for j in range(b)] for v in range(1, 1 << b)], dtype=np.uint8)
            scope = "all_nonzero_differences"
        else:
            differences = rng.integers(0, 2, size=(16, b), dtype=np.uint8)
            assert np.all(differences.any(axis=1))
            scope = "16_seeded_nonzero_differences_only"
        ranks = [rank(generator[(generator @ difference) % 2 == 0]) for difference in differences]
        controls.append({"b": b, "d": d, "crc": crc, "scope": scope,
                         "minimum_zero_mask_rank": min(ranks), "maximum_zero_mask_rank": max(ranks),
                         "unique_pair_rank_needed": b - 1})
    assert examples
    assert max(errors) < 1e-11
    print(json.dumps({"same_pattern_fixtures": 20, "payload_pairs_per_fixture": 16,
                      "maximum_identity_error": max(errors), "xor_guide_counterexample": examples[0],
                      "limited_noiseless_pair_rank_controls": controls}, indent=2))


if __name__ == "__main__":
    main()
