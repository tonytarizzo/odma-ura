"""Exact three-user references and a clearly labelled edge-deletion diagnostic for Report 7."""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from tests import polar_posterior_test as reference
from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import binary_rank, exact_posterior


class PositivePolarPair:
    """Positive b contractions: direct overlap tables and direct XOR sums, avoiding 1+f cancellation."""

    def __init__(self, positions, energy, generator, sigma2, sampling):
        self.patterns, self.values = len(positions), 1 << generator.shape[1]
        signs = 1. - 2. * ((message_bits(np.arange(self.values), generator.shape[1]) @ generator.T) % 2)
        self.kernels = np.exp(-(energy @ signs.T) / sigma2)
        if sampling == "distinct":
            self.kernels[:, 0] = 0
        self.pairs, self.neighbours = [], [[] for _ in positions]
        for p, q in itertools.combinations(range(self.patterns), 2):
            _, left, right = np.intersect1d(positions[p], positions[q], return_indices=True)
            a, ia = np.unique(signs[:, left], axis=0, return_inverse=True)
            b, ib = np.unique(signs[:, right], axis=0, return_inverse=True)
            gram = (a * np.sqrt(energy[p, left])) @ (b * np.sqrt(energy[q, right])).T
            pair = np.exp(-gram / sigma2)
            self.pairs.append((p, q, ia, ib, pair))
            self.neighbours[p].append((q, ia, ib, pair))
            self.neighbours[q].append((p, ib, ia, pair.T))

    def row(self, word):
        p, v = divmod(word, self.values)
        result = np.empty((self.patterns, self.values))
        result[p] = self.kernels[p, np.arange(self.values) ^ v]
        for q, ia, ib, pair in self.neighbours[p]:
            result[q] = pair[ia[v], ib]
        return result.ravel()

    def __call__(self, x):
        x = np.asarray(x).reshape(self.patterns, self.values)
        result = np.zeros_like(x)
        indices = np.arange(self.values)
        for p in range(self.patterns):
            for shift, coefficient in enumerate(self.kernels[p]):
                result[p] += coefficient * x[p, indices ^ shift]
        for p, q, ia, ib, pair in self.pairs:
            a = np.bincount(ia, weights=x[p], minlength=pair.shape[0])
            b = np.bincount(ib, weights=x[q], minlength=pair.shape[1])
            result[p] += (pair @ b)[ia]
            result[q] += (pair.T @ a)[ib]
        return result.ravel()


def three_user_counts(nu, row, matvec):
    """One streamed pair contraction per word; the three positive count numerators determine Z3."""
    numerators = np.zeros((len(nu), 3))
    for word in range(len(nu)):
        pair = row(word)
        local = nu * pair
        local[word] = 0
        diagonal = pair[word]
        numerators[word, 0] = 3 * nu[word] * (local @ matvec(local))
        numerators[word, 1] = 3 * nu[word] ** 2 * diagonal * (pair @ local)
        numerators[word, 2] = nu[word] ** 3 * diagonal ** 3
    normalizer = (numerators @ np.arange(1, 4)).sum() / 3
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise FloatingPointError("Three-user normalizer is nonpositive or nonfinite; rescaling/precision is required")
    positive = numerators / normalizer
    return np.column_stack((1 - positive.sum(1), positive)), normalizer


def triangle_quotient(positions, energy, generator, nu, patterns, sigma2, sampling, marked):
    """Count-marker polynomial for one ordered pattern triple, with exact joint overlap-parity grouping."""
    values = 1 << generator.shape[1]
    signs = 1. - 2. * ((message_bits(np.arange(values), generator.shape[1]) @ generator.T) % 2)
    n = int(positions.max()) + 1
    representatives, unaries, waveforms = [], [], []
    for slot, p in enumerate(patterns):
        other_resources = np.concatenate([positions[q] for i, q in enumerate(patterns) if i != slot])
        local = np.isin(positions[p], other_resources)
        _, first, mapping = np.unique(signs[:, local], axis=0, return_index=True, return_inverse=True)
        representatives.append(first)
        unary = np.zeros((len(first), 2))
        weights = nu[p * values:(p + 1) * values].copy()
        if marked // values == p:
            v = marked % values
            weights[v] = 0
            unary[mapping[v], 1] = nu[marked]
        unary[:, 0] = np.bincount(mapping, weights=weights, minlength=len(first))
        unaries.append(unary)
        wave = np.zeros((len(first), n))
        wave[:, positions[p]] = signs[first] * np.sqrt(energy[p])
        waveforms.append(wave)
    factors = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair = np.exp(-(waveforms[i] @ waveforms[j].T) / sigma2)
        if sampling == "distinct" and patterns[i] == patterns[j]:
            # A repeated pattern exposes the full generator, so each parity class is a single payload.
            if len(representatives[i]) != values or len(representatives[j]) != values:
                raise ValueError("Repeated-pattern exclusion requires an injective payload generator")
            pair[representatives[i][:, None] == representatives[j][None, :]] = 0
        factors.append(pair)
    polynomial = np.zeros(4)
    for degrees in itertools.product(range(2), repeat=3):
        polynomial[sum(degrees)] += np.einsum("i,j,k,ij,ik,jk->", *(u[:, d] for u, d in zip(unaries, degrees)),
                                             *factors, optimize=True)
    return polynomial, [len(r) for r in representatives]


def edge_deletion_diagnostic(phi, y, sigma2, sampling):
    """Exact counterfactual deleting one Gaussian edge while retaining the complete source constraint."""
    m = phi.shape[1]
    words = np.array(list(itertools.product(range(m), repeat=3)))
    if sampling == "distinct":
        words = words[(words[:, 0] != words[:, 1]) & (words[:, 0] != words[:, 2]) & (words[:, 1] != words[:, 2])]
    gram = phi.T @ phi
    unary = phi.T @ y / sigma2
    open_log = unary[words].sum(1) - (gram[words[:, 0], words[:, 1]] + gram[words[:, 1], words[:, 2]]) / sigma2
    closure_log = -gram[words[:, 0], words[:, 2]] / sigma2
    opened, full = softmax(open_log), softmax(open_log + closure_log)
    counts = np.eye(m, dtype=int)[words].sum(1)
    events = (counts[:, :, None] >= np.arange(1, 4)).reshape(len(words), -1)
    open_tails, tails = (opened @ events).reshape(m, 3), (full @ events).reshape(m, 3)
    proposed, best = tail_action(open_tails, 3), tail_action(tails, 3)
    proposed_gain = (tails * (np.arange(3) < proposed[:, None])).sum()
    optimal_gain = (tails * (np.arange(3) < best[:, None])).sum()
    closure = np.exp(closure_log)
    mean = opened @ closure
    covariance = ((opened * closure) @ events - mean * (opened @ events)).reshape(m, 3)
    np.testing.assert_allclose(tails - open_tails, covariance / mean, rtol=1e-10, atol=2e-13)
    return {"max_tail_change": float(abs(tails - open_tails).max()),
            "posterior_pupe_regret": float((optimal_gain - proposed_gain) / 3),
            "closure_mean": float(mean)}


def audit():
    reference.PolarPosteriorTests.setUpClass()
    f = reference.PolarPosteriorTests
    generator = f.base.polar.encode(np.eye(2, dtype=np.uint8)).T
    rows = []
    for sampling in ("iid", "distinct"):
        for sigma2 in (.05, .3, 1.):
            operator = PositivePolarPair(f.base.positions, f.energy, generator, sigma2, sampling)
            for seed in range(4):
                rng = np.random.default_rng(9100 + seed)
                truth = rng.choice(16, 3, replace=sampling == "iid")
                y = f.phi[:, truth].sum(1) + math.sqrt(sigma2) * rng.normal(size=12)
                nu = softmax(f.phi.T @ y / sigma2)
                probabilities, normalizer = three_user_counts(nu, operator.row, operator)
                states, posterior = exact_posterior(f.phi, y, 3, sigma2, sampling)
                expected = np.array([posterior @ (states == j) for j in range(4)]).T
                np.testing.assert_allclose(probabilities, expected, rtol=2e-10, atol=2e-13)
                tails = np.flip(np.cumsum(np.flip(probabilities[:, 1:], axis=1), axis=1), axis=1)
                exact_tails = np.array([posterior @ (states >= j) for j in range(1, 4)]).T
                action, best = tail_action(tails, 3), tail_action(exact_tails, 3)
                gain = (exact_tails * (np.arange(3) < action[:, None])).sum()
                optimal = (exact_tails * (np.arange(3) < best[:, None])).sum()
                if abs(gain - optimal) > 1e-10:
                    raise AssertionError("Three-user posterior produced a non-Bayes action")
                pair = np.exp(-(f.phi.T @ f.phi) / sigma2)
                if sampling == "distinct":
                    np.fill_diagonal(pair, 0)
                matrix = np.sqrt(nu[:, None] * nu[None, :]) * pair
                np.testing.assert_allclose(normalizer, np.trace(matrix @ matrix @ matrix), rtol=2e-10)
                rows.append({"sampling": sampling, "sigma2": sigma2, "seed": seed,
                             "max_count_probability_error": float(abs(probabilities - expected).max()),
                             **edge_deletion_diagnostic(f.phi, y, sigma2, sampling)})
    cuts = []
    for seed in (3201, 3202, 3203):
        base = ODMAPolarBaseline(12, 256, seed, prefix_bits=6, code_length=32, crc_bits=8)
        generator = base.polar.encode(np.eye(6, dtype=np.uint8)).T
        for patterns in itertools.combinations(range(8), 3):
            ranks = []
            for i, p in enumerate(patterns):
                neighbours = np.concatenate([base.positions[q] for j, q in enumerate(patterns) if j != i])
                ranks.append(binary_rank(generator[np.isin(base.positions[p], neighbours)]))
            cuts.append({"seed": seed, "patterns": patterns, "joint_ranks": ranks,
                         "joint_signature_tuples": 1 << sum(ranks)})
    return {"observations": rows, "first_eight_pattern_triples": cuts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"Saved three-user audit to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
