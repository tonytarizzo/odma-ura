"""Finite exact-bound and polar interaction contractions for Report 7; not a production receiver."""

import argparse
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import softmax

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode
from tests import polar_posterior_test as reference
from tests.polar_exact_computation import certificate_search, completion_bound, tail_action
from tests.polar_posterior_test import binary_rank, exact_posterior


class CachedCompletion:
    """Eight root dual trials, with exact support values for every suffix and remaining count."""

    def __init__(self, phi, y, k, sampling, iterations=8):
        self.phi, self.sampling = phi, sampling
        n, m = phi.shape
        point, directions, scores = k * phi.mean(1), [], []
        for _ in range(iterations):
            u = y - point
            score = u @ phi
            directions.append(u)
            scores.append(score)
            vertex = k * phi[:, np.argmax(score)] if sampling == "iid" else phi[:, np.argsort(score)[-k:]].sum(1)
            step = vertex - point
            norm2 = step @ step
            if norm2 == 0:
                break
            point += np.clip(u @ step / norm2, 0, 1) * step
        self.directions = np.asarray(directions)
        self.norm2 = np.square(self.directions).sum(1)
        self.support = np.zeros((len(directions), m, k + 1))
        for i, score in enumerate(scores):
            for depth in range(m):
                if sampling == "iid":
                    self.support[i, depth] = np.arange(k + 1) * score[depth:].max()
                else:
                    cap = min(k, m - depth)
                    self.support[i, depth, 1:cap + 1] = np.cumsum(np.sort(score[depth:])[::-1][:cap])
        self.setup_calls, self.setup_scores = len(directions), len(directions) * m
        self.direction_evaluations = 0

    def __call__(self, depth, residual, h):
        box, _ = completion_bound(self.phi[:, depth:], residual, h, self.sampling, "box")
        trials = 2 * (self.directions @ residual) - self.norm2 - 2 * self.support[:, depth, h]
        self.direction_evaluations += len(trials)
        return max(box, float(trials.max())), 0


@lru_cache(maxsize=None)
def two_count_prior(remaining, h, sampling):
    """Joint counts of two named, different words under a subtree's normalized prior."""
    probability = np.zeros((h + 1, h + 1))
    for a in range(min(h, 1) + 1 if sampling == "distinct" else h + 1):
        for b in range(min(h - a, 1) + 1 if sampling == "distinct" else h - a + 1):
            rest = h - a - b
            if sampling == "iid":
                probability[a, b] = (math.factorial(h) / math.factorial(a) / math.factorial(b)
                                     / math.factorial(rest) * (remaining - 2) ** rest / remaining ** h)
            elif 0 <= rest <= remaining - 2:
                probability[a, b] = math.comb(remaining - 2, rest) / math.comb(remaining, h)
    return probability


def negative_prior_matrix(prefix, h, m, k, sampling):
    """Exact prior probability of each negative tail contrast, conditional on a fixed count prefix."""
    depth, remaining = len(prefix), m - len(prefix)
    thresholds = np.arange(1, k + 1)
    tail = np.zeros((m, k))
    tail[:depth] = np.asarray(prefix)[:, None] >= thresholds
    if remaining >= 2:
        joint = two_count_prior(remaining, h, sampling)
        marginal = joint.sum(1)
    else:
        marginal = np.eye(1, h + 1, h).ravel()
        joint = np.zeros((h + 1, h + 1))
    values = np.arange(h + 1)
    tail[depth:] = marginal @ (values[:, None] >= thresholds)
    flat = tail.ravel()
    negative = (1 - flat[:, None]) * flat[None, :]
    pair = (values[:, None] < thresholds).T @ joint @ (values[:, None] >= thresholds)
    negative[depth * k:, depth * k:] = np.tile(pair, (remaining, remaining))
    same = ((values[:, None, None] < thresholds[None, :, None])
            & (values[:, None, None] >= thresholds[None, None, :]))
    for word in range(depth, m):
        negative[word * k:(word + 1) * k, word * k:(word + 1) * k] = np.einsum("a,aij->ij", marginal, same)
    return negative


def weighted_tree_sums(matvec, nu, k):
    """Exact sums over labelled spanning trees, using a rooted exponential generating function."""
    m = len(nu)
    rooted, branches, exponential = (np.zeros((m, k + 1)) for _ in range(3))
    rooted[:, 1], exponential[:, 0] = 1, 1
    sums = np.ones(k + 1)
    for size in range(2, k + 1):
        degree = size - 1
        branches[:, degree] = matvec(nu * rooted[:, degree])
        exponential[:, degree] = sum(j * branches[:, j] * exponential[:, degree - j]
                                      for j in range(1, degree + 1)) / degree
        rooted[:, size] = exponential[:, degree]
        sums[size] = math.factorial(size - 1) * (nu @ rooted[:, size])
    return sums


def walsh(x):
    result = np.asarray(x, dtype=float).copy()
    width = 1
    while width < result.shape[-1]:
        blocks = result.reshape(*result.shape[:-1], -1, 2 * width)
        left, right = blocks[..., :width].copy(), blocks[..., width:].copy()
        blocks[..., :width], blocks[..., width:] = left + right, left - right
        width *= 2
    return result


class PolarPairOperator:
    """Exact f=exp(-Gram/sigma2)-1 or |f| matvec via overlap parities and XOR convolution."""

    def __init__(self, positions, energy, generator, sigma2, sampling, absolute=True):
        patterns, length = positions.shape
        payloads = message_bits(np.arange(1 << generator.shape[1]), generator.shape[1])
        signs = 1. - 2. * ((payloads @ generator.T) % 2)
        self.patterns, self.values, self.pairs = patterns, len(payloads), []
        kernels = np.expm1(-(energy @ signs.T) / sigma2)
        if sampling == "distinct":
            kernels[:, 0] = -1.
        if absolute:
            kernels = abs(kernels)
        self.spectra = walsh(kernels)
        for p, q in itertools.combinations(range(patterns), 2):
            _, left, right = np.intersect1d(positions[p], positions[q], return_indices=True)
            if not len(left):
                continue
            a, ia = np.unique(signs[:, left], axis=0, return_inverse=True)
            b, ib = np.unique(signs[:, right], axis=0, return_inverse=True)
            gram = (a * np.sqrt(energy[p, left])) @ (b * np.sqrt(energy[q, right])).T
            factor = np.expm1(-gram / sigma2)
            self.pairs.append((p, q, ia, ib, abs(factor) if absolute else factor))

    def __call__(self, x):
        x = np.asarray(x).reshape(self.patterns, self.values)
        result = walsh(walsh(x) * self.spectra) / self.values
        for p, q, ia, ib, factor in self.pairs:
            a = np.bincount(ia, weights=x[p], minlength=factor.shape[0])
            b = np.bincount(ib, weights=x[q], minlength=factor.shape[1])
            result[p] += (factor @ b)[ia]
            result[q] += (factor.T @ a)[ib]
        return result.ravel()


def audit_pair_storage():
    generator = PolarCode(6, 32, 8).encode(np.eye(6, dtype=np.uint8)).T
    rows = []
    for bits in (12, 14):
        for seed in (3201, 3202, 3203):
            positions = ODMAPolarBaseline(bits, 256, seed, prefix_bits=bits - 6, code_length=32, crc_bits=8).positions
            pairs, dense, compressed, histogram = 0, 0, 0, {}
            for p, q in itertools.combinations(range(len(positions)), 2):
                _, left, right = np.intersect1d(positions[p], positions[q], return_indices=True)
                if not len(left):
                    continue
                ranks = binary_rank(generator[left]), binary_rank(generator[right])
                exponent = sum(ranks)
                histogram[exponent] = histogram.get(exponent, 0) + 1
                compressed += 1 << exponent
                dense += 64 ** 2
                pairs += 1
            rows.append({"B": bits, "seed": seed, "intersecting_unordered_pairs": pairs,
                         "dense_pair_entries": dense, "parity_pair_entries": compressed,
                         "same_pattern_kernel_entries": len(positions) * 64, "rank_sum_histogram": histogram})
    return rows


def audit_reuse():
    reference.PolarPosteriorTests.setUpClass()
    f = reference.PolarPosteriorTests
    rows, tree_rows = [], []
    for sampling in ("iid", "distinct"):
        for sigma2 in (.05, .3, 1.):
            for seed in range(4):
                rng = np.random.default_rng(9100 + seed)
                truth = rng.choice(16, 3, replace=sampling == "iid")
                y = f.phi[:, truth].sum(1) + math.sqrt(sigma2) * rng.normal(size=12)
                phi = f.phi[:, np.argsort(-(f.phi.T @ y), kind="stable")]
                states, posterior = exact_posterior(phi, y, 3, sigma2, sampling)
                tails = np.array([posterior @ (states >= j) for j in range(1, 4)]).T
                best = tail_action(tails, 3)
                optimal = (tails * (np.arange(3) < best[:, None])).sum()
                for mode, weighted in (("box", False), ("cached", False), ("cached", True), ("dual", False), ("dual", True)):
                    cache = CachedCompletion(phi, y, 3, sampling) if mode == "cached" else None
                    action, meta = certificate_search(phi, y, 3, sigma2, sampling, mode, "difference", bounder=cache,
                                                      negative_probability=negative_prior_matrix if weighted else None)
                    gain = (tails * (np.arange(3) < action[:, None])).sum()
                    if abs(gain - optimal) > 1e-10:
                        raise AssertionError("Reused/event-specific certificate disagrees with exhaustive Bayes inference")
                    rows.append({"sampling": sampling, "sigma2": sigma2, "seed": seed, "mode": mode,
                                 "prior_weighted": weighted, "cached_direction_evaluations": 0 if cache is None
                                 else cache.direction_evaluations, **meta})
                nu = softmax(phi.T @ y / sigma2)
                pair = np.exp(-(phi.T @ phi) / sigma2)
                if sampling == "distinct":
                    np.fill_diagonal(pair, 0)
                absolute = abs(pair - 1)
                d, epsilon = absolute @ nu, absolute.max()
                old, new = 3 * d.max() ** 2 * (1 + epsilon), 3 * (nu @ np.square(d)) * (1 + epsilon)
                partial = []
                for word in range(16):
                    c1 = np.array([1 - nu[word], nu[word]])
                    degree = (np.arange(16) == word).astype(int)
                    c2 = np.bincount((degree[:, None] + degree[None, :]).ravel(),
                                     weights=(nu[:, None] * nu[None, :] * (pair - 1)).ravel(), minlength=3)
                    polynomial = np.convolve(np.convolve(c1, c1), c1) + 3 * np.convolve(c1, c2)
                    partial.append(np.flip(np.cumsum(np.flip(polynomial[1:]))))
                partial = np.asarray(partial)
                proposed = tail_action(np.maximum(partial - new, 0), 3)
                selected = [(w, proposed[w] - 1) for w in range(16) if proposed[w]]
                next_copy = [(w, proposed[w]) for w in range(16) if proposed[w] < 3]
                certified = all(partial[a] - partial[b] >= new for a in selected for b in next_copy if a[0] != b[0])
                tree_rows.append({"sampling": sampling, "sigma2": sigma2, "seed": seed,
                                  "new_over_old_bound": new / old, "certified": certified})
    return {"search": rows, "trees": tree_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("pairs", "reuse", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    if args.part in ("pairs", "all"):
        result["pairs"] = audit_pair_storage()
    if args.part in ("reuse", "all"):
        result["reuse"] = audit_reuse()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"Saved finite reuse audit to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
