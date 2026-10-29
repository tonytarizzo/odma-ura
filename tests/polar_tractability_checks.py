"""Tiny independent algebra checks for Report 11; no receiver implementation or performance experiment."""

import itertools
import json

import numpy as np

from baselines.polar import PolarCode


def rank(keys):
    pivots = {}
    for value in keys:
        value = int(value)
        while value:
            bit = value.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = value
                break
            value ^= pivots[bit]
    return len(pivots)


def characters(keys, b):
    return np.array([[(-1.) ** ((int(k) & v).bit_count() % 2) for k in keys] for v in range(1 << b)])


def basis_envelope(keys, coefficients, b, h0=0.):
    chosen = []
    for i in sorted(range(len(keys)), key=lambda i: (-abs(coefficients[i]), int(keys[i]))):
        if rank([keys[j] for j in chosen] + [keys[i]]) > len(chosen):
            chosen.append(i)
    signs = characters(keys, b)
    keep = np.zeros(len(keys), dtype=bool)
    keep[chosen] = True
    remainder = float(np.abs(coefficients[~keep]).sum())
    log_z0 = h0 + (b - len(chosen)) * np.log(2.) + np.logaddexp(coefficients[keep], -coefficients[keep]).sum()
    f = np.exp(h0 + signs @ coefficients)
    q = np.exp(h0 + signs[:, keep] @ coefficients[keep] - log_z0)
    envelope = np.exp(log_z0 + remainder)
    accept = np.exp(signs[:, ~keep] @ coefficients[~keep] - remainder)
    return chosen, f, q, envelope, accept


def main():
    rng = np.random.default_rng(11026)
    bound_slack, normalization_error, correction_error, detailed_balance_error, optimum_error = [], [], [], [], []
    for _ in range(40):
        b = 4
        keys = np.sort(rng.choice(np.arange(1, 1 << b), 8, replace=False))
        h = rng.normal(0., 1.1, len(keys))
        chosen, f, q, upper, accept = basis_envelope(keys, h, b)
        assert f.sum() <= upper * (1 + 1e-12)
        assert np.all(accept <= 1 + 1e-12)
        normalization_error.append(abs(q.sum() - 1))
        correction_error.append(float(np.max(np.abs(q * accept - f / upper))))
        bound_slack.append(float(upper / f.sum()))
        max_weight = max(sum(abs(h[i]) for i in subset) for subset in itertools.combinations(range(len(keys)), len(chosen))
                         if rank([keys[i] for i in subset]) == len(chosen))
        assert np.isclose(sum(abs(h[i]) for i in chosen), max_weight)
        log_envelopes = []
        for mask in range(1 << len(keys)):
            indices = [i for i in range(len(keys)) if (mask >> i) & 1]
            if rank([keys[i] for i in indices]) != len(indices):
                continue
            omitted = [i for i in range(len(keys)) if i not in indices]
            log_envelopes.append((b - len(indices)) * np.log(2.) + np.logaddexp(h[indices], -h[indices]).sum()
                                 + np.abs(h[omitted]).sum())
        optimum_error.append(abs(np.log(upper) - min(log_envelopes)))
        selected = np.argsort(-f)[:3]
        list_q = np.zeros_like(q)
        list_q[selected] = f[selected] / f[selected].sum()
        mixed_q = .25 * q + .75 * list_q
        target = f / f.sum()
        transition = np.empty((len(f), len(f)))
        for i in range(len(f)):
            transition[i] = mixed_q * np.minimum(1., f * mixed_q[i] / (f[i] * mixed_q))
            transition[i, i] += 1 - transition[i].sum()
        flux = target[:, None] * transition
        detailed_balance_error.append(float(np.max(np.abs(flux - flux.T))))
        actual_miss = 1 - f[selected].sum() / f.sum()
        certified_miss = 1 - f[selected].sum() / upper
        assert actual_miss <= certified_miss + 1e-12

    polar = PolarCode(6, 32, crc_bits=8, list_size=128)
    generator = polar.encode(np.eye(6, dtype=np.uint8)).T
    row_keys = generator @ (1 << np.arange(6))
    unique = sorted(set(map(int, row_keys)) - {0})
    edge_error = []
    for keys, h, b, h0 in ((np.array([], dtype=int), np.array([]), 4, -2.3),
                           (np.array([1, 2, 3]), np.array([.8, -1.4, .2]), 5, 1.7),
                           (np.array([1, 2, 4]), np.array([0., .2, -.3]), 3, -.4)):
        _, f, q, upper, accept = basis_envelope(keys, h, b, h0)
        edge_error.extend([abs(q.sum() - 1), float(np.max(abs(q * accept - f / upper)))])
        assert f.sum() <= upper * (1 + 1e-12)
    prefix_error = []
    grouped_fixtures = []
    for fixed in itertools.product((0, 1), repeat=2):
        coefficients = rng.normal(size=32)
        payload = np.array([list(fixed) + [int((z >> j) & 1) for j in range(4)] for z in range(16)], dtype=np.uint8)
        direct = (1. - 2. * ((payload @ generator.T) % 2)) @ coefficients
        prefix_sign = 1. - 2. * ((generator[:, :2] @ np.array(fixed)) % 2)
        restricted_keys = generator[:, 2:] @ (1 << np.arange(4))
        groups = {}
        for key, value in zip(restricted_keys, coefficients * prefix_sign):
            groups[int(key)] = groups.get(int(key), 0.) + float(value)
        h0 = groups.pop(0, 0.)
        keys, h = np.array(list(groups)), np.array(list(groups.values()))
        _, f, q, upper, accept = basis_envelope(keys, h, 4, h0)
        prefix_error.append(float(np.max(abs(np.log(f) - direct))))
        grouped_fixtures.append((f, q, upper, accept))
    total_upper = sum(item[2] for item in grouped_fixtures)
    multipattern_error = max(float(np.max(abs(q * upper / total_upper * accept - f / total_upper)))
                             for f, q, upper, accept in grouped_fixtures)
    triangle = next((a, b, a ^ b) for a, b in itertools.combinations(unique, 2) if (a ^ b) in unique)
    frustration = []
    for t in (1., 2., 4., 8.):
        _, f, q, upper, accept = basis_envelope(np.array(triangle), np.array([t, t, -t]), 6)
        exact = (3 * np.exp(t) + np.exp(-3 * t)) / (np.exp(t) * (2 * np.cosh(t)) ** 2)
        assert np.isclose(f.sum() / upper, exact)
        frustration.append({"t": t, "acceptance": float(exact), "expected_attempts": float(1 / exact)})

    metric_error = []
    for _ in range(12):
        llr = rng.normal(0., 3., 32)
        words, metrics = polar.decode_list(llr)
        if len(words):
            signs = 1. - 2. * polar.encode(words)
            direct = np.logaddexp(0., -signs * llr).sum(axis=1)
            metric_error.append(float(np.max(np.abs(metrics - direct))))
    assert metric_error

    # Every legal count state for a three-word, four-copy law; sparse histogram versus dense tails.
    states = np.array([a for a in itertools.product(range(5), repeat=3) if sum(a) == 4])
    weights = rng.random(len(states))
    weights /= weights.sum()
    sparse = {}
    for a, weight in zip(states, weights):
        for word in np.flatnonzero(a):
            sparse.setdefault(int(word), {}).setdefault(int(a[word]), 0.)
            sparse[int(word)][int(a[word])] += float(weight)
    dense = np.array([[(weights * (states[:, w] >= j)).sum() for j in range(1, 5)] for w in range(3)])
    sparse_tails = np.zeros_like(dense)
    for w, histogram in sparse.items():
        for count, mass in histogram.items():
            sparse_tails[w, :count] += mass
    assert np.allclose(sparse_tails, dense)
    allocation = np.zeros(3, dtype=int)
    for _ in range(4):
        gains = np.array([sparse_tails[w, allocation[w]] if allocation[w] < 4 else -1. for w in range(3)])
        allocation[np.argmax(gains)] += 1
    risk = lambda d: 1 - (weights[:, None] * np.minimum(states, d)).sum() / 4
    assert np.isclose(risk(allocation), min(map(risk, states)))

    output = {"random_fixtures": 40, "payloads_per_fixture": 16,
              "maximum_normalization_error": max(normalization_error),
              "maximum_rejection_identity_error": max(correction_error),
              "maximum_detailed_balance_error": max(detailed_balance_error),
              "maximum_log_envelope_optimality_error": max(optimum_error), "constant_rank_deficient_cases": 3,
              "maximum_constant_rank_deficient_error": max(edge_error), "affine_prefix_cases": 4,
              "maximum_affine_prefix_score_error": max(prefix_error), "multipattern_identity_error": multipattern_error,
              "envelope_over_partition_range": [min(bound_slack), max(bound_slack)],
              "selected_code_nonzero_character_rows": len(unique), "selected_code_triangle": triangle,
              "frustration": frustration, "scl_metric_calls": len(metric_error),
              "maximum_scl_metric_error": max(metric_error), "sparse_tail_error": float(np.max(abs(dense - sparse_tails)))}
    assert max(normalization_error + correction_error + detailed_balance_error + metric_error + optimum_error
               + edge_error + prefix_error + [multipattern_error]) < 1e-11
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
