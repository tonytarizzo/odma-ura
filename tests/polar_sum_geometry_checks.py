"""Finite sum-geometry certificates for Reports 7/8/11; no receiver or PUPE benchmark."""

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode
from benchmarks.polar_energy_geometry import balanced_energy


def parity(x):
    return int(x).bit_count() & 1


def basis(keys):
    pivots = {}
    for key in keys:
        x = int(key)
        while x:
            bit = x.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = x
                break
            x ^= pivots[bit]
    return list(pivots.values())


def null_basis(keys, b):
    rows, pivots = list(map(int, keys)), []
    rank = 0
    for bit in range(b):
        candidates = [j for j in range(rank, len(rows)) if rows[j] >> bit & 1]
        if not candidates:
            continue
        j = candidates[0]
        rows[rank], rows[j] = rows[j], rows[rank]
        for j in range(len(rows)):
            if j != rank and rows[j] >> bit & 1:
                rows[j] ^= rows[rank]
        pivots.append(bit)
        rank += 1
    return [(1 << free) | sum(((row >> free) & 1) << p for p, row in zip(pivots, rows))
            for free in range(b) if free not in pivots]


def span(keys):
    values = [0]
    for key in keys:
        values += [x ^ key for x in values]
    return sorted(set(values))


def characters(b):
    return np.array([[1 - 2 * parity(g & v) for v in range(1 << b)] for g in range(1 << b)], dtype=np.int16)


def ambiguous_classes(signs, k):
    words = np.array(list(itertools.combinations_with_replacement(range(len(signs)), k)), dtype=np.uint8)
    sums = signs[words].sum(1).astype(np.int8)
    _, inverse, sizes = np.unique(sums, axis=0, return_inverse=True, return_counts=True)
    groups = defaultdict(list)
    for j in np.flatnonzero(sizes[inverse] > 1):
        groups[int(inverse[j])].append(tuple(map(int, words[j])))
    classes = {tuple(sorted(group)) for group in groups.values()}
    return classes, {"multisets": len(words), "distinct_sums": len(sizes),
                     "ambiguous_class_sizes": dict(sorted(Counter(map(len, classes)).items()))}


def affine_census(keys, h):
    v = h.shape[1]
    planes = sorted({tuple(sorted((0, x, y, x ^ y))) for x in range(1, v) for y in range(x + 1, v)})
    classes, holes, witnesses = set(), Counter(), {}
    distance_error = 0.
    energy = np.random.default_rng(29).dirichlet(np.ones(len(keys)))
    bad_planes = 0
    for u in planes:
        cosets = sorted({tuple(sorted(a ^ x for x in u)) for a in range(v)})
        sums = np.array([h[:, c].sum(1) for c in cosets])
        buckets = defaultdict(list)
        for c, s in zip(cosets, sums):
            buckets[tuple(s[keys])].append(c)
        bad_planes += any(len(group) > 1 for group in buckets.values())
        for group in buckets.values():
            if len(group) > 1:
                classes.add(tuple(sorted(group)))
            for a, c in itertools.combinations(group, 2):
                f = tuple(map(int, np.flatnonzero(h[:, a].sum(1) - h[:, c].sum(1))))
                holes[f] += 1
                witnesses[f] = (a, c)
        for j, k in itertools.combinations(range(len(cosets)), 2):
            d = cosets[j][0] ^ cosets[k][0]
            f = [g for g in range(v) if all(parity(g & x) == 0 for x in u) and parity(g & d)]
            delta = sums[j] - sums[k]
            assert set(np.flatnonzero(delta)) == set(f)
            assert np.all(np.abs(delta[f]) == 8)
            direct = float((delta[keys] ** 2) @ energy)
            predicted = 64 * sum(e for g, e in zip(keys, energy) if g in f)
            distance_error = max(distance_error, abs(direct - predicted))
    assert len(planes) == 651 and len(holes) == 58 and set(holes.values()) == {8}
    assert distance_error < 1e-12
    return classes, sorted(holes), witnesses, {"planes": len(planes), "bad_planes": bad_planes,
                                             "alias_pairs": sum(holes.values()), "missing_flats": len(holes),
                                             "distance_identity_max_error": distance_error}


def repair_certificate(keys, h, classes, holes):
    missing = sorted(set(range(h.shape[0])) - set(keys))
    masks = {g: sum(1 << i for i, f in enumerate(holes) if g in f) for g in missing}
    maxima, best = {}, {}
    # Every <=6-element set extends to six, so this is an integer lower-bound certificate.
    for k in (3, 6):
        top = (-1, ())
        for choice in itertools.combinations(missing, k):
            mask = 0
            for g in choice:
                mask |= masks[g]
            if mask.bit_count() > top[0]:
                top = (mask.bit_count(), choice)
        maxima[k], best[k] = top
    addition = [8, 13, 19, 52, 60, 61, 63]
    assert maxima == {3: 40, 6: 55}
    assert all(set(f).intersection(addition) for f in holes)
    replacement, seen, new = list(keys), set(), iter(best[3])
    for i, g in enumerate(keys):
        if g in seen:
            replacement[i] = next(new)
        seen.add(g)
    assert set(keys).issubset(replacement)
    split = []
    for group in classes:
        buckets = defaultdict(list)
        for c in group:
            buckets[tuple(h[replacement][:, c].sum(1))].append(c)
        split.extend(x for x in buckets.values() if len(x) > 1)
    return {"maximum_holes_hit": maxima, "maximizers": best, "six_subsets_checked": math.comb(len(missing), 6),
            "minimum_additions": len(addition), "seven_row_witness": addition,
            "three_replacement_class_sizes": dict(sorted(Counter(map(len, split)).items())),
            "three_replacement_ambiguous_multisets": sum(map(len, split)),
            "three_replacement_min_hamming": int(np.sum(h[replacement, 1:] < 0, axis=0).min())}


def counterexamples(keys):
    g = [0, 1, 2, 5, 6, 7, 9, 10, 11, 15]
    h = characters(4)
    assert len(basis(g)) == 4
    assert all(len(basis([x for x in g if not parity(x & d)])) == 3 for d in range(1, 16))
    pair_classes, pair = ambiguous_classes(h[g].T, 2)
    assert not pair_classes
    np.testing.assert_array_equal(h[g][:, [0, 1, 2]].sum(1), h[g][:, [4, 8, 15]].sum(1))
    plain = PolarCode(6, 32, 0)
    signs = 1 - 2 * plain.encode(message_bits(np.arange(64), 6)).astype(np.int16)
    assert sum(np.array_equal(signs[a] + signs[c], np.zeros(32)) for a, c in itertools.combinations(range(64), 2)) == 32
    complete = []
    observed = set(keys)

    def extend(chosen, candidates):
        if len(chosen) == 6:
            complete.append(chosen)
        for i, x in enumerate(candidates):
            if len(basis(chosen + [x])) == len(chosen) + 1:
                extend(chosen + [x], [y for y in candidates[i + 1:] if x ^ y in observed])

    extend([], sorted(observed - {0}))
    assert not complete
    return {"pair_injective_triple_alias_rows": g, "counterexample_pair_sums": pair["distinct_sums"],
            "triple_alias": [[0, 1, 2], [4, 8, 15]], "crc0_complement_pairs": 32,
            "selected_quadratic_complete_bases": len(complete)}


def noise_and_placement(keys, h):
    base = ODMAPolarBaseline(12, 256, 3201, prefix_bits=6, code_length=32, crc_bits=8, list_size=8)
    e, info = balanced_energy(base.positions, 256)
    words = np.array(list(itertools.combinations_with_replacement(range(64), 2)))
    sums = h[keys].T[words].sum(1)
    upper = np.triu_indices(len(words), 1)

    def distances(energy):
        x = sums * np.sqrt(energy)
        norm = np.sum(x * x, axis=1)
        return (norm[:, None] + norm[None, :] - 2 * x @ x.T)[upper]

    du, db = distances(np.full(32, 1 / 32)), distances(e[0])
    selected = int(np.argmin(db))
    a, c = words[upper[0][selected]], words[upper[1][selected]]
    delta = h[keys][:, a].sum(1) - h[keys][:, c].sum(1)
    np.testing.assert_allclose(db[selected], np.dot(e[0], delta ** 2), atol=1e-12)
    rng = np.random.default_rng(53)
    y, sigma2 = rng.normal(size=32), .7
    direct = -np.sum((y - np.sqrt(e[0]) * sums) ** 2, axis=1) / (2 * sigma2)
    compressed = np.zeros(len(words))
    for g in set(keys):
        sites = np.flatnonzero(np.array(keys) == g)
        mass = e[0, sites].sum()
        stat = np.dot(np.sqrt(e[0, sites]), y[sites]) / mass
        compressed -= mass * (stat - sums[:, sites[0]]) ** 2 / (2 * sigma2)
    np.testing.assert_allclose(direct - direct[0], compressed - compressed[0], atol=1e-12)
    replacement = list(keys)
    replacement[17], replacement[21], replacement[30] = 8, 20, 61
    sums = h[replacement].T[words].sum(1)
    ru, rb = distances(np.full(32, 1 / 32)), distances(e[0])
    np.testing.assert_allclose([ru.min(), rb.min()], [1.5, 1.271656876697424], atol=1e-12)
    # Distinct supports; each pattern separately identifies every fixed-K b=1 count vector.
    positions = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
    mixed_distances = []
    for x in (.5, .8):
        energy = np.array([[x, 1 - x], [1 - x, x], [1 - x, x], [x, 1 - x]])
        np.testing.assert_allclose(np.bincount(positions.ravel(), weights=energy.ravel()), np.ones(4))
        phi = np.zeros((4, 4))
        for p in range(4):
            phi[p, positions[p]] = np.sqrt(energy[p])
        mixed_distances.append(float(np.sum((phi[0] + phi[1] - phi[2] - phi[3]) ** 2)))
    assert mixed_distances[0] == 0 and mixed_distances[1] > 0
    return {"fixture": "B12/n256/seed3201/known pattern0/K2", "balance": info,
            "comparisons": len(du), "uniform_min_squared_distance": float(du.min()),
            "balanced_min_squared_distance": float(db.min()), "balanced_closest_pair": [a.tolist(), c.tolist()],
            "increased_distances": int(np.sum(db > du + 1e-10)), "decreased_distances": int(np.sum(db < du - 1e-10)),
            "three_replacement_uniform_min_squared_distance": float(ru.min()),
            "three_replacement_balanced_min_squared_distance": float(rb.min()),
            "balanced_rectangle_distances": mixed_distances}


def orbit_checks(keys, h):
    a, u = 12, [0, 3, 53, 54]
    q = [g for g in keys if all(not parity(g & x) for x in u)]
    w_basis = null_basis(q, 6)
    w = span(w_basis)
    assert set(w) == {d for d in range(64) if all(not parity(g & d) for g in q)}
    orbit = sorted({tuple(sorted(a ^ d ^ x for x in u)) for d in w})
    for c in orbit:
        np.testing.assert_array_equal(h[keys][:, c].sum(1), h[keys][:, orbit[0]].sum(1))
    outside = Counter({12: 2, 1: 1})
    states = [outside + Counter(c) for c in orbit]
    support = {a ^ d for d in w}
    for v in range(64):
        for j in range(1, 8):
            direct = np.mean([s[v] >= j for s in states])
            analytic = float(outside[v] >= j) + float(outside[v] == j - 1 and v in support) / len(orbit)
            assert direct == analytic
    # Count-space prior changes when outside copies coincide, although the ordered-slot move is symmetric.
    factors = [1 / math.prod(math.factorial(s[v]) for v in s) for s in states]
    assert len(set(factors)) > 1
    slot_choices = [math.prod(s[v] for v in c) for s, c in zip(states, orbit)]
    np.testing.assert_allclose(np.array(factors) * slot_choices, np.full(len(orbit), .5))
    # A uniform shift followed by rejecting outside duplicates is symmetric for the distinct source.
    allowed = [c for c in orbit if not set(c).intersection(outside)]
    assert allowed
    # Reject untested small-alphabet shortcuts: the nullspace routine also works with 100-bit payload IDs.
    high_u = [1 << 98, 1 << 99]
    high_q = [1 << j for j in range(97)]
    high_w = null_basis(high_q, 100)
    assert len(high_w) == 3 and all(not parity(g & d) for g in high_q for d in high_w)
    assert len(span(high_u)) == 4 and len(span(high_w)) == 8
    return {"orbit_cosets": orbit, "nullity": len(w_basis), "tail_formula_checked": True,
            "outside_count_prior_weights": factors, "distinct_allowed_cosets": len(allowed),
            "b100_nullity": len(high_w)}


def packet_and_near_alias_checks():
    h, b, q = characters(4), 4, 2
    hash_rows, offsets = [1, 2], [0, 4, 8]
    counts = np.zeros(16, dtype=int)
    counts[[1, 5, 10]] = [2, 1, 3]
    bin_id = np.array([(v & 1) + 2 * ((v >> 1) & 1) for v in range(16)])
    small = characters(q)
    packets = []
    for d in offsets:
        rows = [d ^ ((t & 1) * hash_rows[0]) ^ (((t >> 1) & 1) * hash_rows[1]) for t in range(4)]
        observed = h[rows] @ counts
        bins = small @ observed / 4
        expected = np.array([sum(counts[v] * h[d, v] for v in range(16) if bin_id[v] == j) for j in range(4)])
        np.testing.assert_array_equal(bins, expected)
        packets.append(bins)
    packets = np.array(packets)
    singletons = [j for j in range(4) if packets[0, j] > 0 and np.all(np.abs(packets[1:, j]) == packets[0, j])]
    assert singletons == [2] and packets[0, 2] == 3
    nc, sigma2 = len(offsets) * (1 << q), .7
    noise_map = small * np.sqrt(nc) / (1 << q)
    np.testing.assert_allclose(sigma2 * noise_map @ noise_map.T, np.eye(4) * sigma2 * nc / 4, atol=1e-14)
    rng = np.random.default_rng(71)
    worst_tv = 0.
    for _ in range(20):
        signals = rng.normal(size=(7, 5)) * .08
        y = rng.normal(size=5)
        prior = rng.dirichlet(np.ones(7))
        logs = -np.sum((signals - y) ** 2, axis=1) / (2 * sigma2)
        p = prior * np.exp(logs - logs.max())
        p /= p.sum()
        radius = np.linalg.norm(signals, axis=1).max()
        omega = (2 * np.linalg.norm(y) * radius + radius ** 2) / sigma2
        tv = np.abs(p - prior).sum() / 2
        assert np.ptp(logs) <= omega + 1e-12 and tv <= np.tanh(omega / 4) + 1e-12
        worst_tv = max(worst_tv, float(tv))
    return {"packet_payload_bits": b, "bin_bits": q, "repeated_singleton_count": 3,
            "bin_noise_variance": sigma2 * nc / 4, "near_alias_fixtures": 20, "largest_conditional_tv": worst_tv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    code = PolarCode(6, 32, 8)
    generator = code.encode(np.eye(6, dtype=np.uint8)).T
    keys = list(map(int, generator @ (1 << np.arange(5, -1, -1))))
    h = characters(6)
    signs = 1 - 2 * code.encode(message_bits(np.arange(64), 6)).astype(np.int16)
    np.testing.assert_array_equal(signs, h[keys].T)
    classes, fourth = ambiguous_classes(signs, 4)
    third_classes, third = ambiguous_classes(signs, 3)
    assert not third_classes
    affine, holes, _, geometry = affine_census(keys, h)
    assert classes == affine and fourth["ambiguous_class_sizes"] == {2: 392, 4: 12}
    result = {"model": "known-pattern linear signs; IID counts and distinct control; unit energy/AWGN",
              "generator_rows_msb": keys, "K3": third, "K4": fourth, "affine": geometry,
              "repair": repair_certificate(keys, h, classes, holes), "counterexamples": counterexamples(keys),
              "noise": noise_and_placement(keys, h), "orbit": orbit_checks(keys, h),
              "packets_and_approximation": packet_and_near_alias_checks()}
    result["K26_witness_union_bounds"] = {
        p: {"iid": math.comb(26, 4) / p ** 3 * 832 * 24 / 64 ** 4,
            "distinct": math.comb(26, 4) * p * 832 * 24 / math.prod(range(p * 64 - 3, p * 64 + 1))}
        for p in (64, 256)}
    result["seconds"] = time.monotonic() - start
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
